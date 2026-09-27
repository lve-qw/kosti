"""Measured runtime and end-to-end validation of explicit trained bundles."""
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import platform
import resource
import time

import numpy as np

from .contracts import HIP, SPINE, VIOLATIONS
from .dicom import read_dicom
from .evaluation import binary_metrics, calibration_metrics, grouped_bootstrap_ci, linked_study_groups, multilabel_metrics
from .ml import load_quality_predictor
from .pipeline import process_files
from .preparation import LABELS, write_json
from .training import load_manifest, sha256


def benchmark(model_path, input_root, output, repeats=2):
    if type(repeats) is not int or not 2 <= repeats <= 10:
        raise ValueError('Benchmark needs 2 to 10 repetitions')
    root = Path(input_root).resolve()
    paths = sorted(p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in ('.dcm', '.dicom'))
    if not paths:
        raise ValueError('No DICOM files for benchmark')
    started = time.perf_counter()
    predictor = load_quality_predictor(model_path)
    load_seconds = time.perf_counter() - started
    runs, reference, stable = [], None, True
    for _ in range(repeats):
        started = time.perf_counter()
        rows, diagnostics = process_files(paths, predictor, display_root=root)
        wall = time.perf_counter() - started
        study_times = defaultdict(float)
        current = []
        for row in rows:
            study_times[row.study_uid or row.path_to_study] += row.time_of_processing
            item = asdict(row)
            item.pop('time_of_processing')
            current.append(item)
        if reference is None:
            reference = current
        else:
            stable = stable and reference == current
        runs.append({'wall_seconds': wall, 'files': len(rows),
                     'failures': sum(r.processing_status != 'Success' for r in rows),
                     'studies': len(study_times), 'max_study_seconds': max(study_times.values()),
                     'p95_study_seconds': float(np.percentile(list(study_times.values()), 95)),
                     'mean_file_seconds': wall / len(rows), 'diagnostic_entries': len(diagnostics)})
    memory = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_mib = memory / (1024 * 1024 if platform.system() == 'Darwin' else 1024)
    report = {'schema_version': 1, 'model_manifest_sha256': sha256(model_path),
              'platform': platform.platform(), 'processor': platform.processor(),
              'python': platform.python_version(), 'device': 'cpu',
              'model_load_seconds': load_seconds, 'process_peak_rss_mib': peak_mib,
              'reproducible_outputs': stable, 'runs': runs,
              'passed_on_this_machine': stable and all(r['failures'] == 0 and r['max_study_seconds'] <= 180 for r in runs),
              'scope': 'Measured local CPU only; per-study sum of all supplied files; load time reported separately.'}
    write_json(output, report)
    return report


def validate_oof_models(manifest_path, run_root, data_root, output, bootstrap_repeats=1000):
    """Use only each sample's held-out fold model, including predicted anatomy."""
    manifest = load_manifest(manifest_path)
    samples, root = manifest['samples'], Path(data_root).resolve()
    run_root = Path(run_root)
    completed = run_root / 'completed.json'
    if not completed.is_file():
        raise ValueError('Training run has no completion marker')
    import json
    completion = json.loads(completed.read_text())
    if completion.get('status') != 'complete' or completion.get('mode') != 'cross_validation' or (run_root / 'failed.json').exists():
        raise ValueError('Evaluation requires a completed cross-validation run')
    saved_splits = load_manifest(run_root / 'splits.json')
    if saved_splits != manifest:
        raise ValueError('Training and evaluation splits differ')
    groups = linked_study_groups(samples)
    predictions = {}
    for fold in range(manifest['n_splits']):
        predictor = load_quality_predictor(run_root / f'fold-{fold}' / 'manifest.json')
        for sample in samples:
            if sample['fold'] != fold:
                continue
            path = (root / sample['path']).resolve()
            if not path.is_relative_to(root):
                raise ValueError('Sample path escapes data root')
            image = read_dicom(path)
            if image.pixel_hash != sample['pixel_hash']:
                raise ValueError('Evaluation DICOM differs from frozen manifest')
            predictions[sample['pixel_hash']] = predictor.predict(image)
    y_types, p_types, type_decisions = [], [], []
    actual_quality, predicted_quality, quality_score, quality_groups = [], [], [], []
    anatomy_correct, errors, by_region = [], [], defaultdict(lambda: {'y': [], 'decision': [], 'score': [], 'groups': []})
    for sample, group in zip(samples, groups):
        predicted = predictions[sample['pixel_hash']]
        region = SPINE if sample['region'] == 'spine' else HIP
        anatomy_correct.append(predicted.anatomical_region == region)
        probabilities = []
        decisions = []
        for r in (SPINE, HIP):
            probabilities.extend(predicted.violation_probabilities[name] if r == predicted.anatomical_region else 0.
                                 for name in VIOLATIONS[r])
            decisions.extend(int(predicted.violation_probabilities[name] >= predicted.thresholds.get(name, .5))
                             if r == predicted.anatomical_region else 0 for name in VIOLATIONS[r])
        targets = [np.nan if sample['labels'][k] is None else sample['labels'][k] for k in LABELS[:5]]
        y_types.append(targets); p_types.append(probabilities)
        type_decisions.append(decisions)
        decision = int(any(p >= predicted.thresholds.get(k, .5) for k, p in predicted.violation_probabilities.items()))
        actual = sample['labels'][sample['region'] + '_quality']
        if actual is not None:
            actual_quality.append(actual); predicted_quality.append(decision)
            quality_score.append(predicted.quality_prob); quality_groups.append(group)
            subset = by_region[sample['region']]
            subset['y'].append(actual); subset['decision'].append(decision)
            subset['score'].append(predicted.quality_prob); subset['groups'].append(group)
        mismatches = [name for name, y, decision_type in zip(LABELS[:5], targets, decisions)
                      if np.isfinite(y) and decision_type != y]
        if not anatomy_correct[-1] or mismatches or actual is not None and actual != decision:
            errors.append({'pixel_hash': sample['pixel_hash'], 'fold': sample['fold'],
                           'true_region': sample['region'], 'predicted_region': predicted.anatomical_region,
                           'mismatched_types': mismatches, 'actual_quality': actual, 'predicted_quality': decision})
    def quality_metrics(y, decision, score, unit_groups):
        return {'decision': binary_metrics(y, decision), 'risk_score': binary_metrics(y, score),
                'calibration': calibration_metrics(y, score),
                'f1_ci95': grouped_bootstrap_ci(y, decision, unit_groups, repeats=bootstrap_repeats, seed=42),
                'auc_ci95': grouped_bootstrap_ci(y, score, unit_groups, metric='roc_auc', repeats=bootstrap_repeats, seed=43)}
    violation_metrics = multilabel_metrics(y_types, type_decisions, LABELS[:5])
    score_metrics = multilabel_metrics(y_types, p_types, LABELS[:5])
    for name in LABELS[:5]:
        for key in ('roc_auc', 'average_precision'):
            violation_metrics['per_label'][name][key] = score_metrics['per_label'][name][key]
    report = {'schema_version': 1, 'protocol': 'held-out models; predicted anatomy; thresholds from each model bundle',
              'images': len(samples), 'anatomy_accuracy': float(np.mean(anatomy_correct)),
              'violations': violation_metrics,
              'quality': quality_metrics(actual_quality, predicted_quality, quality_score, quality_groups),
              'quality_by_region': {k: quality_metrics(v['y'], v['decision'], v['score'], v['groups']) for k, v in by_region.items()},
              'errors': errors, 'limitations': ['Image indexing was reviewed by an AI assistant, not independently by a clinical expert.',
                  'Small development cohort, grouped by study and exact pixels; independent patient IDs unavailable.',
                  'This is internal cross-validation, not external clinical validation.']}
    write_json(output, report)
    return report
