"""Explicit reviewed annotations and leakage-safe training manifests; no training."""
from __future__ import annotations

import csv
import json
from pathlib import Path, PurePosixPath

from .evaluation import binary_metrics, grouped_bootstrap_ci, grouped_folds, linked_study_groups, multilabel_metrics

LABELS = ('spine_position', 'spine_axis', 'spine_artifact',
          'hip_position', 'hip_roi', 'spine_quality', 'hip_quality')
IDENTITY = ('path', 'study_uid', 'image_uid', 'pixel_hash')
FIELDS = (*IDENTITY, 'region', 'side', 'label_source', 'reviewed', *LABELS)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Never overwrite an annotation, report, or frozen split by accident.
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')


def audit_records(audit):
    records = audit.get('manifest')
    if not isinstance(records, list) or not records:
        raise ValueError('Audit must contain a nonempty manifest')
    seen = set()
    for row in records:
        if not isinstance(row, dict) or any(not isinstance(row.get(k), str) or not row[k].strip() for k in IDENTITY):
            raise ValueError('Audit requires path, study_uid, image_uid and pixel_hash')
        path = PurePosixPath(row['path'])
        if path.is_absolute() or '..' in path.parts or '\\' in row['path'] or row['path'] in seen:
            raise ValueError('Audit paths must be unique relative paths without traversal')
        seen.add(row['path'])
    return records


def annotation_template(audit_path, output):
    records = audit_records(read_json(audit_path))
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        for row in records:
            writer.writerow({**{k: row[k] for k in IDENTITY}, 'reviewed': '0'})
    return len(records)


def prepare_folds(audit_path, annotations_path, output, n_splits=3, seed=42):
    audit = read_json(audit_path)
    records = audit_records(audit)
    if audit.get('failed_count', 0) or audit.get('failures'):
        raise ValueError('Resolve DICOM audit failures before freezing folds')
    by_path = {row['path']: row for row in records}
    reviewed = {}
    with Path(annotations_path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != list(FIELDS):
            raise ValueError('Annotation columns must match the generated template exactly')
        for row in reader:
            path = row['path']
            if None in row or any(v is None for v in row.values()):
                raise ValueError('Malformed annotation CSV row')
            if path not in by_path or path in reviewed:
                raise ValueError('Unknown or duplicate annotation path')
            if any(row[k] != by_path[path][k] for k in IDENTITY):
                raise ValueError('Annotation identity differs from audit: ' + path)
            if row['reviewed'] != '1' or not row['label_source'].strip():
                raise ValueError('Every image requires reviewed=1 and label_source: ' + path)
            region, side = row['region'], row['side']
            if region not in ('spine', 'hip') or (region == 'spine' and side != '') or (region == 'hip' and side not in ('left', 'right')):
                raise ValueError('Use spine with empty side, or hip with left/right: ' + path)
            labels = {}
            for name in LABELS:
                value = row[name].strip()
                if value not in ('', '0', '1'):
                    raise ValueError('Labels must be empty, 0 or 1: ' + path)
                labels[name] = None if value == '' else int(value)
                if not name.startswith(region + '_') and labels[name] is not None:
                    raise ValueError('Inapplicable labels must remain empty: ' + path)
            parts = [labels[k] for k in LABELS if k.startswith(region + '_') and not k.endswith('_quality')]
            quality = labels[region + '_quality']
            if quality is not None and ((quality == 0 and 1 in parts) or
                                        (all(v is not None for v in parts) and quality != int(any(parts)))):
                raise ValueError('Conflicting quality and violation labels require review: ' + path)
            reviewed[path] = {**{k: row[k] for k in IDENTITY}, 'region': region, 'side': side,
                              'label_source': row['label_source'], 'labels': labels}
    if set(reviewed) != set(by_path):
        raise ValueError('Annotations must cover every audited image')
    rows = [reviewed[p] for p in sorted(reviewed)]
    # All copies take part in unioning studies BEFORE any training deduplication.
    folds = grouped_folds([r['study_uid'] for r in rows], [r['pixel_hash'] for r in rows], n_splits, seed)
    unique = {}
    for row, fold in zip(rows, folds):
        row['fold'] = int(fold)
        previous = unique.get(row['pixel_hash'])
        if previous is not None:
            if any(row[k] != previous[k] for k in ('region', 'side', 'labels')):
                raise ValueError('Duplicate pixels have inconsistent annotations: ' + row['path'])
            previous['copies'].append({k: row[k] for k in IDENTITY})
        else:
            unique[row['pixel_hash']] = {**row, 'copies': [{k: row[k] for k in IDENTITY}]}
    samples = list(unique.values())
    counts = []
    for fold in range(n_splits):
        subset = [s for s in samples if s['fold'] == fold]
        counts.append({'fold': fold, 'images': len(subset), 'labels': {
            k: {'observed': sum(s['labels'][k] is not None for s in subset),
                'positive': sum(s['labels'][k] == 1 for s in subset)} for k in LABELS}})
    result = {'schema_version': 1, 'seed': seed, 'n_splits': n_splits,
              'label_names': list(LABELS), 'samples': samples, 'fold_counts': counts,
              'source_images': len(rows), 'unique_images': len(samples),
              'warnings': ['Folds are grouped, not stratified. Inspect rare-label counts before training.',
                           'Study grouping cannot guarantee patient-level independence.']}
    write_json(output, result)
    return result


def evaluate_predictions(manifest_path, predictions_path, output, bootstrap_repeats=0, seed=42):
    """Evaluate supplied held-out probabilities, never fit thresholds on OOF labels."""
    if type(bootstrap_repeats) is not int or not 0 <= bootstrap_repeats <= 10000:
        raise ValueError('bootstrap_repeats must be an integer in [0, 10000]')
    if type(seed) is not int or seed < 0:
        raise ValueError('bootstrap seed must be a nonnegative integer')
    manifest = read_json(manifest_path)
    if manifest.get('schema_version') != 1 or manifest.get('label_names') != list(LABELS):
        raise ValueError('Unsupported prepared manifest')
    samples = manifest.get('samples', [])
    if not samples or len({s['pixel_hash'] for s in samples}) != len(samples):
        raise ValueError('Manifest must have unique nonempty samples')
    for sample in samples:
        representative = {key: sample[key] for key in IDENTITY}
        if representative not in sample.get('copies', []):
            raise ValueError('Duplicate provenance must include its representative image')
    predictions = {}
    with Path(predictions_path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if reader.fieldnames != ['pixel_hash', 'fold', *LABELS]:
            raise ValueError('Prediction columns must be pixel_hash, fold, then label_names')
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError('Malformed prediction CSV row')
            key = row['pixel_hash']
            if key in predictions:
                raise ValueError('Duplicate prediction pixel_hash')
            predictions[key] = row
    if set(predictions) != {s['pixel_hash'] for s in samples}:
        raise ValueError('Predictions must cover exactly the prepared samples')
    targets, probabilities = [], []
    for sample in samples:
        row = predictions[sample['pixel_hash']]
        if row['fold'] != str(sample['fold']):
            raise ValueError('Prediction fold differs from frozen split')
        targets.append([float('nan') if sample['labels'][k] is None else sample['labels'][k] for k in LABELS])
        probabilities.append([float(row[k]) for k in LABELS])
    groups = linked_study_groups(samples)
    group_folds = {}
    for sample, group in zip(samples, groups):
        previous_fold = group_folds.setdefault(group, sample['fold'])
        if previous_fold != sample['fold']:
            raise ValueError('Linked studies or exact copies span multiple folds')

    def interval(y, p, unit_groups, metric, offset):
        if bootstrap_repeats == 0:
            return None
        return grouped_bootstrap_ci(y, p, unit_groups, metric=metric,
                                    repeats=bootstrap_repeats, seed=seed + offset)

    # The binary result exported by the API is OR over applicable violation heads.
    # The separate quality head supplies the risk score used for ROC AUC.
    quality = {}
    for region in ('spine', 'hip', 'all'):
        indices = [i for i, sample in enumerate(samples) if region == 'all' or sample['region'] == region]
        if not indices:
            continue
        known, head_scores, or_decisions, region_groups = [], [], [], []
        for i in indices:
            sample = samples[i]
            actual = sample['labels'][sample['region'] + '_quality']
            if actual is None:
                continue
            row = predictions[sample['pixel_hash']]
            applicable = ('spine_position', 'spine_axis', 'spine_artifact') if sample['region'] == 'spine' else ('hip_position', 'hip_roi')
            known.append(actual)
            head_scores.append(float(row[sample['region'] + '_quality']))
            or_decisions.append(int(any(float(row[name]) >= 0.5 for name in applicable)))
            region_groups.append(groups[i])
        if not known:
            continue
        quality[region] = {
            'quality_head': binary_metrics(known, head_scores),
            'quality_class_or': binary_metrics(known, or_decisions),
            'independent_groups': len(set(region_groups)),
        }
        if bootstrap_repeats:
            quality[region]['confidence_intervals_95'] = {
                'quality_head_roc_auc': interval(known, head_scores, region_groups, 'roc_auc', 100 + len(quality)),
                'quality_class_or_f1': interval(known, or_decisions, region_groups, 'f1', 200 + len(quality)),
            }

    report = {'schema_version': 1, 'threshold': 0.5,
              'metrics': multilabel_metrics(targets, probabilities, LABELS),
              'violations': multilabel_metrics([r[:5] for r in targets], [r[:5] for r in probabilities], LABELS[:5]),
              'quality_by_region': quality,
              'bootstrap': {'repeats': bootstrap_repeats, 'seed': seed, 'unit': 'linked studies and exact pixel copies'},
              'warnings': ['Fold IDs are checked; this cannot prove the model did not train on held-out images.',
                           'Threshold is fixed at 0.5; no calibration is performed.']}
    if bootstrap_repeats:
        report['label_confidence_intervals_95'] = {
            name: {'f1': interval([row[i] for row in targets], [row[i] for row in probabilities], groups, 'f1', 300 + i),
                   'roc_auc': interval([row[i] for row in targets], [row[i] for row in probabilities], groups, 'roc_auc', 400 + i)}
            for i, name in enumerate(LABELS)
        }
    write_json(output, report)
    return report
