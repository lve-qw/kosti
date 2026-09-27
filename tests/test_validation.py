from kosti.contracts import HIP, Prediction, VIOLATIONS
from kosti import validation
from test_dicom import write_dicom
from test_training import initial_weights, run_inputs


class Predictor:
    def predict(self, image):
        return Prediction(HIP, .2, dict.fromkeys(VIOLATIONS[HIP], .2))


def test_benchmark_measures_real_files_and_repeatability(tmp_path, monkeypatch):
    data = tmp_path / 'input'; data.mkdir()
    write_dicom(data / 'one.dcm')
    model = tmp_path / 'manifest.json'; model.write_text('{}')
    monkeypatch.setattr(validation, 'load_quality_predictor', lambda path: Predictor())
    report = validation.benchmark(model, data, tmp_path / 'benchmark.json')
    assert report['passed_on_this_machine'] and report['reproducible_outputs']
    assert len(report['runs']) == 2
    assert all(r['files'] == 1 and r['studies'] == 1 and r['failures'] == 0 for r in report['runs'])
    assert report['model_load_seconds'] >= 0 and report['process_peak_rss_mib'] > 0


def test_benchmark_failure_is_not_a_pass(tmp_path, monkeypatch):
    data = tmp_path / 'input'; data.mkdir()
    (data / 'broken.dcm').write_bytes(b'broken')
    model = tmp_path / 'manifest.json'; model.write_text('{}')
    monkeypatch.setattr(validation, 'load_quality_predictor', lambda path: Predictor())
    report = validation.benchmark(model, data, tmp_path / 'benchmark.json')
    assert not report['passed_on_this_machine']
    assert all(r['failures'] == 1 for r in report['runs'])


def test_benchmark_detects_unstable_output(tmp_path, monkeypatch):
    data = tmp_path / 'input'; data.mkdir()
    write_dicom(data / 'one.dcm')
    model = tmp_path / 'manifest.json'; model.write_text('{}')
    class Unstable:
        count = 0
        def predict(self, image):
            self.count += 1
            return Prediction(HIP, self.count * .1, dict.fromkeys(VIOLATIONS[HIP], .2))
    monkeypatch.setattr(validation, 'load_quality_predictor', lambda path: Unstable())
    report = validation.benchmark(model, data, tmp_path / 'benchmark.json')
    assert not report['reproducible_outputs'] and not report['passed_on_this_machine']


def test_oof_uses_bundle_thresholds_and_preserves_probability_auc(run_inputs, monkeypatch):
    import json
    from kosti.contracts import SPINE
    path, _, manifest = run_inputs
    run = path.parent / 'run'
    run.mkdir()
    (run / 'completed.json').write_text('{"status":"complete","mode":"cross_validation"}')
    (run / 'splits.json').write_text(json.dumps(manifest))
    by_hash = {s['pixel_hash']: s for s in manifest['samples']}

    class ThresholdPredictor:
        def predict(self, image):
            sample = by_hash[image.pixel_hash]
            region = SPINE if sample['region'] == 'spine' else HIP
            positive = sample['labels'][sample['region'] + '_quality']
            probability = .4 if positive else .2
            return Prediction(region, probability, dict.fromkeys(VIOLATIONS[region], probability),
                              dict.fromkeys(VIOLATIONS[region], .3))

    monkeypatch.setattr(validation, 'load_quality_predictor', lambda path: ThresholdPredictor())
    result = validation.validate_oof_models(path.parent / 'splits.json', run, path.parent,
                                            path.parent / 'report.json', bootstrap_repeats=10)
    assert result['violations']['macro_f1'] == 1
    assert result['quality']['decision']['f1'] == 1
    assert result['errors'] == []
    assert all(m['roc_auc'] == 1 for m in result['violations']['per_label'].values())


def test_oof_rejects_changed_labels(run_inputs, monkeypatch):
    import json
    import pytest
    path, _, manifest = run_inputs
    run = path.parent / 'run'
    run.mkdir()
    (run / 'completed.json').write_text('{"status":"complete","mode":"cross_validation"}')
    manifest['samples'][0]['labels']['spine_axis'] = None
    (run / 'splits.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='splits differ'):
        validation.validate_oof_models(path.parent / 'splits.json', run, path.parent,
                                       path.parent / 'report.json')
