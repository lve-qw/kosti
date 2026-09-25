"""Training contracts and orchestration; no optimizer steps or model training."""
import csv
import json

import numpy as np
import pytest

from kosti import training
from kosti.cli import main
from kosti.dicom import read_dicom
from kosti.ml import load_quality_predictor
from kosti.preparation import IDENTITY, LABELS
from test_dicom import write_dicom
from test_ml import metadata


@pytest.fixture(scope='module')
def initial_weights(tmp_path_factory):
    torch = pytest.importorskip('torch')
    from torchvision.models import resnet18
    torch.set_num_threads(1)
    path = tmp_path_factory.mktemp('synthetic-initializer') / 'initial.pth'
    torch.save(resnet18(weights=None).state_dict(), path)
    return path


@pytest.fixture
def run_inputs(tmp_path, initial_weights):
    samples = []
    for fold in range(3):
        for region in ('spine', 'hip'):
            for positive in (0, 1):
                path = tmp_path / f'{len(samples)}.dcm'
                write_dicom(path, np.arange(64, dtype=np.uint16).reshape(8, 8) + len(samples))
                image = read_dicom(path)
                sample = {k: getattr(image, k) for k in IDENTITY if k != 'path'}
                sample.update(path=path.name, region=region, side='' if region == 'spine' else 'left',
                              label_source='synthetic-test', fold=fold,
                              labels={k: positive if k.startswith(region + '_') else None for k in LABELS})
                sample['copies'] = [{k: sample[k] for k in IDENTITY}]
                samples.append(sample)
    manifest = {'schema_version': 1, 'label_names': list(LABELS), 'samples': samples, 'n_splits': 3}
    (tmp_path / 'splits.json').write_text(json.dumps(manifest))
    config = dict(schema_version=1, manifest='splits.json', data_root='.', output='run',
                  mode='cross_validation', seed=42, epochs=2, batch_size=3,
                  learning_rate=0.001, weight_decay=0.01, device='cpu', freeze_backbone=True,
                  initial_weights=str(initial_weights), initial_weights_sha256=training.sha256(initial_weights),
                  preprocessing=metadata()['preprocessing'])
    path = tmp_path / 'config.json'
    path.write_text(json.dumps(config))
    return path, config, manifest


def test_dry_run_does_not_optimize_or_write(run_inputs, monkeypatch, capsys):
    path, _, _ = run_inputs
    def forbidden(*args, **kwargs):
        pytest.fail('Dry run attempted training')
    monkeypatch.setattr(training, 'fit_model', forbidden)
    assert main(['train', '--config', str(path)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result['status'] == 'validated_without_training'
    assert len(result['partitions']) == 3
    assert not (path.parent / 'run').exists()


@pytest.mark.parametrize('key,value', [('epochs', 0), ('batch_size', 1), ('learning_rate', float('nan')),
    ('seed', True), ('freeze_backbone', 'yes'), ('device', 'mps'), ('mode', 'unknown'),
    ('initial_weights_sha256', '0' * 64)])
def test_config_fails_before_outputs(run_inputs, key, value):
    path, config, _ = run_inputs
    config[key] = value
    path.write_text(json.dumps(config))
    with pytest.raises(ValueError):
        training.run_training(path)
    assert not (path.parent / 'run').exists()


def test_manifest_detects_leakage_through_copy(run_inputs):
    path, _, manifest = run_inputs
    first, other = manifest['samples'][0], manifest['samples'][4]
    extra = {**first['copies'][0], 'path': 'extra.dcm', 'study_uid': other['study_uid']}
    first['copies'].append(extra)
    (path.parent / 'splits.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='leakage'):
        training.run_training(path)


def test_tampered_image_rejected(run_inputs):
    path, _, manifest = run_inputs
    sample = manifest['samples'][0]
    write_dicom(path.parent / sample['path'])
    with pytest.raises(ValueError, match='identity changed'):
        training.run_training(path)


def test_escape_via_symlink_rejected(run_inputs, tmp_path):
    path, config, manifest = run_inputs
    root = tmp_path / 'inside'
    root.mkdir()
    sample = manifest['samples'][0]
    (root / sample['path']).symlink_to(tmp_path / sample['path'])
    with pytest.raises(ValueError, match='escapes'):
        training.image_tensor(sample, root, config['preprocessing'])


def test_missing_class_blocks_training_partition(run_inputs):
    path, _, manifest = run_inputs
    for s in manifest['samples']:
        if s['region'] == 'hip':
            s['labels']['hip_roi'] = None
    (path.parent / 'splits.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='both observed classes for hip_roi'):
        training.run_training(path)


def test_reviewed_conflict_requires_explicit_flag(run_inputs):
    path, _, manifest = run_inputs
    sample = next(s for s in manifest['samples'] if s['region'] == 'spine')
    sample['labels'].update(spine_position=0, spine_axis=0, spine_artifact=0, spine_quality=1)
    target = path.parent / 'splits.json'
    target.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Contradictory'):
        training.load_manifest(target)
    sample['conflict_reviewed'] = True
    target.write_text(json.dumps(manifest))
    loaded = training.load_manifest(target)
    assert next(s for s in loaded['samples'] if s['region'] == 'spine')['conflict_reviewed'] is True


def test_frozen_backbone_batchnorm_and_bundle_roundtrip(run_inputs, tmp_path):
    torch = pytest.importorskip('torch')
    path, _, manifest = run_inputs
    config = training.load_config(path)
    model = training.initialize_model(config, 42)
    training.training_mode(model, True)
    assert model.fc.training and not model.bn1.training
    assert all(p.requires_grad == name.startswith('fc.') for name, p in model.named_parameters())
    sample = manifest['samples'][0]
    tensor = training.image_tensor(sample, config['data_root'], config['preprocessing'])
    before = model.bn1.running_mean.clone()
    with torch.inference_mode():
        expected = model(tensor[None])
    assert torch.equal(before, model.bn1.running_mean)
    training.save_bundle(model, tmp_path / 'bundle', config)
    predictor = load_quality_predictor(tmp_path / 'bundle/manifest.json')
    with torch.inference_mode():
        actual = predictor.model(tensor[None])
    torch.testing.assert_close(expected, actual)
    assert predictor.predict(read_dicom(path.parent / sample['path'])).anatomical_region in training.REGIONS


def test_each_fold_has_fresh_model_and_only_train_samples(run_inputs, monkeypatch):
    path, _, manifest = run_inputs
    models, seen = [], []
    original = training.initialize_model
    def initialize(config, seed):
        model = original(config, seed)
        models.append(model)
        return model
    def no_optimization(model, samples, config):
        assert config['seed'] == 42 + len(seen)
        seen.append({s['pixel_hash'] for s in samples})
        return [{'epoch': 1, 'training_loss': 0.0}]
    monkeypatch.setattr(training, 'initialize_model', initialize)
    monkeypatch.setattr(training, 'fit_model', no_optimization)
    # Real decoding, forward inference and bundle export, but optimization replaced.
    result = training.run_training(path, execute=True)
    assert result['status'] == 'complete'
    assert len(models) == 3 and len({id(m) for m in models}) == 3
    for fold, hashes in enumerate(seen):
        assert hashes == {s['pixel_hash'] for s in manifest['samples'] if s['fold'] != fold}
    with (path.parent / 'run/oof.csv').open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 12 and len({r['pixel_hash'] for r in rows}) == 12
    assert (path.parent / 'run/completed.json').exists()
    assert not (path.parent / 'run/failed.json').exists()
    with pytest.raises(ValueError, match='already exists'):
        training.run_training(path)


def test_failure_is_not_marked_complete(run_inputs, monkeypatch):
    path, _, _ = run_inputs
    def failure(*args):
        raise RuntimeError('simulated failure before optimization')
    monkeypatch.setattr(training, 'fit_model', failure)
    with pytest.raises(RuntimeError, match='simulated'):
        training.run_training(path, execute=True)
    assert (path.parent / 'run/failed.json').exists()
    assert not (path.parent / 'run/completed.json').exists()


def test_final_has_no_held_out_or_oof(run_inputs, monkeypatch):
    path, config, _ = run_inputs
    config['mode'] = 'final'
    path.write_text(json.dumps(config))
    monkeypatch.setattr(training, 'fit_model', lambda *args: [])
    result = training.run_training(path, execute=True)
    assert len(result['partitions']) == 1
    assert result['partitions'][0]['held_out_images'] == 0
    assert (path.parent / 'run/final/manifest.json').exists()
    assert not (path.parent / 'run/oof.csv').exists()


@pytest.mark.parametrize('length,batch_size', [(7, 3), (2, 8), (4, 3), (9, 2)])
def test_batches_never_drop_images_or_leave_singleton(length, batch_size):
    batches = training.batches(length, batch_size, np.random.default_rng(42))
    assert all(len(b) >= 2 for b in batches)
    assert sorted(i for b in batches for i in b) == list(range(length))


def test_config_generator_records_local_hash_without_training(run_inputs):
    path, config, _ = run_inputs
    target = path.parent / 'generated.json'
    assert main(['training-config', str(path.parent / 'splits.json'), '--data-root', str(path.parent),
                 '--initial-weights', config['initial_weights'], '--output', str(path.parent / 'future'),
                 '--config', str(target)]) == 0
    generated = training.load_config(target)
    assert generated['initial_weights_sha256'] == config['initial_weights_sha256']
    assert generated['freeze_backbone'] is True
    assert generated['preprocessing']['size'] == 224
    assert not (path.parent / 'future').exists()
