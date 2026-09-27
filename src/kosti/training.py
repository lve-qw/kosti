"""Reproducible offline ResNet18 baseline. Optimization requires explicit execute."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from pathlib import Path

import numpy as np

from .dicom import read_dicom
from .evaluation import masked_bce_with_logits
from .ml import REGIONS, VIOLATION_LABELS, preprocess, validate_metadata
from .preparation import IDENTITY, LABELS, audit_records, evaluate_predictions, read_json, write_json


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def bundle_metadata(preprocessing, checksum):
    return validate_metadata({
        'schema_version': 1, 'kind': 'quality', 'architecture': 'resnet18',
        'anatomy_classes': list(REGIONS), 'quality_regions': list(REGIONS),
        'violation_labels': [list(v) for v in VIOLATION_LABELS],
        'preprocessing': preprocessing, 'weights': 'weights.pth', 'weights_sha256': checksum,
        'quality_thresholds': [0.5] * 2, 'violation_thresholds': [0.5] * 5,
    })


def create_config(manifest, data_root, initial_weights, output, config_path, mode='cross_validation'):
    """Write a concrete config from local paths, without downloads or training."""
    if mode not in ('cross_validation', 'final'):
        raise ValueError('Invalid training mode')
    # Validate the prepared manifest now; actual DICOM/checkpoint loading is a dry run.
    load_manifest(manifest)
    if not Path(data_root).is_dir():
        raise ValueError('data_root must be an existing directory')
    config = {
        'schema_version': 1, 'manifest': str(Path(manifest).resolve()),
        'data_root': str(Path(data_root).resolve()), 'output': str(Path(output).resolve()),
        'mode': mode, 'seed': 42, 'epochs': 10, 'batch_size': 16,
        'learning_rate': 0.001, 'weight_decay': 0.01, 'device': 'cpu',
        'freeze_backbone': True, 'initial_weights': str(Path(initial_weights).resolve()),
        'initial_weights_sha256': sha256(initial_weights),
        'preprocessing': {'mode': 'minmax_letterbox_rgb', 'interpolation': 'bilinear',
                          'size': 224, 'padding_value': 0,
                          'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]},
    }
    write_json(config_path, config)
    return config


def load_config(path):
    path = Path(path).resolve()
    config = read_json(path)
    required = {'schema_version', 'manifest', 'data_root', 'output', 'mode', 'seed',
                'epochs', 'batch_size', 'learning_rate', 'weight_decay', 'device',
                'freeze_backbone', 'initial_weights', 'initial_weights_sha256', 'preprocessing'}
    if not isinstance(config, dict) or set(config) - {'balance_classes'} != required or config['schema_version'] != 1:
        raise ValueError('Training config must match schema v1 exactly')
    config.setdefault('balance_classes', False)
    if type(config['balance_classes']) is not bool:
        raise ValueError('balance_classes must be boolean')
    if config['mode'] not in ('cross_validation', 'final') or config['device'] not in ('cpu', 'cuda'):
        raise ValueError('Expected cross_validation/final mode and cpu/cuda device')
    for key, minimum in (('seed', 0), ('epochs', 1), ('batch_size', 2)):
        if type(config[key]) is not int or config[key] < minimum:
            raise ValueError(f'{key} must be an integer >= {minimum}')
    if config['seed'] >= 2**32 - 10000:
        raise ValueError('Seed is out of range')
    for key in ('learning_rate', 'weight_decay'):
        value = config[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f'{key} must be finite and nonnegative')
    if config['learning_rate'] == 0 or type(config['freeze_backbone']) is not bool:
        raise ValueError('Positive learning_rate and boolean freeze_backbone required')
    bundle_metadata(config['preprocessing'], '0' * 64)
    for key in ('manifest', 'data_root', 'output', 'initial_weights'):
        if not isinstance(config[key], str) or not config[key].strip():
            raise ValueError(f'{key} must name an explicit local path')
        config[key] = str((path.parent / config[key]).resolve())
    checksum = config['initial_weights_sha256']
    if not isinstance(checksum, str) or len(checksum) != 64 or any(c not in '0123456789abcdef' for c in checksum):
        raise ValueError('initial_weights_sha256 must be an explicit lowercase SHA256')
    if sha256(config['initial_weights']) != checksum:
        raise ValueError('Initial weight checksum mismatch')
    return config


def load_manifest(path):
    manifest = read_json(path)
    if not isinstance(manifest, dict) or manifest.get('schema_version') != 1 or manifest.get('label_names') != list(LABELS):
        raise ValueError('Expected prepared manifest schema v1')
    samples = manifest.get('samples')
    audit_records({'manifest': samples})
    n_splits = manifest.get('n_splits')
    if type(n_splits) is not int or not 2 <= n_splits <= 10000:
        raise ValueError('Invalid number of folds')
    hashes, paths, studies = set(), set(), {}
    for sample in samples:
        fold = sample.get('fold')
        if type(fold) is not int or not 0 <= fold < n_splits:
            raise ValueError('Invalid sample fold')
        if sample['pixel_hash'] in hashes:
            raise ValueError('Prepared samples must have unique pixel hashes')
        hashes.add(sample['pixel_hash'])
        region, side = sample.get('region'), sample.get('side')
        if region not in ('spine', 'hip') or (region == 'spine' and side != '') or (region == 'hip' and side not in ('', 'left', 'right')):
            raise ValueError('Invalid reviewed region/side')
        if not isinstance(sample.get('label_source'), str) or not sample['label_source'].strip():
            raise ValueError('Missing reviewed label source')
        labels = sample.get('labels')
        if not isinstance(labels, dict) or set(labels) != set(LABELS):
            raise ValueError('Expected seven nullable labels')
        for name, value in labels.items():
            if value is not None and (type(value) is not int or value not in (0, 1)):
                raise ValueError('Labels must be binary integers or null')
            if not name.startswith(region + '_') and value is not None:
                raise ValueError('Inapplicable labels must be null')
        parts = [labels[k] for k in LABELS if k.startswith(region + '_') and not k.endswith('_quality')]
        quality = labels[region + '_quality']
        if quality is not None and ((quality == 0 and 1 in parts) or
                                    (all(v is not None for v in parts) and quality != int(any(parts)))):
            raise ValueError('Contradictory quality labels')
        copies = sample.get('copies')
        audit_records({'manifest': copies})
        if {k: sample[k] for k in IDENTITY} not in copies:
            raise ValueError('Copies must include the representative image')
        for copy in copies:
            if copy['pixel_hash'] != sample['pixel_hash'] or copy['path'] in paths:
                raise ValueError('Invalid or repeated duplicate provenance')
            paths.add(copy['path'])
            previous = studies.setdefault(copy['study_uid'], fold)
            if previous != fold:
                raise ValueError('Study leakage across folds, including duplicate copies')
    if {s['fold'] for s in samples} != set(range(n_splits)):
        raise ValueError('Every fold must contain samples')
    return manifest


def partitions(manifest, mode):
    samples = manifest['samples']
    if mode == 'final':
        return [('final', samples, [])]
    return [(f'fold-{fold}', [s for s in samples if s['fold'] != fold],
             [s for s in samples if s['fold'] == fold]) for fold in range(manifest['n_splits'])]


def training_support(samples):
    if {s['region'] for s in samples} != {'spine', 'hip'}:
        raise ValueError('Each training partition must contain both anatomical classes')
    support = {}
    for name in LABELS:
        known = [s['labels'][name] for s in samples if s['labels'][name] is not None]
        support[name] = {'observed': len(known), 'positive': sum(known)}
        if set(known) != {0, 1}:
            raise ValueError(f'Training partition needs both observed classes for {name}')
    return support


def image_tensor(sample, root, preprocessing):
    """Recheck identity at use time; never trust a path outside the dataset root."""
    import torch
    root = Path(root).resolve()
    path = (root / sample['path']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Image path escapes data_root')
    image = read_dicom(path)
    if any(getattr(image, key) != sample[key] for key in ('study_uid', 'image_uid', 'pixel_hash')):
        raise ValueError('DICOM identity changed since audit: ' + sample['path'])
    return torch.from_numpy(preprocess(image.pixels, preprocessing))


class TrainingDataset:
    def __init__(self, samples, config):
        self.samples, self.config = samples, config

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        import torch
        sample = self.samples[index]
        return (image_tensor(sample, self.config['data_root'], self.config['preprocessing']),
                0 if sample['region'] == 'spine' else 1,
                torch.tensor([float('nan') if sample['labels'][k] is None else sample['labels'][k]
                              for k in LABELS], dtype=torch.float32))


def initialize_model(config, seed):
    import torch
    from torchvision.models import resnet18
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Local torchvision ResNet18 ImageNet state_dict, including its original 1000-class head.
    model = resnet18(weights=None)
    state = torch.load(config['initial_weights'], map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model.fc = torch.nn.Linear(model.fc.in_features, 9)
    if config['freeze_backbone']:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        for parameter in model.fc.parameters():
            parameter.requires_grad_(True)
    return model


def training_mode(model, freeze_backbone):
    if freeze_backbone:
        # Frozen batch-normalization buffers must not drift with training batches.
        model.eval()
        model.fc.train()
    else:
        model.train()


def batches(length, batch_size, rng):
    """Keep every image; merge a final singleton to avoid BatchNorm failures."""
    order = rng.permutation(length).tolist()
    chunks = [order[i:i + batch_size] for i in range(0, length, batch_size)]
    if len(chunks) > 1 and len(chunks[-1]) == 1:
        chunks[-2].extend(chunks.pop())
    return chunks


def fit_model(model, samples, config):
    """The only function that performs optimization. Never called by a dry run."""
    import torch
    model.to(config['device'])
    dataset = TrainingDataset(samples, config)
    pos_weight = None
    if config.get('balance_classes', False):
        support = training_support(samples)
        pos_weight = torch.tensor([(support[k]['observed'] - support[k]['positive']) /
                                   support[k]['positive'] for k in LABELS],
                                  dtype=torch.float32, device=config['device'])
    # With no augmentations and a frozen encoder, features are constant across epochs.
    # Cache only this partition's features; BatchNorm remains in evaluation mode.
    cached = None
    if config['freeze_backbone']:
        model.eval()
        encoder = torch.nn.Sequential(*list(model.children())[:-1])
        cached_features, cached_anatomy, cached_targets = [], [], []
        with torch.no_grad():
            for start in range(0, len(dataset), config['batch_size']):
                items = [dataset[i] for i in range(start, min(start + config['batch_size'], len(dataset)))]
                images = torch.stack([i[0] for i in items]).to(config['device'])
                cached_features.append(encoder(images).flatten(1).detach())
                cached_anatomy.extend(i[1] for i in items)
                cached_targets.extend(i[2] for i in items)
        cached = (torch.cat(cached_features), torch.tensor(cached_anatomy, device=config['device']),
                  torch.stack(cached_targets).to(config['device']))
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                                  lr=config['learning_rate'], weight_decay=config['weight_decay'])
    rng, history = np.random.default_rng(config['seed']), []
    for epoch in range(config['epochs']):
        training_mode(model, config['freeze_backbone'])
        loss_sum = 0.
        for indices in batches(len(dataset), config['batch_size'], rng):
            if cached is None:
                items = [dataset[i] for i in indices]
                images = torch.stack([i[0] for i in items]).to(config['device'])
                anatomy = torch.tensor([i[1] for i in items], device=config['device'])
                targets = torch.stack([i[2] for i in items]).to(config['device'])
            else:
                anatomy, targets = cached[1][indices], cached[2][indices]
            optimizer.zero_grad(set_to_none=True)
            logits = model(images) if cached is None else model.fc(cached[0][indices])
            loss = torch.nn.functional.cross_entropy(logits[:, :2], anatomy) + masked_bce_with_logits(logits[:, 2:], targets, pos_weight)
            if not torch.isfinite(loss):
                raise ValueError('Non-finite training loss')
            loss.backward()
            optimizer.step()
            loss_sum += float(loss.detach()) * len(indices)
        history.append({'epoch': epoch + 1, 'training_loss': loss_sum / len(dataset)})
    return history


def predict_held_out(model, samples, config):
    import torch
    model.to(config['device']).eval()
    rows = []
    with torch.inference_mode():
        for sample in samples:
            tensor = image_tensor(sample, config['data_root'], config['preprocessing'])
            logits = model(tensor[None].to(config['device']))
            if tuple(logits.shape) != (1, 9) or not torch.isfinite(logits).all():
                raise ValueError('Invalid held-out model output')
            values = torch.sigmoid(logits[0, 2:]).cpu().tolist()
            rows.append({'pixel_hash': sample['pixel_hash'], 'fold': sample['fold'], **dict(zip(LABELS, values))})
    return rows


def save_bundle(model, directory, config):
    import torch
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    weights = directory / 'weights.pth'
    state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if any(not torch.isfinite(value).all() for value in state.values()):
        raise ValueError('Cannot export non-finite model state')
    torch.save(state, weights)
    write_json(directory / 'manifest.json', bundle_metadata(config['preprocessing'], sha256(weights)))


def run_training(config_path, *, execute=False):
    config = load_config(config_path)
    manifest = load_manifest(config['manifest'])
    output = Path(config['output'])
    if output.exists():
        raise ValueError('Output already exists; choose a new run directory')
    plan = partitions(manifest, config['mode'])
    summary = [{'name': name, 'training_images': len(train), 'held_out_images': len(test),
                'training_support': training_support(train)} for name, train, test in plan]
    # DICOM forward preprocessing and checkpoint compatibility only; no optimizer.
    for sample in manifest['samples']:
        image_tensor(sample, config['data_root'], config['preprocessing'])
    model = initialize_model(config, config['seed'])
    if not execute:
        return {'status': 'validated_without_training', 'partitions': summary,
                'output_created': False, 'device_checked': False}
    import torch
    if config['device'] == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA requested but unavailable')
    torch.use_deterministic_algorithms(True)
    if config['device'] == 'cuda':
        # Must be set before creating the CUDA context on the target machine.
        import os
        if os.getenv('CUBLAS_WORKSPACE_CONFIG') not in (':4096:8', ':16:8'):
            raise ValueError('Set CUBLAS_WORKSPACE_CONFIG=:4096:8 before CUDA execution')
        torch.backends.cudnn.benchmark = False
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', config)
    write_json(output / 'splits.json', manifest)
    write_json(output / 'run.json', {'status': 'started', 'manifest_sha256': sha256(config['manifest']),
                                  'torch_version': str(torch.__version__), 'partitions': summary})
    predictions = []
    try:
        for index, (name, train, test) in enumerate(plan):
            fold_config = {**config, 'seed': config['seed'] + index}
            if index:
                model = initialize_model(config, fold_config['seed'])
            history = fit_model(model, train, fold_config)
            save_bundle(model, output / name, config)
            write_json(output / name / 'history.json', history)
            predictions.extend(predict_held_out(model, test, config))
        if config['mode'] == 'cross_validation':
            with (output / 'oof.csv').open('x', encoding='utf-8', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=['pixel_hash', 'fold', *LABELS])
                writer.writeheader()
                writer.writerows(predictions)
            evaluate_predictions(output / 'splits.json', output / 'oof.csv', output / 'metrics.json')
        write_json(output / 'completed.json', {'status': 'complete', 'mode': config['mode'],
                                              'clinical_validation': False})
    except Exception:
        write_json(output / 'failed.json', {'status': 'failed', 'partial_outputs': True})
        raise
    return {'status': 'complete', 'output': str(output), 'partitions': summary}
