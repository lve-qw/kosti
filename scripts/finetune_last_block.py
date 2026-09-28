"""Fixed development experiment: fine-tune layer4 with frozen batch norm.

Caching the unchanged stem through layer3 makes CPU training practical.
No augmentation that could remove or manufacture positioning violations.
"""
import argparse
import csv

import numpy as np
import torch

from kosti.evaluation import masked_bce_with_logits
from kosti.preparation import LABELS, evaluate_predictions, write_json
from kosti.training import (batches, image_tensor, initialize_model, load_config,
                            load_manifest, save_bundle, sha256, training_support)
from pathlib import Path


def run(config_path, output, epochs=20, horizontal_flips=False):
    if epochs < 1:
        raise ValueError('Positive epoch count required')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = load_config(config_path)
    manifest = load_manifest(config['manifest'])
    samples = manifest['samples']
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', {
        'config': config, 'manifest_sha256': sha256(config['manifest']),
        'epochs': epochs, 'backbone_lr': .0001, 'head_lr': .001,
        'horizontal_flips': horizontal_flips,
        'frozen_batchnorm': True, 'threshold': .5,
        'scope': 'Fixed experiment on development folds; not independent confirmation',
    })
    model = initialize_model(config, config['seed']).eval()
    encoder = torch.nn.Sequential(*list(model.children())[:-3])
    features, flipped_features = [], []
    with torch.no_grad():
        for i, sample in enumerate(samples):
            tensor = image_tensor(sample, config['data_root'], config['preprocessing'])[None]
            features.append(encoder(tensor)[0])
            if horizontal_flips:
                flipped_features.append(encoder(tensor.flip(-1))[0])
            if i % 50 == 0:
                print(f'Features {i}/{len(samples)}', flush=True)
    x = torch.stack(features)
    flipped = torch.stack(flipped_features) if horizontal_flips else None
    y = torch.tensor([[np.nan if s['labels'][k] is None else s['labels'][k] for k in LABELS] for s in samples])
    regions = torch.tensor([int(s['region'] == 'hip') for s in samples])
    rows = []
    for fold in range(manifest['n_splits']):
        train = [i for i,s in enumerate(samples) if s['fold'] != fold]
        test = [i for i,s in enumerate(samples) if s['fold'] == fold]
        model = initialize_model(config, config['seed'] + fold).eval()
        for p in model.layer4.parameters():
            p.requires_grad_(True)
        head = torch.nn.Sequential(model.layer4, model.avgpool, torch.nn.Flatten(), model.fc)
        optimizer = torch.optim.AdamW([
            {'params': model.layer4.parameters(), 'lr': .0001},
            {'params': model.fc.parameters(), 'lr': .001}], weight_decay=.01)
        support = training_support([samples[i] for i in train])
        pos_weight = torch.tensor([(support[k]['observed']-support[k]['positive'])/support[k]['positive'] for k in LABELS])
        rng, history = np.random.default_rng(config['seed'] + fold), []
        for epoch in range(epochs):
            total = 0.
            for positions in batches(len(train), 16, rng):
                indices = [train[i] for i in positions]
                inputs = x[indices]
                if flipped is not None:
                    mask = torch.tensor(rng.random(len(indices)) < .5).reshape(-1, 1, 1, 1)
                    inputs = torch.where(mask, flipped[indices], inputs)
                logits = head(inputs)
                loss = torch.nn.functional.cross_entropy(logits[:, :2], regions[indices])
                loss = loss + masked_bce_with_logits(logits[:, 2:], y[indices], pos_weight)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 5.)
                optimizer.step()
                total += float(loss.detach()) * len(indices)
            history.append({'epoch': epoch+1, 'loss': total / len(train)})
            if epoch % 5 == 4:
                print(f'Fold {fold}, epoch {epoch+1}: loss {total/len(train):.4f}', flush=True)
        save_bundle(model, output / f'fold-{fold}', config)
        write_json(output / f'fold-{fold}' / 'history.json', history)
        with torch.no_grad():
            for indices in [test[i:i+16] for i in range(0, len(test), 16)]:
                logits = head(x[indices])
                for index, probabilities in zip(indices, torch.sigmoid(logits[:, 2:]).tolist()):
                    s = samples[index]
                    rows.append({'pixel_hash': s['pixel_hash'], 'fold': fold, **dict(zip(LABELS, probabilities))})
    with (output / 'oof.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['pixel_hash', 'fold', *LABELS])
        writer.writeheader()
        writer.writerows(rows)
    report = evaluate_predictions(config['manifest'], output / 'oof.csv', output / 'metrics.json')
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--horizontal-flips', action='store_true')
    args = parser.parse_args()
    run(args.config, args.output, args.epochs, args.horizontal_flips)
