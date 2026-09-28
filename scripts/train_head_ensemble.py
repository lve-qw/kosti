"""Reduce initialization variance by averaging linear heads on one frozen encoder.

The resulting bundle remains a single ResNet18: no inference-time ensemble cost.
Uses the identity-checked frozen feature cache produced by fit_nested_probe.py.
"""
import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from kosti.evaluation import masked_bce_with_logits
from kosti.preparation import LABELS, evaluate_predictions, write_json
from kosti.training import batches, initialize_model, load_config, load_manifest, save_bundle, sha256, training_support


def run(config_path, cache, output):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = load_config(config_path)
    manifest = load_manifest(config['manifest'])
    protocol = json.loads((cache / 'protocol.json').read_text())
    if protocol['manifest_sha256'] != sha256(config['manifest']):
        raise ValueError('Cache manifest mismatch')
    for key in ('preprocessing', 'initial_weights_sha256', 'data_root'):
        if protocol['config'][key] != config[key]:
            raise ValueError('Feature cache mismatch: ' + key)
    samples = manifest['samples']
    x = torch.load(cache / 'features.pth', weights_only=True, map_location='cpu')
    if x.shape != (len(samples), 512) or not torch.isfinite(x).all():
        raise ValueError('Invalid frozen feature cache')
    output.mkdir(parents=True, exist_ok=False)
    seeds = [42, 142, 242, 342, 442, 542, 642]
    write_json(output / 'protocol.json', {'config': config, 'seeds': seeds,
        'feature_sha256': sha256(cache / 'features.pth'), 'aggregation': 'arithmetic mean of linear weights and biases',
        'scope': 'Fixed development experiment, not independent validation'})
    y = torch.tensor([[np.nan if s['labels'][k] is None else s['labels'][k] for k in LABELS] for s in samples])
    anatomy = torch.tensor([int(s['region'] == 'hip') for s in samples])
    rows = []
    for fold in range(manifest['n_splits']):
        train = [i for i,s in enumerate(samples) if s['fold'] != fold]
        test = [i for i,s in enumerate(samples) if s['fold'] == fold]
        support = training_support([samples[i] for i in train])
        pw = torch.tensor([(support[k]['observed']-support[k]['positive'])/support[k]['positive'] for k in LABELS])
        weights, biases = [], []
        for seed in seeds:
            model = initialize_model(config, seed+fold).eval()
            head = model.fc
            optimizer = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
            rng = np.random.default_rng(seed+fold)
            for _ in range(config['epochs']):
                for positions in batches(len(train), config['batch_size'], rng):
                    indices = [train[i] for i in positions]
                    logits = head(x[indices])
                    loss = torch.nn.functional.cross_entropy(logits[:, :2], anatomy[indices])
                    loss = loss + masked_bce_with_logits(logits[:, 2:], y[indices], pw)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            weights.append(head.weight.detach().clone())
            biases.append(head.bias.detach().clone())
        with torch.no_grad():
            model.fc.weight.copy_(torch.stack(weights).mean(0))
            model.fc.bias.copy_(torch.stack(biases).mean(0))
            probabilities = torch.sigmoid(model.fc(x[test])[:, 2:]).tolist()
        save_bundle(model, output / f'fold-{fold}', config)
        for i, p in zip(test, probabilities):
            rows.append({'pixel_hash': samples[i]['pixel_hash'], 'fold': fold, **dict(zip(LABELS, p))})
        print(f'Fold {fold} complete', flush=True)
    with (output / 'oof.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['pixel_hash', 'fold', *LABELS])
        writer.writeheader()
        writer.writerows(rows)
    report = evaluate_predictions(config['manifest'], output / 'oof.csv', output / 'metrics.json')
    print('Violations macro F1:', report['violations']['macro_f1'], flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'cache', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.cache, args.output)
