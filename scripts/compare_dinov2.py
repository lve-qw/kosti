"""Fixed-protocol frozen DINOv2 linear probe; no downloads or test-fold tuning."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch

from kosti.evaluation import masked_bce_with_logits
from kosti.preparation import LABELS, evaluate_predictions, write_json
from kosti.training import batches, image_tensor, load_config, load_manifest, sha256, training_support


def run(config_path, upstream, weights, output):
    config = load_config(config_path)
    manifest = load_manifest(config['manifest'])
    samples = manifest['samples']
    output = Path(output)
    if output.exists():
        raise FileExistsError('Choose a new experiment directory')
    upstream = Path(upstream).resolve()
    revision = subprocess.check_output(['git', '-C', str(upstream), 'rev-parse', 'HEAD'], text=True).strip()
    sys.path.insert(0, str(upstream))
    from dinov2.hub.backbones import dinov2_vits14
    torch.set_num_threads(1)
    torch.manual_seed(config['seed'])
    encoder = dinov2_vits14(pretrained=False).eval()
    encoder.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True), strict=True)
    features = []
    with torch.no_grad():
        for sample in samples:
            features.append(encoder(image_tensor(sample, config['data_root'], config['preprocessing'])[None])[0])
    features = torch.stack(features)
    targets = torch.tensor([[float('nan') if s['labels'][k] is None else s['labels'][k]
                             for k in LABELS] for s in samples])
    regions = torch.tensor([0 if s['region'] == 'spine' else 1 for s in samples])
    output.mkdir(parents=True)
    write_json(output / 'protocol.json', {'encoder': 'dinov2_vits14', 'upstream_commit': revision,
               'weights_sha256': sha256(weights), 'config': config,
               'fixed_threshold': .5, 'frozen_features': True,
               'scope': 'Exploratory comparison on same development folds; not independent confirmation.'})
    rows, anatomy_correct = [], []
    for fold in range(manifest['n_splits']):
        train = [i for i, s in enumerate(samples) if s['fold'] != fold]
        test = [i for i, s in enumerate(samples) if s['fold'] == fold]
        torch.manual_seed(config['seed'] + fold)
        head = torch.nn.Linear(features.shape[1], 9)
        optimizer = torch.optim.AdamW(head.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
        support = training_support([samples[i] for i in train])
        pos_weight = torch.tensor([(support[k]['observed']-support[k]['positive'])/support[k]['positive'] for k in LABELS])
        rng, history = np.random.default_rng(config['seed'] + fold), []
        for epoch in range(config['epochs']):
            total = 0.
            for positions in batches(len(train), config['batch_size'], rng):
                indices = [train[i] for i in positions]
                logits = head(features[indices])
                loss = torch.nn.functional.cross_entropy(logits[:, :2], regions[indices]) + masked_bce_with_logits(
                    logits[:, 2:], targets[indices], pos_weight if config['balance_classes'] else None)
                optimizer.zero_grad(); loss.backward(); optimizer.step()
                total += float(loss.detach()) * len(indices)
            history.append({'epoch': epoch + 1, 'training_loss': total/len(train)})
        torch.save(head.state_dict(), output / f'fold-{fold}-head.pth')
        write_json(output / f'fold-{fold}-history.json', history)
        with torch.no_grad():
            logits = head(features[test])
            values = torch.sigmoid(logits[:, 2:]).tolist()
            predictions = logits[:, :2].argmax(1).tolist()
        for index, probabilities, prediction in zip(test, values, predictions):
            sample = samples[index]
            anatomy_correct.append(prediction == int(regions[index]))
            rows.append({'pixel_hash': sample['pixel_hash'], 'fold': fold, **dict(zip(LABELS, probabilities))})
    with (output / 'oof.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['pixel_hash', 'fold', *LABELS]); writer.writeheader(); writer.writerows(rows)
    report = evaluate_predictions(config['manifest'], output / 'oof.csv', output / 'metrics.json')
    write_json(output / 'anatomy.json', {'accuracy': float(np.mean(anatomy_correct)), 'images': len(samples)})
    write_json(output / 'completed.json', {'status': 'complete', 'production_bundle': False})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('config', 'upstream', 'weights', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    args = parser.parse_args()
    run(args.config, args.upstream, args.weights, args.output)
    print('DINOv2 comparison complete')
