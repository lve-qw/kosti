"""Fixed-protocol spine-axis specialist on grouped outer folds.

The experiment uses only horizontal reflection augmentation; rotations are
forbidden because they would manufacture or remove the target violation.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from kosti.evaluation import binary_metrics
from kosti.preparation import write_json
from kosti.training import image_tensor, initialize_model, load_config, load_manifest


def run(config_path, output, epochs=2):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = load_config(config_path)
    manifest = load_manifest(config['manifest'])
    samples = [s for s in manifest['samples'] if s['region'] == 'spine']
    if sum(s['labels']['spine_axis'] == 1 for s in samples) != 10:
        raise ValueError('Frozen protocol expects ten positive axis examples')
    output.mkdir(parents=True, exist_ok=False)
    base = initialize_model(config, config['seed']).eval()
    stem = torch.nn.Sequential(*list(base.children())[:-3])
    normal, mirrored = [], []
    with torch.inference_mode():
        for i, sample in enumerate(samples):
            tensor = image_tensor(sample, config['data_root'], config['preprocessing'])[None]
            normal.append(stem(tensor)[0])
            mirrored.append(stem(tensor.flip(-1))[0])
            if i % 20 == 0:
                print(f'features {i}/{len(samples)}', flush=True)
    normal, mirrored = torch.stack(normal), torch.stack(mirrored)
    labels = torch.tensor([s['labels']['spine_axis'] for s in samples], dtype=torch.float32)
    predictions = torch.zeros(len(samples))
    histories = {}
    for fold in range(manifest['n_splits']):
        train = torch.tensor([i for i,s in enumerate(samples) if s['fold'] != fold])
        test = torch.tensor([i for i,s in enumerate(samples) if s['fold'] == fold])
        torch.manual_seed(config['seed'] + fold)
        model = initialize_model(config, config['seed'] + fold).eval()
        model.fc = torch.nn.Linear(512, 1)
        head = torch.nn.Sequential(model.layer4, model.avgpool, torch.nn.Flatten(), model.fc)
        for p in head.parameters():
            p.requires_grad_(True)
        positive = labels[train].sum()
        pos_weight = torch.tensor([(len(train)-positive)/positive])
        optimizer = torch.optim.AdamW([
            {'params': model.layer4.parameters(), 'lr': 2e-5},
            {'params': model.fc.parameters(), 'lr': 2e-4}], weight_decay=.05)
        rng = np.random.default_rng(config['seed'] + fold)
        history = []
        for epoch in range(epochs):
            order = rng.permutation(len(train))
            total = 0.
            for start in range(0, len(order), 8):
                indices = train[order[start:start+8]]
                use_mirror = torch.tensor(rng.random(len(indices)) < .5).reshape(-1,1,1,1)
                features = torch.where(use_mirror, mirrored[indices], normal[indices])
                logits = head(features).flatten()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits, labels[indices], pos_weight=pos_weight)
                optimizer.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(head.parameters(), 2.)
                optimizer.step(); total += float(loss.detach()) * len(indices)
            history.append({'epoch': epoch+1, 'loss': total/len(train)})
        histories[str(fold)] = history
        with torch.inference_mode():
            # Reflection-invariant test-time averaging in logit space.
            logits = (head(normal[test]).flatten() + head(mirrored[test]).flatten()) / 2
            predictions[test] = torch.sigmoid(logits)
        print(f'fold {fold} complete', flush=True)
    metrics = binary_metrics(labels.numpy(), predictions.numpy())
    write_json(output/'report.json', {
        'protocol': {'epochs': epochs, 'outer_grouped_folds': 3,
                     'rotation_augmentation': False, 'horizontal_reflection': True,
                     'scope': 'Development cross-validation, not independent confirmation'},
        'metrics': metrics, 'history': histories,
        'predictions': [{'pixel_hash': s['pixel_hash'], 'fold': s['fold'],
                         'label': int(y), 'probability': float(p)}
                        for s,y,p in zip(samples, labels, predictions)]})
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=2)
    args = parser.parse_args()
    run(args.config, args.output, args.epochs)
