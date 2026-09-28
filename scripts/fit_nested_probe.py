"""Train-only regularization/threshold selection on grouped development folds.

No external downloads. Outer folds are never used to fit scalers or select heads.
This is development evaluation, not a new independent clinical test.
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from kosti.evaluation import binary_metrics, multilabel_metrics
from kosti.preparation import LABELS, write_json
from kosti.training import image_tensor, initialize_model, load_config, load_manifest, sha256


def fit(x, y, penalty):
    if x.ndim != 2 or y.shape != (len(x),) or not torch.isfinite(x).all():
        raise ValueError('Expected finite feature matrix and aligned labels')
    if not np.isfinite(penalty) or penalty <= 0:
        raise ValueError('Positive finite regularization required')
    observed = torch.isfinite(y)
    if torch.isinf(y).any() or not torch.isin(y[observed], torch.tensor([0., 1.], dtype=y.dtype)).all():
        raise ValueError('Observed labels must be binary')
    x, y = x[observed].double(), y[observed].double()
    if not len(y):
        raise ValueError('No observed training labels')
    mean = x.mean(0)
    scale = x.std(0, unbiased=False).clamp_min(.1)
    z = (x - mean) / scale
    w = torch.zeros(x.shape[1], dtype=torch.float64, requires_grad=True)
    b = torch.tensor(float(torch.logit((y.sum()+.5)/(len(y)+1))), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([w, b], max_iter=100, tolerance_grad=1e-7,
                                line_search_fn='strong_wolfe')

    def closure():
        optimizer.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(z @ w + b, y)
        loss = loss + penalty * w.square().sum() / 2
        loss.backward()
        return loss

    optimizer.step(closure)
    # Fold normalization into a normal linear head, no runtime scaler needed.
    return (w / scale).detach().float(), (b - (mean * w / scale).sum()).detach().float()


def select(x, y, folds):
    if folds.shape != y.shape or len(set(folds.tolist())) < 2:
        raise ValueError('At least two aligned inner folds required')
    candidates = []
    for penalty in (.01, .1, 1., 10.):
        p = torch.zeros(len(y))
        for fold in sorted(set(folds.tolist())):
            train, test = folds != fold, folds == fold
            w, b = fit(x[train], y[train], penalty)
            p[test] = torch.sigmoid(x[test] @ w + b)
        for threshold in (.05, .1, .2, .3, .4, .5):
            metrics = binary_metrics(y.numpy(), p.numpy(), threshold)
            candidates.append((metrics['f1'], penalty, threshold))
    # On ties prefer stronger shrinkage, then the higher threshold.
    score, penalty, threshold = max(candidates)
    w, b = fit(x, y, penalty)
    return w, b, {'penalty': penalty, 'threshold': threshold, 'inner_f1': score}


def run(config_path, output):
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    config = load_config(config_path)
    manifest = load_manifest(config['manifest'])
    samples = manifest['samples']
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'protocol.json', {
        'config': config, 'manifest_sha256': sha256(config['manifest']),
        'penalties': [.01, .1, 1., 10.], 'thresholds': [.05, .1, .2, .3, .4, .5],
        'selection': 'Per-label inner grouped OOF F1; tie stronger shrinkage then higher threshold',
        'scope': 'Existing development folds; not independent confirmation',
    })
    model = initialize_model(config, config['seed']).eval()
    encoder = torch.nn.Sequential(*list(model.children())[:-1])
    features = []
    with torch.no_grad():
        for i, sample in enumerate(samples):
            features.append(encoder(image_tensor(sample, config['data_root'], config['preprocessing'])[None]).flatten())
            if i % 50 == 0:
                print(f'Features {i}/{len(samples)}', flush=True)
    x = torch.stack(features)
    torch.save(x, output / 'features.pth')
    y = torch.tensor([[np.nan if s['labels'][k] is None else s['labels'][k] for k in LABELS] for s in samples])
    folds = torch.tensor([s['fold'] for s in samples])
    probabilities, decisions = torch.zeros_like(y), torch.zeros_like(y)
    selections = {}
    for fold in sorted(set(folds.tolist())):
        train, test = folds != fold, folds == fold
        selections[str(fold)] = {}
        for j, label in enumerate(LABELS):
            w, b, choice = select(x[train], y[train, j], folds[train])
            probabilities[test, j] = torch.sigmoid(x[test] @ w + b)
            decisions[test, j] = (probabilities[test, j] >= choice['threshold']).float()
            selections[str(fold)][label] = choice
        print(f'Outer fold {fold} complete', flush=True)
    write_json(output / 'selection.json', selections)
    report = {'fixed_half': multilabel_metrics(y.numpy(), probabilities.numpy(), LABELS),
              'nested_threshold': multilabel_metrics(y.numpy(), decisions.numpy(), LABELS)}
    # Ranking metrics must use probabilities, not the binarized decisions.
    for label in LABELS:
        for metric in ('roc_auc', 'average_precision'):
            report['nested_threshold']['per_label'][label][metric] = report['fixed_half']['per_label'][label][metric]
    report['violations_macro_f1'] = float(np.mean([report['nested_threshold']['per_label'][k]['f1'] for k in LABELS[:5]]))
    write_json(output / 'metrics.json', report)
    with (output / 'oof.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['pixel_hash', 'fold', *LABELS])
        writer.writeheader()
        for s, row in zip(samples, probabilities.tolist()):
            writer.writerow({'pixel_hash': s['pixel_hash'], 'fold': s['fold'], **dict(zip(LABELS, row))})
    torch.save(decisions, output / 'oof-decisions.pth')
    print(report, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.config, args.output)
