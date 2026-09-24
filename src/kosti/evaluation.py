"""Leakage-safe grouping and metrics. Missing labels are NaN, never normal."""
from __future__ import annotations
import numpy as np


def grouped_folds(study_uids, pixel_hashes, n_splits=3, seed=42):
    """Union connected studies/exact duplicate pixels before assigning folds.

    Returns one fold index per image; this is grouping, not stratification.
    Check positive counts per fold before evaluating rare labels.
    """
    studies, hashes = list(study_uids), list(pixel_hashes)
    if len(studies) != len(hashes) or not studies:
        raise ValueError("Aligned nonempty study UIDs and hashes required")
    if any(not isinstance(x, str) or not x for x in studies + hashes):
        raise ValueError("Missing study/hash would undermine leakage protection")
    if type(n_splits) is not int or n_splits < 2:
        raise ValueError("At least two folds required")
    parent = list(range(len(studies)))
    def find(i):
        while i != parent[i]:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    for values in (studies, hashes):
        seen = {}
        for i, value in enumerate(values):
            if value in seen:
                parent[find(i)] = find(seen[value])
            else:
                seen[value] = i
    groups = {}
    for i in range(len(studies)):
        groups.setdefault(find(i), []).append(i)
    if len(groups) < n_splits:
        raise ValueError("Fewer independent groups than folds")
    components = list(groups.values())
    np.random.default_rng(seed).shuffle(components)
    components.sort(key=len, reverse=True)
    sizes, folds = np.zeros(n_splits, dtype=int), np.empty(len(studies), dtype=int)
    for component in components:
        fold = int(sizes.argmin())
        folds[component] = fold
        sizes[fold] += len(component)
    return folds


def _observed(y_true, probability):
    y, p = np.asarray(y_true, dtype=float), np.asarray(probability, dtype=float)
    if y.ndim != 1 or y.shape != p.shape:
        raise ValueError("Expected aligned one-dimensional labels and probabilities")
    if np.isinf(y).any() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("Invalid labels or probabilities")
    mask = ~np.isnan(y)
    if not np.isin(y[mask], [0, 1]).all():
        raise ValueError("Observed labels must be binary")
    return y[mask].astype(int), p[mask]


def binary_metrics(y_true, probability, threshold=0.5):
    if not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold")
    y, p = _observed(y_true, probability)
    predicted = p >= threshold
    tp = int(((y == 1) & predicted).sum())
    fn = int(((y == 1) & ~predicted).sum())
    tn = int(((y == 0) & ~predicted).sum())
    fp = int(((y == 0) & predicted).sum())
    positive, negative = p[y == 1], p[y == 0]
    auc = None
    if len(positive) and len(negative):
        # Mann-Whitney definition with half credit for ties.
        auc = float(np.mean((positive[:, None] > negative) + 0.5 * (positive[:, None] == negative)))
    ap = None
    if len(positive):
        order = np.argsort(-p, kind="stable")
        sorted_y, sorted_p = y[order], p[order]
        ends = np.r_[np.flatnonzero(np.diff(sorted_p)), len(y) - 1]
        cumulative_tp = np.cumsum(sorted_y)[ends]
        recall = cumulative_tp / len(positive)
        ap = float(np.sum(np.diff(np.r_[0., recall]) * cumulative_tp / (ends + 1)))
    balanced = ((tp / (tp + fn) + tn / (tn + fp)) / 2 if tp + fn and tn + fp else None)
    return {"average_precision": ap, "balanced_accuracy": balanced, "n": len(y), "positive": tp + fn, "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "f1": (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else (0.0 if len(y) else None)),
            "sensitivity": tp / (tp + fn) if tp + fn else None,
            "specificity": tn / (tn + fp) if tn + fp else None, "roc_auc": auc}


def multilabel_metrics(y_true, probability, names, thresholds=None):
    y, p = np.asarray(y_true, dtype=float), np.asarray(probability, dtype=float)
    if y.ndim != 2 or y.shape != p.shape or y.shape[1] != len(names) or len(set(names)) != len(names):
        raise ValueError("Aligned matrices and unique label names required")
    thresholds = [0.5] * len(names) if thresholds is None else list(thresholds)
    if len(thresholds) != len(names):
        raise ValueError("One threshold per label required")
    per_label = {name: binary_metrics(y[:, i], p[:, i], thresholds[i]) for i, name in enumerate(names)}
    f1s = [v["f1"] for v in per_label.values() if v["f1"] is not None]
    return {"per_label": per_label, "macro_f1": float(np.mean(f1s)) if f1s else None}


def grouped_bootstrap_ci(y_true, probability, groups, metric="f1", threshold=0.5, repeats=1000, seed=42):
    """Resample whole studies; report how many draws have a defined metric."""
    y, p = np.asarray(y_true, dtype=float), np.asarray(probability, dtype=float)
    _observed(y, p)
    groups = np.asarray(groups)
    if groups.shape != y.shape or not len(y) or type(repeats) is not int or repeats < 1:
        raise ValueError("Aligned nonempty group IDs and positive repeats required")
    if metric not in ("f1", "roc_auc", "sensitivity", "specificity", "average_precision", "balanced_accuracy"):
        raise ValueError("Unsupported bootstrap metric")
    if any(not isinstance(g, (str, np.str_)) or not g for g in groups):
        raise ValueError("Bootstrap groups must be nonempty strings")
    unique = np.unique(groups)
    rng, scores = np.random.default_rng(seed), []
    for _ in range(repeats):
        indices = np.concatenate([np.flatnonzero(groups == g) for g in rng.choice(unique, size=len(unique), replace=True)])
        value = binary_metrics(y[indices], p[indices], threshold)[metric]
        if value is not None:
            scores.append(value)
    interval = np.quantile(scores, [0.025, 0.975]).tolist() if scores else [None, None]
    return {"lower": interval[0], "upper": interval[1], "valid_resamples": len(scores), "repeats": repeats}


def linked_study_groups(samples):
    """One resampling unit for studies connected by duplicate pixel arrays."""
    if not samples:
        raise ValueError("Nonempty prepared samples required")
    parent = list(range(len(samples)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    by_study, by_hash = {}, {}
    for index, sample in enumerate(samples):
        copies = sample.get("copies")
        if not isinstance(copies, list) or not copies:
            raise ValueError("Each sample needs nonempty duplicate provenance")
        for copy in copies:
            study, pixel_hash = copy.get("study_uid"), copy.get("pixel_hash")
            if not isinstance(study, str) or not study or not isinstance(pixel_hash, str) or not pixel_hash:
                raise ValueError("Missing study/hash in duplicate provenance")
            for seen, value in ((by_study, study), (by_hash, pixel_hash)):
                if value in seen:
                    parent[find(index)] = find(seen[value])
                else:
                    seen[value] = index
    return [str(find(index)) for index in range(len(samples))]


def masked_bce_with_logits(logits, targets):
    """Torch loss; NaN targets are ignored, including region-inapplicable labels."""
    import torch
    if logits.shape != targets.shape or not torch.isfinite(logits).all() or torch.isinf(targets).any():
        raise ValueError("Invalid aligned logits/targets")
    mask = ~torch.isnan(targets)
    observed = targets[mask]
    if not ((observed == 0) | (observed == 1)).all():
        raise ValueError("Observed labels must be binary")
    if not mask.any():
        return logits.sum() * 0
    return torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], observed)
