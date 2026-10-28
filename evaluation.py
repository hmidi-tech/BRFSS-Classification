"""Binary metrics and deterministic stratification, implemented in NumPy."""

import numpy as np


def stratified_split(y, seed=433):
    """Disjoint 60% development, 20% tuning, 20% final holdout indices."""
    rng = np.random.default_rng(seed)
    partitions = [[], [], []]
    for label in np.unique(y):
        ids = rng.permutation(np.flatnonzero(y == label))
        a, b = int(0.6 * ids.size), int(0.8 * ids.size)
        for dest, source in zip(partitions, [ids[:a], ids[a:b], ids[b:]]):
            dest.append(source)
    return tuple(rng.permutation(np.concatenate(p)) for p in partitions)


def stratified_folds(y, k=3, seed=433):
    rng = np.random.default_rng(seed)
    folds = [[] for _ in range(k)]
    for label in np.unique(y):
        chunks = np.array_split(rng.permutation(np.flatnonzero(y == label)), k)
        for fold, chunk in zip(folds, chunks):
            fold.append(chunk)
    return [rng.permutation(np.concatenate(fold)) for fold in folds]


def metrics(y, scores, threshold=0.5, probabilities=True):
    """Evaluate +1 as positive; undefined precision/AUC returns None."""
    y = np.asarray(y) == 1
    scores = np.asarray(scores, dtype=float)
    if scores.shape != y.shape or not y.size or not np.isfinite(scores).all():
        raise ValueError("Nonempty, aligned finite scores are required")
    pred = scores >= threshold
    tp = int(np.sum(y & pred))
    tn = int(np.sum(~y & ~pred))
    fp = int(np.sum(~y & pred))
    fn = int(np.sum(y & ~pred))
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    result = {
        "accuracy": (tp + tn) / y.size,
        "balanced_accuracy": (
            (recall + specificity) / 2
            if recall is not None and specificity is not None
            else None
        ),
        "precision": precision,
        "recall": recall,
        "specificity": specificity,
        "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
        "false_positive_rate": 1 - specificity if specificity is not None else None,
        "false_negative_rate": 1 - recall if recall is not None else None,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "threshold": float(threshold),
    }
    if y.any() and (~y).any():
        order = np.argsort(-scores, kind="stable")
        sorted_scores, sorted_y = scores[order], y[order]
        endpoints = np.r_[np.flatnonzero(np.diff(sorted_scores) != 0), y.size - 1]
        tps = np.cumsum(sorted_y)[endpoints].astype(float)
        fps = (endpoints + 1) - tps
        tpr, fpr = np.r_[0, tps / y.sum()], np.r_[0, fps / (~y).sum()]
        # Trapezoids give half-credit to positive/negative tied scores.
        result["roc_auc"] = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
        result["average_precision"] = float(np.sum(np.diff(tpr) * tps / (tps + fps)))
    else:
        result.update(roc_auc=None, average_precision=None)
    if probabilities:
        if np.any((scores < 0) | (scores > 1)):
            raise ValueError("Probabilities must be within [0,1]")
        clipped = np.clip(scores, 1e-15, 1 - 1e-15)
        result["binary_cross_entropy"] = float(
            -np.mean(y * np.log(clipped) + (~y) * np.log1p(-clipped))
        )
        result["brier"] = float(np.mean((scores - y) ** 2))
    return result


def choose_threshold(y, scores, objective="f1"):
    """Fixed grid, independent of holdout: 0.05..0.65 at intervals of 0.01.

    Equal objective values prefer higher thresholds (fewer positive calls).
    """
    grid = np.arange(5, 66) / 100
    rows = []
    positive = np.asarray(y) == 1
    for threshold in grid:
        pred = scores >= threshold
        tp = np.sum(pred & positive)
        fp = np.sum(pred & ~positive)
        fn = np.sum(~pred & positive)
        accuracy = float(np.mean(pred == positive))
        f1 = float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 0.0
        rows.append({"threshold": float(threshold), "accuracy": accuracy, "f1": f1})
    chosen = max(rows, key=lambda row: (row[objective], row["threshold"]))
    return chosen["threshold"], rows
