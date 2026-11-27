"""Metrics written by hand with numpy so each line can be explained at the viva.

Labels: y = 1 for melanoma, 0 otherwise. Scores: higher = more likely melanoma.
"""
import numpy as np


def auroc(y, s):
    """Probability a random melanoma gets a higher score than a random non-melanoma (ties count 0.5).
    Computed from ranks (Mann-Whitney U)."""
    y = np.asarray(y); s = np.asarray(s, dtype=float)
    n1, n0 = int((y == 1).sum()), int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s)); sorted_s = s[order]
    i = 0
    while i < len(s):                      # average ranks for tied scores
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def threshold_at_specificity(y, s, specificity=0.90):
    """Cut-off chosen on VALIDATION data so that `specificity` of non-melanomas score below it."""
    neg = np.sort(np.asarray(s)[np.asarray(y) == 0])
    k = int(np.ceil(specificity * len(neg))) - 1
    return float(neg[max(k, 0)])


def sensitivity(y, s, thr):
    """Share of real melanomas flagged (score > thr)."""
    pos = np.asarray(s)[np.asarray(y) == 1]
    return float((pos > thr).mean()) if len(pos) else float("nan")


def false_positive_rate(y, s, thr):
    neg = np.asarray(s)[np.asarray(y) == 0]
    return float((neg > thr).mean()) if len(neg) else float("nan")


def bootstrap_ci(y, s, groups, thr, stat, n_boot=1000, seed=0, cluster=None):
    """95% CI by resampling clusters (patients/lesions) with replacement.
    `stat(y, s, thr)` is any function above. `cluster` = array of cluster ids (default: each row alone)."""
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    cluster = np.arange(len(y)) if cluster is None else np.asarray(cluster)
    ids = np.unique(cluster)
    idx_by = {c: np.where(cluster == c)[0] for c in ids}
    vals = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([idx_by[c] for c in pick])
        v = stat(y[idx], s[idx], thr)
        if not np.isnan(v):
            vals.append(v)
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))) if vals else (float("nan"),) * 2
