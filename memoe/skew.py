"""Expert-popularity skew analysis."""
from __future__ import annotations
import numpy as np


def _gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = x.size
    if n == 0 or x.sum() == 0:
        return 0.0
    idx = np.arange(1, n + 1)
    return float((2 * (idx * x).sum()) / (n * x.sum()) - (n + 1) / n)


def fit_zipf_s(counts):
    """Least-squares slope of log(freq) vs log(rank) -> Zipf exponent."""
    c = np.sort(np.asarray(counts, dtype=float))[::-1]
    c = c[c > 0]
    if c.size < 3:
        return 0.0
    r = np.log(np.arange(1, c.size + 1))
    f = np.log(c / c.sum())
    return float(-np.polyfit(r, f, 1)[0])


def skew_stats(counts):
    """counts: (n_layers, n_experts) -> dict of per-model skew metrics."""
    counts = np.asarray(counts, dtype=float)
    per_layer = []
    for row in counts:
        tot = row.sum()
        if tot == 0:
            continue
        p = row / tot
        nz = p[p > 0]
        H = float(-(nz * np.log2(nz)).sum())
        Hmax = float(np.log2(row.size))
        srt = np.sort(p)[::-1]
        per_layer.append({
            "entropy_bits": H,
            "norm_entropy": H / Hmax if Hmax else 0.0,
            "gini": _gini(row),
            "zipf_s": fit_zipf_s(row),
            "top10pct_share": float(srt[:max(1, row.size // 10)].sum()),
            "top25pct_share": float(srt[:max(1, row.size // 4)].sum()),
            "top50pct_share": float(srt[:max(1, row.size // 2)].sum()),
            "max_share": float(srt[0]),
            "dead_experts": int((row == 0).sum()),
        })
    keys = per_layer[0].keys() if per_layer else []
    agg = {k: float(np.mean([d[k] for d in per_layer])) for k in keys}
    agg["n_layers"] = len(per_layer)
    return {"aggregate": agg, "per_layer": per_layer}


def coverage_curve(counts, points=100):
    """Fraction of token-routings served vs fraction of experts kept resident.

    It is the upper bound on the hit rate a static popularity-based tiering
    policy can reach at a given HBM capacity.
    """
    counts = np.asarray(counts, dtype=float)
    n_layers, n_experts = counts.shape
    fracs = np.linspace(0, 1, points + 1)
    cov = np.zeros_like(fracs)
    for row in counts:
        tot = row.sum()
        if tot == 0:
            continue
        srt = np.sort(row)[::-1]
        cum = np.concatenate([[0.0], np.cumsum(srt) / tot])
        k = np.clip((fracs * n_experts).round().astype(int), 0, n_experts)
        cov += cum[k]
    return fracs, cov / max(1, n_layers)
