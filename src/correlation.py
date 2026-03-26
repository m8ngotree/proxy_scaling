"""Correlation computation utilities."""
import numpy as np
from scipy.stats import spearmanr, pearsonr


def bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute Spearman correlation with bootstrap confidence interval.

    Returns:
        (rho, ci_lower, ci_upper)
    """
    rho = float(spearmanr(x, y).statistic)
    rng = np.random.default_rng(seed)
    alpha = 1 - confidence
    rs = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, len(x), len(x))
        r = spearmanr(x[idx], y[idx]).statistic
        if not np.isnan(r):
            rs.append(r)
    if not rs:
        return rho, rho, rho
    ci_lo = float(np.percentile(rs, 100 * alpha / 2))
    ci_hi = float(np.percentile(rs, 100 * (1 - alpha / 2)))
    return rho, ci_lo, ci_hi


def compute_pairwise_correlations(
    proxy_bpb: dict[int, float],
    target_bpb: dict[int, float],
) -> dict:
    """Compute correlation between proxy and target BPB scores.

    Args:
        proxy_bpb: dict mapping mixture_id to proxy avg BPB
        target_bpb: dict mapping mixture_id to target avg BPB

    Returns:
        dict with spearman_r, pearson_r, ci_95, n_mixtures
    """
    shared_ids = sorted(set(proxy_bpb.keys()) & set(target_bpb.keys()))
    if len(shared_ids) < 3:
        return {"error": f"only {len(shared_ids)} shared mixtures"}

    px = np.array([proxy_bpb[i] for i in shared_ids])
    tx = np.array([target_bpb[i] for i in shared_ids])

    sp_r, ci_lo, ci_hi = bootstrap_spearman_ci(px, tx)
    pe_r = float(pearsonr(px, tx).statistic)

    return {
        "n_mixtures": len(shared_ids),
        "spearman_r": round(sp_r, 4),
        "pearson_r": round(pe_r, 4),
        "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
    }
