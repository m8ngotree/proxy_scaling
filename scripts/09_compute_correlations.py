#!/usr/bin/env python3
"""
Compute Spearman rank correlation between proxy and target BPB across mixtures.

Result: 3x2 matrix (3 proxy sizes x 2 new target sizes)
Plus the paper's values for T=1B (3rd column, free).

Output: results/correlation_matrix.json
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import json
import argparse
from pathlib import Path

# Paper's reported values (Figure 3, Olmix paper) for proxy vs 1B target
PAPER_1B_VALUES = {1: 0.73, 15: 0.89, 30: 0.896}


def bootstrap_ci(x, y, n=1000, seed=42):
    """Bootstrap 95% confidence interval for Spearman correlation."""
    rng = np.random.default_rng(seed)
    rs = []
    for _ in range(n):
        idx = rng.integers(0, len(x), len(x))
        r = spearmanr(x[idx], y[idx]).statistic
        if not np.isnan(r):
            rs.append(r)
    if not rs:
        return 0.0, 0.0
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))


def make_synthetic_data():
    """Synthetic data for smoke test mode."""
    rng = np.random.default_rng(42)
    n = 20
    true_perf = rng.normal(0, 1, n)
    results = []

    # Proxy rows
    for size_m, run_type_name in [(1, "proxy_1m"), (15, "proxy_15m"), (30, "proxy_30m")]:
        noise = rng.normal(0, 0.3, n)
        bpb = 2.0 + true_perf * 0.2 + noise
        for i in range(n):
            results.append(
                {
                    "run_name": f"{run_type_name}_mix{i:02d}",
                    "run_type": run_type_name,
                    "model_class": "proxy",
                    "size_m": size_m,
                    "mixture_id": i,
                    "avg_bpb": float(bpb[i]),
                }
            )

    # Target rows
    for target_m, run_type_name in [(60, "target_60m"), (150, "target_150m")]:
        noise = rng.normal(0, 0.1, n)
        bpb = 1.5 + true_perf * 0.15 + noise
        n_mix = n if target_m == 60 else 8
        for i in range(n_mix):
            results.append(
                {
                    "run_name": f"{run_type_name}_mix{i:02d}",
                    "run_type": run_type_name,
                    "model_class": "target",
                    "size_m": target_m,
                    "mixture_id": i,
                    "avg_bpb": float(bpb[i]),
                }
            )

    return pd.DataFrame(results)


def compute_matrix(df):
    """Compute the correlation matrix between all proxy-target pairs."""
    proxy_sizes = [1, 15, 30]
    target_sizes = [60, 150]
    matrix = {}

    for p in proxy_sizes:
        proxy_type = f"proxy_{p}m" if p != 30 else "proxy_30m"
        proxy_df = (
            df[df["run_type"] == proxy_type]
            .sort_values("mixture_id")
            .set_index("mixture_id")
        )

        for t in target_sizes:
            target_df = (
                df[df["run_type"] == f"target_{t}m"]
                .sort_values("mixture_id")
                .set_index("mixture_id")
            )

            shared_ids = sorted(set(proxy_df.index) & set(target_df.index))
            if len(shared_ids) < 5:
                print(
                    f"  WARNING: only {len(shared_ids)} shared mixtures for "
                    f"P={p}M, T={t}M - skipping"
                )
                continue

            px = proxy_df.loc[shared_ids, "avg_bpb"].values
            tx = target_df.loc[shared_ids, "avg_bpb"].values

            sp_r = float(spearmanr(px, tx).statistic)
            pe_r = float(pearsonr(px, tx).statistic)
            ci_lo, ci_hi = bootstrap_ci(px, tx)

            key = f"P{p}M_T{t}M"
            matrix[key] = {
                "proxy_size_m": p,
                "target_size_m": t,
                "n_mixtures": len(shared_ids),
                "spearman_r": round(sp_r, 4),
                "pearson_r": round(pe_r, 4),
                "ci_95": [round(ci_lo, 4), round(ci_hi, 4)],
            }
            print(
                f"  P={p:2d}M vs T={t}M: Spearman={sp_r:.3f} "
                f"[{ci_lo:.3f},{ci_hi:.3f}] Pearson={pe_r:.3f} (n={len(shared_ids)})"
            )

    # Add paper's 1B column
    for p in proxy_sizes:
        if p in PAPER_1B_VALUES:
            matrix[f"P{p}M_T1000M_paper"] = {
                "proxy_size_m": p,
                "target_size_m": 1000,
                "source": "Olmix paper Figure 3",
                "spearman_r": PAPER_1B_VALUES[p],
            }

    return matrix


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    if args.synthetic:
        print("SYNTHETIC MODE")
        df = make_synthetic_data()
    else:
        df = pd.read_csv("results/all_results.csv")

    print("\nCorrelation matrix:")
    matrix = compute_matrix(df)

    json.dump(matrix, open("results/correlation_matrix.json", "w"), indent=2)
    print("\nSaved results/correlation_matrix.json")

    # Print summary table
    print("\n" + "=" * 60)
    print(f"{'':15} {'T=60M':>12} {'T=150M':>12} {'T=1B (paper)':>14}")
    print("-" * 60)
    for p in [1, 15, 30]:
        row = f"P={p:2d}M proxy   "
        for t in [60, 150]:
            k = f"P{p}M_T{t}M"
            if k in matrix:
                row += f"  {matrix[k]['spearman_r']:>8.3f}    "
            else:
                row += f"  {'N/A':>8}    "
        row += f"  {PAPER_1B_VALUES.get(p, 'N/A'):>8}"
        print(row)
    print("=" * 60)


if __name__ == "__main__":
    main()
