#!/usr/bin/env python3
"""
Select 20 diverse mixtures from the Olmix 30M proxy swarm.
Outputs (committed to git):
  data/shared_mixtures.json        - 20 mixture vectors + run IDs
  data/shared_mixtures_150m.json   - 8-mixture subset for 150M target
  data/proxy_30m_bpb.csv           - 30M proxy BPB for the 20 mixtures
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

# These paths are relative to the repo root.
# Verify against data/dataset_structure.md after running script 00.
RATIOS_PATH = "data/olmix_release/dclm_swarm/ratios.csv"
METRICS_PATH = "data/olmix_release/dclm_swarm/metrics.csv"
META_PATH = "data/olmix_release/dclm_swarm/meta.json"

N_SHARED = 20  # mixtures for 1M, 15M proxies and 60M targets
N_SUBSET = 8  # mixtures for 150M targets (cost constraint)
SEED = 42


def greedy_diverse_select(weights: np.ndarray, n: int, seed: int) -> list[int]:
    """Greedy farthest-point selection for maximum pairwise L1 diversity."""
    rng = np.random.default_rng(seed)
    selected = [int(rng.integers(0, len(weights)))]
    while len(selected) < n:
        min_dists = np.full(len(weights), np.inf)
        for s in selected:
            dists = np.abs(weights - weights[s]).sum(axis=1)
            min_dists = np.minimum(min_dists, dists)
        min_dists[selected] = -1
        selected.append(int(np.argmax(min_dists)))
    return selected


def main():
    Path("data").mkdir(exist_ok=True)

    ratios_df = pd.read_csv(RATIOS_PATH)
    metrics_df = pd.read_csv(METRICS_PATH)
    meta = json.load(open(META_PATH))

    id_col = "run" if "run" in ratios_df.columns else "run_id"
    # Domain columns are any column that isn't an ID/metadata column.
    # The actual CSV uses bare topic names (no "dclm:" prefix).
    NON_DOMAIN_COLS = {id_col, "name", "index"}
    raw_domain_cols = [c for c in ratios_df.columns if c not in NON_DOMAIN_COLS]
    # Normalise to "dclm:{topic}" format to match the rest of the codebase.
    # Build a mapping from raw name -> dclm: name.
    domain_col_map = {}
    for c in raw_domain_cols:
        if c.startswith("dclm:"):
            domain_col_map[c] = c
        else:
            domain_col_map[c] = f"dclm:{c}"
    # Rename in dataframe so downstream code sees "dclm:" prefixed columns.
    ratios_df = ratios_df.rename(columns=domain_col_map)
    domain_cols = list(domain_col_map.values())

    print(f"Loaded {len(ratios_df)} proxy runs")
    print(f"Domain columns ({len(domain_cols)}): {domain_cols}")

    # Validate: all rows sum to ~1
    weights = ratios_df[domain_cols].values
    sums = weights.sum(axis=1)
    valid_mask = np.abs(sums - 1.0) < 0.01
    if not valid_mask.all():
        print(
            f"WARNING: {(~valid_mask).sum()} rows invalid (weights don't sum to 1)"
        )
    ratios_df = ratios_df[valid_mask].reset_index(drop=True)
    metrics_df = metrics_df[
        metrics_df[id_col].isin(ratios_df[id_col])
    ].reset_index(drop=True)
    weights = ratios_df[domain_cols].values

    # Select 20 diverse mixtures
    selected_idx = greedy_diverse_select(weights, N_SHARED, SEED)
    selected_ratios = ratios_df.iloc[selected_idx].reset_index(drop=True)
    selected_run_ids = selected_ratios[id_col].tolist()

    # Print diversity stats
    sel_weights = weights[selected_idx]
    pairwise_l1 = [
        np.abs(sel_weights[i] - sel_weights[j]).sum()
        for i in range(N_SHARED)
        for j in range(i + 1, N_SHARED)
    ]
    print(f"\nSelected {N_SHARED} mixtures")
    print(
        f"Pairwise L1 diversity - min: {min(pairwise_l1):.3f}, "
        f"mean: {np.mean(pairwise_l1):.3f}, max: {max(pairwise_l1):.3f}"
    )

    # Select 8-mixture subset from the 20 (for 150M targets)
    subset_idx = greedy_diverse_select(sel_weights, N_SUBSET, SEED)
    subset_run_ids = [selected_run_ids[i] for i in subset_idx]

    # Get 30M proxy BPB for selected mixtures
    metric_cols = [
        c for c in metrics_df.columns if c not in [id_col, "name", "index"]
    ]
    selected_metrics = (
        metrics_df[metrics_df[id_col].isin(selected_run_ids)]
        .set_index(id_col)
        .loc[selected_run_ids]  # preserve order
        .reset_index()
    )

    # Compute avg BPB across all tasks
    selected_metrics["avg_bpb"] = selected_metrics[metric_cols].mean(axis=1)

    # Save shared_mixtures.json
    shared = {
        "n_mixtures": N_SHARED,
        "domain_cols": domain_cols,
        "id_column": id_col,
        "natural_distribution": meta.get("relative_sizes", {}),
        "token_counts": meta.get("token_counts", {}),
        "mixtures": [
            {
                "mixture_id": i,
                "run_id": selected_run_ids[i],
                "weights": {
                    col: float(selected_ratios.iloc[i][col]) for col in domain_cols
                },
            }
            for i in range(N_SHARED)
        ],
    }
    with open("data/shared_mixtures.json", "w") as f:
        json.dump(shared, f, indent=2)

    # Save 150M subset
    subset = {
        "n_mixtures": N_SUBSET,
        "mixture_ids": subset_idx,
        "run_ids": subset_run_ids,
        "note": "Subset of shared_mixtures.json for 150M target training (cost constraint)",
    }
    with open("data/shared_mixtures_150m.json", "w") as f:
        json.dump(subset, f, indent=2)

    # Save 30M proxy BPB
    proxy_30m = selected_metrics[[id_col] + metric_cols + ["avg_bpb"]].copy()
    proxy_30m.insert(0, "mixture_id", range(N_SHARED))
    proxy_30m.to_csv("data/proxy_30m_bpb.csv", index=False)

    print(f"\nSaved data/shared_mixtures.json ({N_SHARED} mixtures)")
    print(f"Saved data/shared_mixtures_150m.json ({N_SUBSET} mixtures)")
    print(f"Saved data/proxy_30m_bpb.csv")
    print(f"\nCommit these three files to git before starting any training.")


if __name__ == "__main__":
    main()
