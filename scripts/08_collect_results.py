#!/usr/bin/env python3
"""
Parse OLMES eval outputs and compile results CSV.
Output: results/all_results.csv
"""
import json
import glob
import pandas as pd
import numpy as np
from pathlib import Path


def parse_olmes_output(eval_dir: str) -> dict | None:
    """Parse OLMES output directory for BPB scores."""
    result_files = list(Path(eval_dir).glob("*.json")) + list(
        Path(eval_dir).glob("**/*.json")
    )
    for f in result_files:
        try:
            data = json.load(open(f))
            if "results" not in data:
                continue
            per_task = {}
            for task, metrics in data["results"].items():
                # OLMES reports bpb or loglikelihood metrics
                for k, v in metrics.items():
                    if isinstance(v, float) and ("bpb" in k or "bits" in k):
                        per_task[task] = v
                        break
                    elif (
                        isinstance(v, float)
                        and "acc" in k
                        and "stderr" not in k
                    ):
                        per_task[task] = v  # fallback to accuracy
            if per_task:
                return {
                    "avg_bpb": float(np.mean(list(per_task.values()))),
                    "per_task": per_task,
                }
        except Exception:
            continue
    return None


def main():
    Path("results").mkdir(exist_ok=True)
    manifest = json.load(open("run_configs/manifest.json"))
    shared = json.load(open("data/shared_mixtures.json"))
    proxy30 = pd.read_csv("data/proxy_30m_bpb.csv")

    rows = []

    # Released 30M proxy data
    for _, row in proxy30.iterrows():
        rows.append(
            {
                "run_name": f"proxy_30m_mix{int(row['mixture_id']):02d}",
                "run_type": "proxy_30m",
                "model_class": "proxy",
                "size_m": 30,
                "mixture_id": int(row["mixture_id"]),
                "avg_bpb": row["avg_bpb"],
                "source": "released",
            }
        )

    # Newly trained models
    for run in manifest:
        name = run["run_name"]
        eval_dir = f"results/eval/{name}"
        result = parse_olmes_output(eval_dir)
        if result is None:
            print(f"WARNING: no eval result for {name}")
            continue
        rows.append(
            {
                "run_name": name,
                "run_type": run["run_type"],
                "model_class": run["model_class"],
                "size_m": run["size_m"],
                "mixture_id": run["mixture_id"],
                "avg_bpb": result["avg_bpb"],
                "source": "trained",
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv("results/all_results.csv", index=False)
    print(f"Compiled {len(df)} results to results/all_results.csv")

    # Summary
    for run_type in df["run_type"].unique():
        sub = df[df["run_type"] == run_type]
        print(
            f"  {run_type}: {len(sub)} rows, "
            f"avg_bpb range [{sub['avg_bpb'].min():.3f}, {sub['avg_bpb'].max():.3f}]"
        )


if __name__ == "__main__":
    main()
