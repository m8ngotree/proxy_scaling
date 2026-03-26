#!/usr/bin/env python3
"""
Generate one OLMo training config per run.
Runs:
  - 1M proxy  x 20 mixtures = 20 configs
  - 15M proxy x 20 mixtures = 20 configs
  - 60M target x 20 mixtures = 20 configs
  - 150M target x 8 mixtures = 8 configs
Total: 68 configs

OLMo controls data mixing by path repetition: each domain's .npy file is
included N times proportional to its weight. We use a resolution of 100
total path entries to approximate the mixture weights.
"""
import json
import math
import yaml
from pathlib import Path
from copy import deepcopy

CHINCHILLA_TOKENS = {
    "proxy_1m": 40_000_000,       # 2x Chinchilla x 1M params x 20 tokens/param
    "proxy_15m": 600_000_000,     # 2x Chinchilla x 15M params x 20 tokens/param
    "target_60m": 1_200_000_000,  # 1x Chinchilla x 60M params x 20 tokens/param
    "target_150m": 3_000_000_000, # 1x Chinchilla x 150M params x 20 tokens/param
}

# Resolution for converting weights to path repetitions
PATH_RESOLUTION = 100


def load_base_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def weights_to_paths(weights: dict, domain_data_dir: str) -> list[str]:
    """Convert domain weights to repeated file paths for OLMo's data loading.

    OLMo samples uniformly from the list of paths, so we repeat each domain's
    path proportionally to its weight. We use PATH_RESOLUTION total entries.
    """
    # Filter out negligible weights
    filtered = {k: v for k, v in weights.items() if v >= 0.001}
    total = sum(filtered.values())
    if total == 0:
        return []

    # Normalize and convert to counts
    counts = {}
    for domain, weight in filtered.items():
        norm_weight = weight / total
        count = max(1, round(norm_weight * PATH_RESOLUTION))
        counts[domain] = count

    # Build path list
    paths = []
    for domain, count in counts.items():
        domain_name = domain.replace("dclm:", "")
        bin_path = f"{domain_data_dir}/{domain_name}/train.npy"
        for _ in range(count):
            paths.append(bin_path)

    return paths


def main():
    Path("run_configs").mkdir(exist_ok=True)
    shared = json.load(open("data/shared_mixtures.json"))
    subset_150m = json.load(open("data/shared_mixtures_150m.json"))
    subset_ids = set(subset_150m["mixture_ids"])

    manifest = []

    run_types = [
        ("proxy_1m", "configs/proxy_1m_base.yaml", "proxy", 1),
        ("proxy_15m", "configs/proxy_15m_base.yaml", "proxy", 15),
        ("target_60m", "configs/target_60m_base.yaml", "target", 60),
        ("target_150m", "configs/target_150m_base.yaml", "target", 150),
    ]

    for run_type, base_config_path, model_class, size_m in run_types:
        base = load_base_config(base_config_path)
        token_budget = CHINCHILLA_TOKENS[run_type]

        # For 150M, only run on the 8-mixture subset
        if run_type == "target_150m":
            mixtures_to_use = [
                m for i, m in enumerate(shared["mixtures"]) if i in subset_ids
            ]
        else:
            mixtures_to_use = shared["mixtures"]

        for mixture in mixtures_to_use:
            mid = mixture["mixture_id"]
            run_name = f"{run_type}_mix{mid:02d}"

            cfg = deepcopy(base)
            cfg["run_name"] = run_name
            cfg["save_folder"] = f"./checkpoints/{run_name}"

            # Set data paths from mixture weights
            data_paths = weights_to_paths(mixture["weights"], "data/domains")
            cfg["data"]["paths"] = data_paths

            # Set max_duration as token count string
            cfg["max_duration"] = f"{token_budget} tokens"

            # Add wandb config
            cfg["wandb"] = {
                "project": "proxy-scaling",
                "name": run_name,
                "tags": [model_class, f"{size_m}m", f"mix{mid:02d}"],
            }

            config_path = f"run_configs/{run_name}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)

            manifest.append(
                {
                    "run_name": run_name,
                    "run_type": run_type,
                    "model_class": model_class,
                    "size_m": size_m,
                    "mixture_id": mid,
                    "token_budget": token_budget,
                    "config_path": config_path,
                    "status": "pending",
                }
            )

    with open("run_configs/manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Generated {len(manifest)} configs")

    # Summary
    from collections import Counter

    counts = Counter(r["run_type"] for r in manifest)
    for rt, n in counts.items():
        print(f"  {rt}: {n} runs")


if __name__ == "__main__":
    main()
