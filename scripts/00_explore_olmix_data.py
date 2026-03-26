#!/usr/bin/env python3
"""
Explore allenai/olmix dataset structure.
Run first. Output: data/dataset_structure.md
"""
from huggingface_hub import list_repo_files, hf_hub_download
import pandas as pd
import json
from pathlib import Path


def main():
    Path("data").mkdir(exist_ok=True)
    print("Listing files in allenai/olmix...")
    files = list_repo_files("allenai/olmix", repo_type="dataset")

    ratio_files = [f for f in files if f.endswith("ratios.csv")]
    metrics_files = [f for f in files if f.endswith("metrics.csv")]
    meta_files = [f for f in files if f.endswith("meta.json")]

    lines = [
        "# Olmix Dataset Structure\n\n",
        f"Total files: {len(files)}\n",
        f"ratios.csv files: {len(ratio_files)}\n",
        f"metrics.csv files: {len(metrics_files)}\n\n",
        "## All files:\n",
    ]
    for f in files:
        lines.append(f"  {f}\n")
    lines.append("\n## Swarm details:\n")

    for ratio_path in ratio_files:
        try:
            local = hf_hub_download("allenai/olmix", ratio_path, repo_type="dataset")
            df = pd.read_csv(local)
            domain_cols = [
                c
                for c in df.columns
                if c not in ["run", "run_id", "name", "index"]
            ]
            dclm_cols = [c for c in domain_cols if c.startswith("dclm:")]
            lines.append(f"\n### {ratio_path}\n")
            lines.append(f"- Rows: {len(df)}\n")
            lines.append(f"- Domain columns ({len(domain_cols)}): {domain_cols}\n")
            lines.append(f"- DCLM columns ({len(dclm_cols)}): {dclm_cols}\n")
            lines.append(f"- Sample row 0 domain weights:\n")
            for col in domain_cols[:6]:
                lines.append(f"    {col}: {df.iloc[0][col]:.4f}\n")
            print(f"  {ratio_path}: {len(df)} rows, {len(domain_cols)} domains")
        except Exception as e:
            lines.append(f"\n### {ratio_path}\n- ERROR: {e}\n")

    for meta_path in meta_files[:5]:
        try:
            local = hf_hub_download("allenai/olmix", meta_path, repo_type="dataset")
            meta = json.load(open(local))
            lines.append(f"\n### {meta_path}\n")
            lines.append(f"- Keys: {list(meta.keys())}\n")
            if "relative_sizes" in meta:
                lines.append(
                    f"- relative_sizes (natural distribution): "
                    f"{meta['relative_sizes']}\n"
                )
            if "token_counts" in meta:
                lines.append(
                    f"- token_counts sample: "
                    f"{dict(list(meta['token_counts'].items())[:4])}\n"
                )
        except Exception as e:
            lines.append(f"\n### {meta_path}\n- ERROR: {e}\n")

    with open("data/dataset_structure.md", "w") as f:
        f.writelines(lines)

    print("\nSaved to data/dataset_structure.md")
    print("Read this file before running script 02.")


if __name__ == "__main__":
    main()
