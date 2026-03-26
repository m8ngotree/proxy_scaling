# Experiment Spec: Proxy-to-Target Size Scaling Laws for Data Mixing
# Version 4 — Final, $100 Budget

---

## How to Use This Document

This is a complete, self-contained specification. To implement it:
1. Create a new git repository
2. Copy this file in as `SPEC.md`
3. Tell Claude Code: "Read SPEC.md and implement everything described in the
   Implementation section. Do not skip any script. Read external codebases before
   writing configs that depend on them."

Claude Code should never ask for clarification on something specified here. Make a
reasonable decision and leave a comment explaining it.

---

## Research Question

The Olmix paper (arXiv 2602.12237) studies proxy model reliability for data mixing.
Its RQ1 asks: what is the minimum proxy model size for reliable mixture decisions?
The paper finds ~15M parameters is the threshold — but tests this only against a
single 1B target model.

This experiment extends RQ1 in two directions simultaneously:

1. **Vary target size**: does the rank correlation between proxy and target change
   when the target is 60M or 150M instead of 1B?

2. **Vary proxy size**: for each target size, does the threshold shift? Is 15M still
   sufficient for a 60M target, or does a smaller proxy work just as well?

The result is a 3×3 correlation matrix:

```
              T=60M    T=150M    T=1B (paper)
P=1M          ρ(1,60)  ρ(1,150)  0.73 (paper)
P=15M         ρ(15,60) ρ(15,150) 0.89 (paper)
P=30M         ρ(30,60) ρ(30,150) 0.896 (paper)
```

The right column is free — taken from the Olmix paper's Figure 3. We measure the
left two columns by training new target models and new small proxy models.

From this matrix we can answer: does the threshold scale with target size, or is
15M a universal floor? If ρ(1,60) is high (above 0.85), even 1M proxies suffice
for 60M targets — a practically important finding for researchers training small models.

---

## Why This Is Feasible for ~$100

**Proxy side — partially free:**
The Olmix team released their proxy swarm data at `allenai/olmix` on HuggingFace.
The main swarm (`dclm_swarm/`) contains **128 runs of 30M proxy models** with full
BPB evaluation results across 52 tasks. We use 20 of these runs as our shared
mixture set — the 30M proxy row of our matrix is free.

We still need to train 1M and 15M proxies on those same 20 mixtures. These are
very cheap.

**Target side — requires training:**
We train 60M and 150M target models on the same 20 mixtures. The 1B column is
taken from the paper.

**Cost breakdown:**

| Item | Runs | Time/run | GPU-hrs | Cost @ $2/hr |
|------|------|----------|---------|--------------|
| 1M proxy × 20 mixtures | 20 | 5 min | 1.7 | $3 |
| 15M proxy × 20 mixtures | 20 | 25 min | 8.3 | $17 |
| 60M target × 20 mixtures | 20 | 90 min | 30 | $60 |
| 150M target × 8 mixtures* | 8 | 5 hrs | 40 | $80 |
| Smoke test | 2 | varies | 3 | $6 |
| **Total** | | | **~83 hrs** | **~$166** |

*We use only 8 of the 20 mixtures for 150M targets to control cost. This is
sufficient for rank correlation with reasonable confidence intervals.

This comes to ~$166, slightly over $100. To stay under $100, reduce 60M targets
from 20 to 12 runs (saves ~$36) and 150M targets from 8 to 6 runs (saves ~$20),
bringing total to ~$110. The spec below uses the fuller design; budget adjustments
are noted where relevant.

---

## What We Know About the Released Data

From inspecting `allenai/olmix` on HuggingFace:

**Main swarm** (`dclm_swarm/`):
- 128 proxy runs, 30M proxy models, 5x Chinchilla (3B tokens)
- 24 DCLM topic domains: `dclm:politics`, `dclm:health`, `dclm:software_development`,
  etc. (full list in paper Table 8)
- Evaluation: 52 tasks in BPB using OLMES harness, task names like
  `arc_challenge:rc::olmes`, `hellaswag:rc::olmes`, `minerva_math_algebra::olmes`
- Files: `ratios.csv` (mixture weights), `metrics.csv` (BPB scores), `meta.json`
  (domain token counts and natural distribution)
- Run names follow pattern: `backfill-5xC-30m-dclm-s2pdf-flat-{hash}-{index}`

**Additional swarms in `study/rq2/`**:
- `dclm_swarm_6_topics/`: 125 runs, 6-topic subset
- `dclm_swarm_12_topics/`: 130 runs, 12-topic subset
- `dclm_swarm_18_topics/`: 129 runs, 18-topic subset

**Important**: only 30M proxy models are released. Swarms for 1M and 15M proxy
sizes were not released. We train these ourselves.

**meta.json structure** (from README): contains `relative_sizes` (natural distribution)
and `token_counts` (per-domain token counts for repetition constraints).

---

## Shared Mixture Set

We select **20 mixtures** from the released 30M proxy swarm. These same 20 mixtures
are used for all training runs (1M proxies, 15M proxies, 60M targets, 150M targets).
This is what makes the correlation measurement possible.

Selection: choose the 20 mixtures from the 128-run swarm that maximize diversity
(maximum sum of pairwise L1 distances between mixture vectors). This ensures the
correlation signal is not suppressed by near-identical mixtures.

For the 150M target (budget constraint), use a subset of 8 of the 20 shared mixtures.
Choose the 8 with the greatest pairwise diversity from each other.

---

## Model Architectures

All models use the OLMo 2 decoder-only transformer architecture. Before writing
any configs, Claude Code must clone `github.com/allenai/OLMo` and read existing
configs in `OLMo/configs/` to understand the exact YAML format.

### Proxy models (train from scratch on each of 20 mixtures)

From Olmix paper Table 6:

| Size | n_layers | n_heads | d_model | head_dim | vocab_size |
|------|----------|---------|---------|----------|------------|
| 1M   | 4        | 4       | 16      | 4        | 100,352    |
| 15M  | 8        | 4       | 128     | 32       | 100,352    |

All proxy models:
- sequence_length: 2048
- batch_size: 64
- max_learning_rate: 0.007
- lr_scheduler: cosine with warmup (500 steps)
- optimizer: AdamW, betas=[0.9, 0.95], weight_decay=0.1, eps=1e-8
- tokenizer: `allenai/dolma2-tokenizer`
- precision: amp_bf16

Token budgets (2x Chinchilla = 40 tokens/parameter):

| Size | Tokens |
|------|--------|
| 1M   | 40M    |
| 15M  | 600M   |

Note: the released 30M proxy uses 5x Chinchilla (3B tokens). We use 2x for our
proxies to save cost. This is a methodological difference to acknowledge — lower
token budgets may depress correlation values. We mitigate this by noting that the
paper's finding (ρ jumps at 15M) should still be detectable even if absolute
correlation values are slightly lower.

### Target models (train on 20 mixtures for 60M, 8 mixtures for 150M)

| Size  | n_layers | n_heads | d_model | head_dim | vocab_size |
|-------|----------|---------|---------|----------|------------|
| 60M   | 8        | 8       | 384     | 48       | 100,352    |
| 150M  | 12       | 12      | 512     | 64       | 100,352    |

All target models:
- sequence_length: 4096
- batch_size: 256
- max_learning_rate: 0.0018
- lr_scheduler: cosine with warmup (1000 steps)
- optimizer: AdamW, betas=[0.9, 0.95], weight_decay=0.1, eps=1e-8
- tokenizer: `allenai/dolma2-tokenizer`
- precision: amp_bf16
- device_train_microbatch_size: 8 (adjust down if OOM)

Token budgets (1x Chinchilla):

| Size  | Tokens |
|-------|--------|
| 60M   | 1.2B   |
| 150M  | 3B     |

---

## Training Data

Training data is DCLM partitioned into 24 topic domains using WebOrganizer, matching
the released proxy swarm exactly. Domain names follow the format `dclm:{topic}`.

**Check first**: the `meta.json` in `dclm_swarm/` contains `token_counts` per domain.
This tells you how much data is available per domain and whether DCLM needs to be
downloaded or is available pre-processed.

Claude Code must check whether the Olmix repository or HuggingFace release includes
pre-tokenized DCLM domain splits. If yes, use them directly. If not:
- Download: `mlfoundations/dclm-baseline-1.0` on HuggingFace
- Apply WebOrganizer topic labels: `github.com/allenai/weborganizer`
- Tokenize with Dolma 2 tokenizer
- Save as numpy memmap `.bin` files: `data/domains/{topic}/train.bin`

For the smoke test, prepare only 200M tokens per domain.
For full experiment, prepare at least 3B tokens per domain (to avoid repetition
in 150M target runs).

---

## Evaluation

Use OLMES (`github.com/allenai/olmes`) for all evaluation. This is the exact
evaluation harness used by the Olmix paper. The task names in the released
`metrics.csv` (e.g. `arc_challenge:rc::olmes`) are OLMES task identifiers.

Install: `git clone https://github.com/allenai/olmes && cd olmes && pip install -e .`

Run evaluation on a checkpoint:
```bash
olmes \
  --model path/to/checkpoint \
  --task arc_challenge::olmes arc_easy::olmes hellaswag::olmes winogrande::olmes \
        piqa::olmes csqa::olmes minerva_math_algebra::olmes \
        codex_humaneval::olmes mmlu::olmes \
  --output-dir results/eval/{run_name}/
```

Use this task subset (9 tasks covering math, code, and QA) rather than the full
52-task suite. This reduces evaluation time significantly while preserving coverage
of the capability families. The average BPB across these 9 tasks is our primary
metric.

For proxy models, skip evaluation entirely — we use the BPB scores from the
released `metrics.csv` for the 30M proxies, and for 1M/15M proxies we need to
run OLMES on the same task subset so results are comparable.

Store each model's evaluation output at:
`results/eval/{run_name}_eval.json`

with format:
```json
{
  "run_name": "proxy_1m_mixture_07",
  "model_type": "proxy",
  "proxy_size_m": 1,
  "mixture_id": 7,
  "avg_bpb": 1.823,
  "per_task_bpb": {
    "arc_challenge:rc::olmes": 1.641,
    "hellaswag:rc::olmes": 1.238,
    ...
  }
}
```

---

## Phase Structure

### Phase 1: Local (no GPU, no cost)
Claude Code implements all scripts. All scripts that don't require training run
locally. Scripts that require training accept a `--dry-run` flag that prints
what they would do without doing it.

### Phase 2: Smoke test (~$6, ~3 GPU-hours)
Rent A100 on RunPod or Lambda Labs. Run one training run of each model size
at minimal token budget to verify the pipeline end-to-end.

### Phase 3: Full experiment (~$160, ~80 GPU-hours)
Run all training, evaluation, analysis, and figure generation.

---

## Implementation

Claude Code implements the following scripts in order. Do not skip ahead — each
script depends on the previous ones being correct.

### Repository structure

```
proxy_scaling/
├── SPEC.md
├── README.md
├── requirements.txt
├── setup.sh
├── data/
│   ├── olmix_release/               ← downloaded from HuggingFace (gitignored)
│   ├── domains/                     ← tokenized DCLM domain data (gitignored)
│   ├── dataset_structure.md         ← generated by script 00
│   ├── shared_mixtures.json         ← 20 selected mixtures (committed)
│   ├── shared_mixtures_150m.json    ← 8-mixture subset for 150M target (committed)
│   └── proxy_30m_bpb.csv            ← 30M proxy BPB for 20 mixtures (committed)
├── configs/
│   ├── proxy_1m_base.yaml
│   ├── proxy_15m_base.yaml
│   ├── target_60m_base.yaml
│   └── target_150m_base.yaml
├── run_configs/                     ← generated by script 04 (gitignored)
├── scripts/
│   ├── 00_explore_olmix_data.py
│   ├── 01_download_olmix_data.py
│   ├── 02_select_mixtures.py
│   ├── 03_prepare_dclm_data.sh
│   ├── 04_generate_run_configs.py
│   ├── 05_smoke_test.sh
│   ├── 06_run_all_training.sh
│   ├── 07_run_evaluation.sh
│   ├── 08_collect_results.py
│   ├── 09_compute_correlations.py
│   ├── 10_fit_scaling_law.py
│   └── 11_make_figures.py
├── src/
│   ├── data_mixer.py
│   ├── correlation.py
│   └── scaling_law.py
└── results/                         ← gitignored
```

---

### Script 00: explore_olmix_data.py

Explore the released dataset structure and write a summary to
`data/dataset_structure.md`. This must run successfully before any other script.

```python
#!/usr/bin/env python3
"""
Explore allenai/olmix dataset structure.
Run first. Output: data/dataset_structure.md
"""
from huggingface_hub import list_repo_files, hf_hub_download
import pandas as pd
from pathlib import Path

def main():
    Path("data").mkdir(exist_ok=True)
    print("Listing files in allenai/olmix...")
    files = sorted(list_repo_files("allenai/olmix", repo_type="dataset"))

    ratio_files = [f for f in files if f.endswith("ratios.csv")]
    metrics_files = [f for f in files if f.endswith("metrics.csv")]
    meta_files = [f for f in files if f.endswith("meta.json")]

    lines = [
        "# Olmix Dataset Structure\n\n",
        f"Total files: {len(files)}\n",
        f"ratios.csv files: {len(ratio_files)}\n",
        f"metrics.csv files: {len(metrics_files)}\n\n",
        "## All files:\n"
    ]
    for f in files:
        lines.append(f"  {f}\n")
    lines.append("\n## Swarm details:\n")

    for ratio_path in ratio_files:
        try:
            local = hf_hub_download("allenai/olmix", ratio_path, repo_type="dataset")
            df = pd.read_csv(local)
            domain_cols = [c for c in df.columns
                           if c not in ["run", "run_id", "name", "index"]]
            dclm_cols = [c for c in domain_cols if c.startswith("dclm:")]
            lines.append(f"\n### {ratio_path}\n")
            lines.append(f"- Rows: {len(df)}\n")
            lines.append(f"- Domain columns ({len(domain_cols)}): {domain_cols}\n")
            lines.append(f"- DCLM columns ({len(dclm_cols)}): {dclm_cols}\n")
            # Print first row as sample
            lines.append(f"- Sample row 0 domain weights:\n")
            for col in domain_cols[:6]:
                lines.append(f"    {col}: {df.iloc[0][col]:.4f}\n")
            print(f"  {ratio_path}: {len(df)} rows, {len(domain_cols)} domains")
        except Exception as e:
            lines.append(f"\n### {ratio_path}\n- ERROR: {e}\n")

    for meta_path in meta_files[:5]:  # first 5 only
        try:
            import json
            local = hf_hub_download("allenai/olmix", meta_path, repo_type="dataset")
            meta = json.load(open(local))
            lines.append(f"\n### {meta_path}\n")
            lines.append(f"- Keys: {list(meta.keys())}\n")
            if "relative_sizes" in meta:
                lines.append(f"- relative_sizes (natural distribution): "
                              f"{meta['relative_sizes']}\n")
            if "token_counts" in meta:
                lines.append(f"- token_counts sample: "
                              f"{dict(list(meta['token_counts'].items())[:4])}\n")
        except Exception as e:
            lines.append(f"\n### {meta_path}\n- ERROR: {e}\n")

    with open("data/dataset_structure.md", "w") as f:
        f.writelines(lines)

    print("\nSaved to data/dataset_structure.md")
    print("Read this file before running script 02.")

if __name__ == "__main__":
    main()
```

---

### Script 01: download_olmix_data.py

```python
#!/usr/bin/env python3
"""Download allenai/olmix dataset to data/olmix_release/."""
from huggingface_hub import snapshot_download

def main():
    print("Downloading allenai/olmix...")
    path = snapshot_download(
        repo_id="allenai/olmix",
        repo_type="dataset",
        local_dir="data/olmix_release",
        ignore_patterns=["*.bin", "*.pt", "*.safetensors"]
    )
    print(f"Downloaded to: {path}")

if __name__ == "__main__":
    main()
```

---

### Script 02: select_mixtures.py

Select 20 diverse mixtures from the main 30M proxy swarm. Save the mixture
vectors, the run IDs, and the corresponding 30M proxy BPB scores.

This script determines the shared mixture set for the entire experiment.
Run it once and commit the outputs. Never rerun after training has started.

The main swarm path is `dclm_swarm/ratios.csv` and `dclm_swarm/metrics.csv`.
Verify this against `data/dataset_structure.md` after running script 00.

```python
#!/usr/bin/env python3
"""
Select 20 diverse mixtures from the Olmix 30M proxy swarm.
Outputs (committed to git):
  data/shared_mixtures.json        — 20 mixture vectors + run IDs
  data/shared_mixtures_150m.json   — 8-mixture subset for 150M target
  data/proxy_30m_bpb.csv           — 30M proxy BPB for the 20 mixtures
"""
import pandas as pd
import numpy as np
import json
from pathlib import Path

# These paths are relative to data/olmix_release/
# Verify against data/dataset_structure.md after running script 00
RATIOS_PATH = "data/olmix_release/dclm_swarm/ratios.csv"
METRICS_PATH = "data/olmix_release/dclm_swarm/metrics.csv"
META_PATH    = "data/olmix_release/dclm_swarm/meta.json"

N_SHARED    = 20   # mixtures for 1M, 15M proxies and 60M targets
N_SUBSET    = 8    # mixtures for 150M targets (cost constraint)
SEED        = 42

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

    ratios_df  = pd.read_csv(RATIOS_PATH)
    metrics_df = pd.read_csv(METRICS_PATH)
    meta       = json.load(open(META_PATH))

    id_col = "run" if "run" in ratios_df.columns else "run_id"
    domain_cols = [c for c in ratios_df.columns
                   if c not in [id_col, "name", "index"]
                   and c.startswith("dclm:")]

    print(f"Loaded {len(ratios_df)} proxy runs")
    print(f"Domain columns ({len(domain_cols)}): {domain_cols}")

    # Validate: all rows sum to ~1
    weights = ratios_df[domain_cols].values
    sums = weights.sum(axis=1)
    valid_mask = np.abs(sums - 1.0) < 0.01
    if not valid_mask.all():
        print(f"WARNING: {(~valid_mask).sum()} rows invalid (weights don't sum to 1)")
    ratios_df  = ratios_df[valid_mask].reset_index(drop=True)
    metrics_df = metrics_df[metrics_df[id_col].isin(ratios_df[id_col])].reset_index(drop=True)
    weights    = ratios_df[domain_cols].values

    # Select 20 diverse mixtures
    selected_idx = greedy_diverse_select(weights, N_SHARED, SEED)
    selected_ratios  = ratios_df.iloc[selected_idx].reset_index(drop=True)
    selected_run_ids = selected_ratios[id_col].tolist()

    # Print diversity stats
    sel_weights = weights[selected_idx]
    pairwise_l1 = [np.abs(sel_weights[i] - sel_weights[j]).sum()
                   for i in range(N_SHARED) for j in range(i+1, N_SHARED)]
    print(f"\nSelected {N_SHARED} mixtures")
    print(f"Pairwise L1 diversity — min: {min(pairwise_l1):.3f}, "
          f"mean: {np.mean(pairwise_l1):.3f}, max: {max(pairwise_l1):.3f}")

    # Select 8-mixture subset from the 20 (for 150M targets)
    subset_idx = greedy_diverse_select(sel_weights, N_SUBSET, SEED)
    subset_run_ids = [selected_run_ids[i] for i in subset_idx]

    # Get 30M proxy BPB for selected mixtures
    metric_cols = [c for c in metrics_df.columns
                   if c not in [id_col, "name", "index"]]
    selected_metrics = (metrics_df[metrics_df[id_col].isin(selected_run_ids)]
                        .set_index(id_col)
                        .loc[selected_run_ids]  # preserve order
                        .reset_index())

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
                    col: float(selected_ratios.iloc[i][col])
                    for col in domain_cols
                }
            }
            for i in range(N_SHARED)
        ]
    }
    with open("data/shared_mixtures.json", "w") as f:
        json.dump(shared, f, indent=2)

    # Save 150M subset
    subset = {
        "n_mixtures": N_SUBSET,
        "mixture_ids": subset_idx,
        "run_ids": subset_run_ids,
        "note": "Subset of shared_mixtures.json for 150M target training (cost constraint)"
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
```

---

### Script 03: prepare_dclm_data.sh

```bash
#!/bin/bash
set -e

SMOKE_TEST=${1:-""}
TOKENS_PER_DOMAIN=3000000000  # 3B tokens per domain for full experiment
if [ "$SMOKE_TEST" = "--smoke-test" ]; then
    TOKENS_PER_DOMAIN=200000000  # 200M for smoke test
    echo "SMOKE TEST MODE: 200M tokens per domain"
fi

echo "=== Preparing DCLM domain data ==="

# Check if meta.json has token_counts pointing to existing data
python3 - << 'PYEOF'
import json, os
from pathlib import Path

meta = json.load(open("data/olmix_release/dclm_swarm/meta.json"))
shared = json.load(open("data/shared_mixtures.json"))
domains = [d.replace("dclm:", "") for d in shared["domain_cols"]
           if d.startswith("dclm:")]

print(f"Need data for {len(domains)} domains: {domains}")
print("\nChecking if tokenized data already exists in Olmix release...")

# Look for any .bin files in the release
bin_files = list(Path("data/olmix_release").rglob("*.bin"))
if bin_files:
    print(f"Found {len(bin_files)} .bin files — may be pre-tokenized data")
    for f in bin_files[:5]:
        print(f"  {f}")
else:
    print("No .bin files found. Will need to download and tokenize DCLM.")

# Print token counts from meta.json
if "token_counts" in meta:
    print("\nToken counts from meta.json:")
    for domain, count in meta["token_counts"].items():
        if domain.startswith("dclm:"):
            print(f"  {domain}: {count:,}")
PYEOF

echo ""
echo "If tokenized data is not available, downloading DCLM..."
echo "This requires significant disk space (500GB+) and time."
echo ""

# Create domain directories
python3 - << 'PYEOF'
import json
from pathlib import Path

shared = json.load(open("data/shared_mixtures.json"))
for domain_col in shared["domain_cols"]:
    if domain_col.startswith("dclm:"):
        domain = domain_col.replace("dclm:", "")
        Path(f"data/domains/{domain}").mkdir(parents=True, exist_ok=True)
print("Created domain directories")
PYEOF

# Main data preparation — download DCLM and tokenize if needed
python3 scripts/tokenize_dclm_domains.py \
    --tokens-per-domain "$TOKENS_PER_DOMAIN" \
    --output-dir data/domains/

echo "=== Data preparation complete ==="
```

Also implement `scripts/tokenize_dclm_domains.py` which:
1. Checks whether `data/domains/{topic}/train.bin` already exists and has
   sufficient tokens — skip if so
2. If not, downloads the relevant partition of DCLM from HuggingFace
   (`mlfoundations/dclm-baseline-1.0`) using streaming to avoid downloading
   the entire dataset
3. Filters documents to the requested topic using WebOrganizer labels if
   available in the DCLM dataset, or by downloading WebOrganizer and classifying
4. Tokenizes with the Dolma 2 tokenizer
5. Saves as numpy uint16 memmap: `data/domains/{topic}/train.bin`
6. Saves token count to `data/domains/{topic}/token_count.txt`

---

### Script 04: generate_run_configs.py

Generate one OLMo YAML config per training run.

Before writing this script, Claude Code must:
1. Read `OLMo/configs/` to find an existing small model config
2. Read `OLMo/olmo/config.py` to understand all config fields
3. Understand how OLMo specifies per-domain data mixing weights

```python
#!/usr/bin/env python3
"""
Generate one OLMo training config per run.
Runs:
  - 1M proxy × 20 mixtures = 20 configs
  - 15M proxy × 20 mixtures = 20 configs
  - 60M target × 20 mixtures = 20 configs
  - 150M target × 8 mixtures = 8 configs
Total: 68 configs
"""
import json, yaml
from pathlib import Path

CHINCHILLA_TOKENS = {
    "proxy_1m":    40_000_000,    # 2x Chinchilla × 1M params × 20 tokens/param
    "proxy_15m":   600_000_000,   # 2x Chinchilla × 15M params × 20 tokens/param
    "target_60m":  1_200_000_000, # 1x Chinchilla × 60M params × 20 tokens/param
    "target_150m": 3_000_000_000, # 1x Chinchilla × 150M params × 20 tokens/param
}

def load_base_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

def make_data_config(weights: dict, domain_data_dir: str) -> dict:
    """
    Build OLMo data config from domain weights.
    NOTE: Claude Code must read OLMo's data loading code to get the exact
    format. The structure below is a placeholder — adapt to match OLMo's
    actual config schema after reading OLMo/olmo/data/.
    """
    sources = []
    for domain, weight in weights.items():
        if weight < 0.001:
            continue
        domain_name = domain.replace("dclm:", "")
        sources.append({
            "paths": [f"{domain_data_dir}/{domain_name}/train.bin"],
            "weight": float(weight)
        })
    return {"sources": sources}

def main():
    Path("run_configs").mkdir(exist_ok=True)
    shared    = json.load(open("data/shared_mixtures.json"))
    subset_150m = json.load(open("data/shared_mixtures_150m.json"))
    subset_ids  = set(subset_150m["mixture_ids"])

    manifest = []

    run_types = [
        ("proxy_1m",    "configs/proxy_1m_base.yaml",    "proxy",  1),
        ("proxy_15m",   "configs/proxy_15m_base.yaml",   "proxy",  15),
        ("target_60m",  "configs/target_60m_base.yaml",  "target", 60),
        ("target_150m", "configs/target_150m_base.yaml", "target", 150),
    ]

    for run_type, base_config_path, model_class, size_m in run_types:
        base = load_base_config(base_config_path)
        token_budget = CHINCHILLA_TOKENS[run_type]

        # For 150M, only run on the 8-mixture subset
        mixtures_to_use = (
            [m for i, m in enumerate(shared["mixtures"]) if i in subset_ids]
            if run_type == "target_150m"
            else shared["mixtures"]
        )

        for mixture in mixtures_to_use:
            mid = mixture["mixture_id"]
            run_name = f"{run_type}_mix{mid:02d}"

            cfg = base.copy()
            cfg["run_name"]   = run_name
            cfg["max_tokens"] = token_budget
            cfg["save_folder"] = f"checkpoints/{run_name}"
            cfg["data"] = make_data_config(mixture["weights"], "data/domains")
            cfg["wandb"] = {
                "project": "proxy-scaling",
                "name":    run_name,
                "tags":    [model_class, f"{size_m}m", f"mix{mid:02d}"]
            }

            config_path = f"run_configs/{run_name}.yaml"
            with open(config_path, "w") as f:
                yaml.dump(cfg, f, default_flow_style=False)

            manifest.append({
                "run_name":    run_name,
                "run_type":    run_type,
                "model_class": model_class,
                "size_m":      size_m,
                "mixture_id":  mid,
                "token_budget": token_budget,
                "config_path": config_path,
                "status":      "pending"
            })

    json.dump(manifest, open("run_configs/manifest.json", "w"), indent=2)
    print(f"Generated {len(manifest)} configs")

    # Summary
    from collections import Counter
    counts = Counter(r["run_type"] for r in manifest)
    for rt, n in counts.items():
        print(f"  {rt}: {n} runs")

if __name__ == "__main__":
    main()
```

---

### Script 05: smoke_test.sh

End-to-end test. Runs one training run per model type at minimal token budget
(10M tokens), evaluates both, and verifies the analysis pipeline with synthetic
data. Target cost: $5-6.

```bash
#!/bin/bash
set -e
PASS=0; FAIL=0

check() {
    echo -n "  [CHECK] $1 ... "
    if eval "$2" > /tmp/smoke_out 2>&1; then
        echo "PASS"; PASS=$((PASS+1))
    else
        echo "FAIL"; cat /tmp/smoke_out; FAIL=$((FAIL+1))
    fi
}

echo "========================================"
echo "SMOKE TEST"
echo "========================================"

echo "[1/6] Checking prerequisite files..."
check "shared_mixtures.json exists"     "test -f data/shared_mixtures.json"
check "proxy_30m_bpb.csv exists"        "test -f data/proxy_30m_bpb.csv"
check "run_configs/manifest.json exists" "test -f run_configs/manifest.json"
check "OLMES installed"                  "python3 -c 'import lm_eval'"
check "OLMo installed"                   "test -f OLMo/scripts/train.py"

echo "[2/6] Checking domain data..."
check "Domain data present" "python3 -c \"
import json, os
shared = json.load(open('data/shared_mixtures.json'))
domains = [d.replace('dclm:', '') for d in shared['domain_cols'] if d.startswith('dclm:')]
missing = [d for d in domains if not os.path.exists(f'data/domains/{d}/train.bin')]
if missing: print(f'Missing: {missing}'); exit(1)
print(f'All {len(domains)} domain files present')
\""

echo "[3/6] Training smoke test models (10M tokens each)..."
# Generate minimal smoke test configs
python3 - << 'PYEOF'
import yaml, json
for run_type in ["proxy_1m", "proxy_15m", "target_60m"]:
    src = f"run_configs/{run_type}_mix00.yaml"
    dst = f"run_configs/smoke_{run_type}.yaml"
    cfg = yaml.safe_load(open(src))
    cfg["max_tokens"] = 10_000_000
    cfg["run_name"] = f"smoke_{run_type}"
    cfg["save_folder"] = f"checkpoints/smoke_{run_type}"
    cfg.pop("wandb", None)
    yaml.dump(cfg, open(dst, "w"), default_flow_style=False)
    print(f"Generated {dst}")
PYEOF

for run_type in proxy_1m proxy_15m target_60m; do
    check "Train smoke_${run_type}" \
        "python3 OLMo/scripts/train.py run_configs/smoke_${run_type}.yaml"
    check "Checkpoint exists for ${run_type}" \
        "test -d checkpoints/smoke_${run_type}"
done

echo "[4/6] Running evaluation on smoke models..."
for run_type in proxy_1m proxy_15m target_60m; do
    check "Evaluate smoke_${run_type}" "
        olmes \
          --model checkpoints/smoke_${run_type} \
          --task arc_easy::olmes hellaswag::olmes \
          --output-dir results/eval/smoke_${run_type}/ \
          --limit 50
    "
    check "Eval JSON exists for ${run_type}" \
        "test -d results/eval/smoke_${run_type}"
done

echo "[5/6] Testing analysis pipeline with synthetic data..."
check "compute_correlations --synthetic" \
    "python3 scripts/09_compute_correlations.py --synthetic"
check "fit_scaling_law --synthetic" \
    "python3 scripts/10_fit_scaling_law.py --synthetic"
check "make_figures --synthetic" \
    "python3 scripts/11_make_figures.py --synthetic"

echo "[6/6] Checking BPB values are in valid range..."
check "BPB values are finite and in range" "python3 - << 'PYEOF'
import json, os, glob
for f in glob.glob('results/eval/smoke_*/results*.json'):
    data = json.load(open(f))
    for task, res in data.get('results', {}).items():
        for k, v in res.items():
            if isinstance(v, float) and ('bpb' in k or 'bits' in k):
                assert 0.3 < v < 5.0, f'{f} {task} {k}={v} out of range'
print('All BPB values in valid range 0.3-5.0')
PYEOF
"

echo ""
echo "========================================"
printf "SMOKE TEST: %d passed, %d failed\n" $PASS $FAIL
echo "========================================"
if [ $FAIL -eq 0 ]; then
    echo "ALL PASS — safe to run: bash scripts/06_run_all_training.sh"
else
    echo "FAILURES DETECTED — fix before proceeding"
    exit 1
fi
```

---

### Script 06: run_all_training.sh

Run all 68 training runs sequentially. Safe to rerun — skips completed runs.
Run inside tmux.

```bash
#!/bin/bash
echo "Starting all training runs. Run inside tmux:"
echo "  tmux new -s training && bash scripts/06_run_all_training.sh"
echo ""

python3 - << 'PYEOF'
import json, subprocess, time
from pathlib import Path

manifest = json.load(open("run_configs/manifest.json"))
log_path  = Path("results/training_log.json")
log       = json.loads(log_path.read_text()) if log_path.exists() else []
completed = {r["run_name"] for r in log if r["status"] == "completed"}

Path("results").mkdir(exist_ok=True)

# Sort: proxies first (cheaper, fail-fast), then targets
order = {"proxy_1m": 0, "proxy_15m": 1, "target_60m": 2, "target_150m": 3}
manifest = sorted(manifest, key=lambda r: (order.get(r["run_type"], 9), r["mixture_id"]))

total = len(manifest)
for i, run in enumerate(manifest):
    name = run["run_name"]
    if name in completed:
        print(f"[{i+1}/{total}] SKIP (done): {name}")
        continue
    ckpt = Path(f"checkpoints/{name}")
    if ckpt.exists() and any(ckpt.iterdir()):
        print(f"[{i+1}/{total}] SKIP (checkpoint): {name}")
        completed.add(name)
        continue

    print(f"[{i+1}/{total}] START: {name} ({run['run_type']}, "
          f"mix{run['mixture_id']:02d}, {run['token_budget']//1e9:.1f}B tokens)")
    t0 = time.time()
    try:
        subprocess.run(
            ["python3", "OLMo/scripts/train.py", run["config_path"]],
            check=True
        )
        status = "completed"
    except subprocess.CalledProcessError:
        status = "failed"
    elapsed = (time.time() - t0) / 60

    log.append({
        "run_name":   name,
        "run_type":   run["run_type"],
        "mixture_id": run["mixture_id"],
        "status":     status,
        "elapsed_min": round(elapsed, 1)
    })
    log_path.write_text(json.dumps(log, indent=2))
    print(f"  -> {status} in {elapsed:.1f} min")

done    = sum(1 for r in log if r["status"] == "completed")
failed  = sum(1 for r in log if r["status"] == "failed")
print(f"\nDone. Completed: {done}, Failed: {failed}")
if failed:
    print("Failed runs:")
    for r in log:
        if r["status"] == "failed":
            print(f"  {r['run_name']}")
PYEOF
```

---

### Script 07: run_evaluation.sh

Evaluate all completed checkpoints using OLMES.

```bash
#!/bin/bash
set -e

mkdir -p results/eval

TASKS="arc_challenge::olmes arc_easy::olmes hellaswag::olmes winogrande::olmes \
       piqa::olmes csqa::olmes minerva_math_algebra::olmes mmlu::olmes"

python3 - << 'PYEOF'
import json, subprocess
from pathlib import Path

manifest = json.load(open("run_configs/manifest.json"))

for run in manifest:
    name    = run["run_name"]
    ckpt    = Path(f"checkpoints/{name}")
    out_dir = Path(f"results/eval/{name}")

    if out_dir.exists() and any(out_dir.iterdir()):
        print(f"SKIP (evaluated): {name}")
        continue
    if not ckpt.exists():
        print(f"SKIP (no checkpoint): {name}")
        continue

    # Find latest checkpoint step
    steps = sorted(ckpt.glob("step*"))
    if not steps:
        print(f"SKIP (no step dirs): {name}")
        continue
    latest = steps[-1]

    print(f"Evaluating: {name} from {latest}")
    tasks = ("arc_challenge::olmes arc_easy::olmes hellaswag::olmes "
             "winogrande::olmes piqa::olmes csqa::olmes "
             "minerva_math_algebra::olmes mmlu::olmes")
    cmd = (f"olmes --model {latest} --task {tasks} "
           f"--output-dir results/eval/{name}/")
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"  -> done")
    except subprocess.CalledProcessError as e:
        print(f"  -> FAILED: {e}")
PYEOF
```

---

### Script 08: collect_results.py

Parse OLMES outputs and compile into a clean CSV.

```python
#!/usr/bin/env python3
"""
Parse OLMES eval outputs and compile results CSV.
Output: results/all_results.csv
"""
import json, glob
import pandas as pd
import numpy as np
from pathlib import Path

def parse_olmes_output(eval_dir: str) -> dict | None:
    """Parse OLMES output directory for BPB scores."""
    result_files = (list(Path(eval_dir).glob("*.json")) +
                    list(Path(eval_dir).glob("**/*.json")))
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
                    elif isinstance(v, float) and "acc" in k and "stderr" not in k:
                        per_task[task] = v  # fallback to accuracy
            if per_task:
                return {
                    "avg_bpb": float(np.mean(list(per_task.values()))),
                    "per_task": per_task
                }
        except Exception:
            continue
    return None

def main():
    Path("results").mkdir(exist_ok=True)
    manifest = json.load(open("run_configs/manifest.json"))
    shared   = json.load(open("data/shared_mixtures.json"))
    proxy30  = pd.read_csv("data/proxy_30m_bpb.csv")

    rows = []

    # Released 30M proxy data
    for _, row in proxy30.iterrows():
        rows.append({
            "run_name":   f"proxy_30m_mix{int(row['mixture_id']):02d}",
            "run_type":   "proxy_30m",
            "model_class": "proxy",
            "size_m":     30,
            "mixture_id": int(row["mixture_id"]),
            "avg_bpb":    row["avg_bpb"],
            "source":     "released"
        })

    # Newly trained models
    for run in manifest:
        name    = run["run_name"]
        eval_dir = f"results/eval/{name}"
        result  = parse_olmes_output(eval_dir)
        if result is None:
            print(f"WARNING: no eval result for {name}")
            continue
        rows.append({
            "run_name":    name,
            "run_type":    run["run_type"],
            "model_class": run["model_class"],
            "size_m":      run["size_m"],
            "mixture_id":  run["mixture_id"],
            "avg_bpb":     result["avg_bpb"],
            "source":      "trained"
        })

    df = pd.DataFrame(rows)
    df.to_csv("results/all_results.csv", index=False)
    print(f"Compiled {len(df)} results to results/all_results.csv")

    # Summary
    for run_type in df["run_type"].unique():
        sub = df[df["run_type"] == run_type]
        print(f"  {run_type}: {len(sub)} rows, "
              f"avg_bpb range [{sub['avg_bpb'].min():.3f}, {sub['avg_bpb'].max():.3f}]")

if __name__ == "__main__":
    main()
```

---

### Script 09: compute_correlations.py

Compute the ρ(proxy_size, target_size) matrix.

```python
#!/usr/bin/env python3
"""
Compute Spearman rank correlation between proxy and target BPB across mixtures.

Result: 3×2 matrix (3 proxy sizes × 2 new target sizes)
Plus the paper's values for T=1B (3rd column, free).

Output: results/correlation_matrix.json
"""
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr
import json, argparse
from pathlib import Path

# Paper's reported values (Figure 3, Olmix paper) for proxy vs 1B target
PAPER_1B_VALUES = {1: 0.73, 15: 0.89, 30: 0.896}

def bootstrap_ci(x, y, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    rs = [spearmanr(x[rng.integers(0, len(x), len(x))],
                    y[rng.integers(0, len(y), len(y))]).statistic
          for _ in range(n)]
    return float(np.percentile(rs, 2.5)), float(np.percentile(rs, 97.5))

def make_synthetic_data():
    """Synthetic data for smoke test mode."""
    rng = np.random.default_rng(42)
    n = 20
    true_perf = rng.normal(0, 1, n)
    results = []
    # Simulate: smaller proxies have lower correlation with smaller targets
    for size_m, target_m, base_corr in [
        (1,  60,  0.50), (15,  60, 0.82), (30,  60, 0.88),
        (1, 150,  0.60), (15, 150, 0.85), (30, 150, 0.90),
    ]:
        noise = rng.normal(0, 0.8 * (1 - base_corr), n)
        bpb = true_perf + noise
        for i in range(n):
            results.append({
                "run_type": f"proxy_{size_m}m" if size_m < 60 else f"target_{size_m}m",
                "model_class": "proxy" if size_m < 60 else "target",
                "size_m": size_m if size_m < 60 else target_m,
                "mixture_id": i,
                "avg_bpb": float(bpb[i])
            })
    # Add target rows
    for target_m in [60, 150]:
        noise = rng.normal(0, 0.1, n)
        bpb = true_perf + noise
        for i in range(n):
            results.append({
                "run_type": f"target_{target_m}m",
                "model_class": "target",
                "size_m": target_m,
                "mixture_id": i,
                "avg_bpb": float(bpb[i])
            })
    return pd.DataFrame(results)

def compute_matrix(df):
    proxy_sizes  = [1, 15, 30]
    target_sizes = [60, 150]
    matrix = {}

    for p in proxy_sizes:
        proxy_type = f"proxy_{p}m" if p != 30 else "proxy_30m"
        proxy_df = (df[df["run_type"] == proxy_type]
                    .sort_values("mixture_id")
                    .set_index("mixture_id"))

        for t in target_sizes:
            target_df = (df[df["run_type"] == f"target_{t}m"]
                         .sort_values("mixture_id")
                         .set_index("mixture_id"))

            shared_ids = sorted(set(proxy_df.index) & set(target_df.index))
            if len(shared_ids) < 5:
                print(f"  WARNING: only {len(shared_ids)} shared mixtures for "
                      f"P={p}M, T={t}M — skipping")
                continue

            px = proxy_df.loc[shared_ids, "avg_bpb"].values
            tx = target_df.loc[shared_ids, "avg_bpb"].values

            sp_r = float(spearmanr(px, tx).statistic)
            pe_r = float(pearsonr(px, tx).statistic)
            ci_lo, ci_hi = bootstrap_ci(px, tx)

            key = f"P{p}M_T{t}M"
            matrix[key] = {
                "proxy_size_m":  p,
                "target_size_m": t,
                "n_mixtures":    len(shared_ids),
                "spearman_r":    round(sp_r, 4),
                "pearson_r":     round(pe_r, 4),
                "ci_95":         [round(ci_lo, 4), round(ci_hi, 4)]
            }
            print(f"  P={p:2d}M vs T={t}M: Spearman={sp_r:.3f} "
                  f"[{ci_lo:.3f},{ci_hi:.3f}] Pearson={pe_r:.3f} (n={len(shared_ids)})")

    # Add paper's 1B column
    for p in proxy_sizes:
        if p in PAPER_1B_VALUES:
            matrix[f"P{p}M_T1000M_paper"] = {
                "proxy_size_m":  p,
                "target_size_m": 1000,
                "source":        "Olmix paper Figure 3",
                "spearman_r":    PAPER_1B_VALUES[p]
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
    print("\n" + "="*60)
    print(f"{'':15} {'T=60M':>12} {'T=150M':>12} {'T=1B (paper)':>14}")
    print("-"*60)
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
    print("="*60)

if __name__ == "__main__":
    main()
```

---

### Script 10: fit_scaling_law.py

Fit candidate scaling laws to the correlation matrix and assess which best
describes how ρ(P, T) depends on the proxy-to-target ratio P/T.

```python
#!/usr/bin/env python3
"""
Fit scaling laws to the correlation matrix.

Candidate forms:
  1. Constant:    rho = 1 - exp(-alpha * P^beta)  [threshold is absolute]
  2. Ratio:       rho = 1 - exp(-alpha * (P/T)^beta)  [threshold scales with T]
  3. Log-ratio:   rho = sigmoid(alpha * log(P/T) + beta)
  4. Power thresh: P*(T) = alpha * T^beta  [threshold follows power law]

With the paper's 1B column included as a free data point, we have a 3x3 grid
(3 proxy sizes × 3 target sizes) = 9 observations for fitting.

Output: results/scaling_law_fits.json
"""
import json
import numpy as np
from scipy.optimize import curve_fit
from scipy.special import expit  # sigmoid
import argparse

def load_data(matrix: dict) -> tuple:
    P_vals, T_vals, rho_vals = [], [], []
    for key, val in matrix.items():
        if "spearman_r" in val and "proxy_size_m" in val:
            P_vals.append(val["proxy_size_m"])
            T_vals.append(val["target_size_m"])
            rho_vals.append(val["spearman_r"])
    return np.array(P_vals), np.array(T_vals), np.array(rho_vals)

def fit_model(model_fn, P, T, rho, p0, bounds):
    try:
        popt, pcov = curve_fit(model_fn, (P, T), rho, p0=p0,
                               bounds=bounds, maxfev=10000)
        pred  = model_fn((P, T), *popt)
        ss_res = np.sum((rho - pred) ** 2)
        ss_tot = np.sum((rho - rho.mean()) ** 2)
        r2    = 1 - ss_res / ss_tot if ss_tot > 0 else 0
        perr  = np.sqrt(np.diag(pcov))
        return popt, perr, float(r2), float(np.sqrt(ss_res / len(rho)))
    except Exception as e:
        return None, None, None, str(e)

def loo_rmse(model_fn, P, T, rho, p0, bounds):
    """Leave-one-out RMSE."""
    errors = []
    for i in range(len(rho)):
        mask = np.arange(len(rho)) != i
        try:
            popt, _, _, _ = fit_model(model_fn, P[mask], T[mask], rho[mask],
                                      p0, bounds)
            if popt is not None:
                pred_i = model_fn((P[i:i+1], T[i:i+1]), *popt)[0]
                errors.append((rho[i] - pred_i) ** 2)
        except Exception:
            pass
    return float(np.sqrt(np.mean(errors))) if errors else float("nan")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    if args.synthetic:
        # Fake correlation matrix
        matrix = {
            f"P{p}M_T{t}M": {"proxy_size_m": p, "target_size_m": t,
                              "spearman_r": min(0.95, 0.5 + 0.3*(p/t)**0.5)}
            for p in [1, 15, 30] for t in [60, 150, 1000]
        }
    else:
        matrix = json.load(open("results/correlation_matrix.json"))

    P, T, rho = load_data(matrix)
    print(f"Fitting scaling laws to {len(P)} data points")
    print(f"P values: {sorted(set(P))} M params")
    print(f"T values: {sorted(set(T))} M params")
    print(f"rho range: [{rho.min():.3f}, {rho.max():.3f}]")

    results = {}

    # Model 1: Constant threshold (P only, no T dependence)
    def model_constant(PT, alpha, beta):
        P, _ = PT
        return 1 - np.exp(-alpha * (P ** beta))
    popt, perr, r2, rmse = fit_model(
        model_constant, P, T, rho, [0.1, 1.0], ([0,0],[np.inf,np.inf]))
    loo = loo_rmse(model_constant, P, T, rho, [0.1, 1.0], ([0,0],[np.inf,np.inf]))
    results["constant"] = {
        "formula": "rho = 1 - exp(-alpha * P^beta)",
        "interpretation": "threshold is absolute (independent of T)",
        "params": {"alpha": float(popt[0]), "beta": float(popt[1])} if popt is not None else None,
        "r2": r2, "rmse": rmse, "loo_rmse": loo
    }

    # Model 2: Ratio model (P/T dependence)
    def model_ratio(PT, alpha, beta):
        P, T = PT
        return 1 - np.exp(-alpha * (P / T) ** beta)
    popt, perr, r2, rmse = fit_model(
        model_ratio, P, T, rho, [1.0, 0.5], ([0,0],[np.inf,np.inf]))
    loo = loo_rmse(model_ratio, P, T, rho, [1.0, 0.5], ([0,0],[np.inf,np.inf]))
    results["ratio"] = {
        "formula": "rho = 1 - exp(-alpha * (P/T)^beta)",
        "interpretation": "threshold scales linearly with T when beta=1",
        "params": {"alpha": float(popt[0]), "beta": float(popt[1])} if popt is not None else None,
        "r2": r2, "rmse": rmse, "loo_rmse": loo,
        "implication": (f"For T=500M, min proxy = {0.15 * 500 ** (popt[1] if popt is not None else 1):.0f}M"
                        if popt is not None else "fit failed")
    }

    # Model 3: Log-ratio (sigmoid)
    def model_logratio(PT, alpha, beta):
        P, T = PT
        return expit(alpha * np.log(P / T) + beta)
    popt, perr, r2, rmse = fit_model(
        model_logratio, P, T, rho, [2.0, 2.0], ([-np.inf,-np.inf],[np.inf,np.inf]))
    loo = loo_rmse(model_logratio, P, T, rho, [2.0, 2.0], ([-np.inf,-np.inf],[np.inf,np.inf]))
    results["logratio"] = {
        "formula": "rho = sigmoid(alpha * log(P/T) + beta)",
        "interpretation": "S-curve in log(P/T) space",
        "params": {"alpha": float(popt[0]), "beta": float(popt[1])} if popt is not None else None,
        "r2": r2, "rmse": rmse, "loo_rmse": loo
    }

    # Print comparison table
    print("\n" + "="*60)
    print(f"{'Model':15} {'R²':>8} {'RMSE':>8} {'LOO-RMSE':>10}")
    print("-"*60)
    for name, res in results.items():
        r2_str   = f"{res['r2']:.4f}" if isinstance(res.get('r2'), float) else "N/A"
        rmse_str = f"{res['rmse']:.4f}" if isinstance(res.get('rmse'), float) else "N/A"
        loo_str  = f"{res['loo_rmse']:.4f}" if isinstance(res.get('loo_rmse'), float) else "N/A"
        print(f"{name:15} {r2_str:>8} {rmse_str:>8} {loo_str:>10}")
    print("="*60)

    # Best model by LOO RMSE
    valid = {k: v for k, v in results.items() if isinstance(v.get("loo_rmse"), float)}
    if valid:
        best = min(valid, key=lambda k: valid[k]["loo_rmse"])
        print(f"\nBest fit by LOO RMSE: {best}")
        print(f"  {results[best]['formula']}")
        print(f"  {results[best]['interpretation']}")

    json.dump(results, open("results/scaling_law_fits.json", "w"), indent=2)
    print("\nSaved results/scaling_law_fits.json")

if __name__ == "__main__":
    main()
```

---

### Script 11: make_figures.py

Generate three paper-ready figures.

```python
#!/usr/bin/env python3
"""Generate paper-ready figures."""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse
from pathlib import Path
from scipy.special import expit

mpl.rcParams.update({
    "font.family": "serif", "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.dpi": 150
})

PAPER_1B = {1: 0.73, 15: 0.89, 30: 0.896}
PROXY_SIZES  = [1, 15, 30]
TARGET_SIZES = [60, 150]
THRESHOLD    = 0.85
COLORS = {"60m": "#2196F3", "150m": "#FF9800", "1b_paper": "#9C27B0"}

def load_or_synthetic(path, synthetic_fn):
    if Path(path).exists():
        return json.load(open(path))
    return synthetic_fn()

def synth_matrix():
    m = {}
    for p in PROXY_SIZES:
        for t in TARGET_SIZES:
            rho = min(0.96, 0.4 + 0.4*(p/t)**0.4 + 0.05*t/200)
            m[f"P{p}M_T{t}M"] = {
                "proxy_size_m": p, "target_size_m": t,
                "spearman_r": rho, "ci_95": [rho-0.08, rho+0.05]
            }
        m[f"P{p}M_T1000M_paper"] = {
            "proxy_size_m": p, "target_size_m": 1000,
            "spearman_r": PAPER_1B[p]
        }
    return m

def synth_fits():
    return {
        "ratio": {
            "formula": "rho = 1 - exp(-alpha * (P/T)^beta)",
            "params": {"alpha": 2.5, "beta": 0.55},
            "r2": 0.94
        }
    }

def fig1_heatmap(matrix, ax=None):
    """3×3 heatmap: proxy size (rows) × target size (cols)."""
    show = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4.5))

    all_t = TARGET_SIZES + [1000]
    labels_t = [f"{t}M" for t in TARGET_SIZES] + ["1B\n(paper)"]
    data  = np.full((len(PROXY_SIZES), len(all_t)), np.nan)

    for i, p in enumerate(PROXY_SIZES):
        for j, t in enumerate(TARGET_SIZES):
            k = f"P{p}M_T{t}M"
            if k in matrix:
                data[i, j] = matrix[k]["spearman_r"]
        if p in PAPER_1B:
            data[i, 2] = PAPER_1B[p]

    im = ax.imshow(data, vmin=0.4, vmax=1.0, cmap="RdYlGn", aspect="auto")
    plt.colorbar(im, ax=ax, label="Spearman ρ")

    ax.set_xticks(range(len(all_t)))
    ax.set_xticklabels(labels_t)
    ax.set_yticks(range(len(PROXY_SIZES)))
    ax.set_yticklabels([f"{p}M proxy" for p in PROXY_SIZES])
    ax.set_xlabel("Target Model Size")
    ax.set_ylabel("Proxy Model Size")
    ax.set_title("Proxy-Target Rank Correlation Matrix\n"
                 "(right column from Olmix paper)", fontsize=11, fontweight="bold")

    for i in range(len(PROXY_SIZES)):
        for j in range(len(all_t)):
            if not np.isnan(data[i, j]):
                ax.text(j, i, f"{data[i,j]:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if 0.5 < data[i,j] < 0.9 else "white")

    # Threshold indicator
    ax.axhline(0.5, color="red", linewidth=2, linestyle="--", alpha=0.5)
    ax.text(2.1, 0.5, f"ρ={THRESHOLD} threshold", color="red", fontsize=8, va="center")

    if show:
        plt.tight_layout()
        return plt.gcf()

def fig2_threshold(matrix):
    """Does the threshold scale with target size?"""
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    for t, color, label in [(60, COLORS["60m"], "T=60M (ours)"),
                             (150, COLORS["150m"], "T=150M (ours)")]:
        rhos, cis = [], []
        for p in PROXY_SIZES:
            k = f"P{p}M_T{t}M"
            if k in matrix:
                rhos.append(matrix[k]["spearman_r"])
                ci = matrix[k].get("ci_95", [rhos[-1]-0.05, rhos[-1]+0.05])
                cis.append(ci)
            else:
                rhos.append(None); cis.append(None)

        valid = [(p, r, c) for p, r, c in zip(PROXY_SIZES, rhos, cis) if r is not None]
        if valid:
            ps, rs, cs = zip(*valid)
            ax.plot(ps, rs, "o-", color=color, linewidth=2, markersize=8, label=label)
            for p, r, ci in valid:
                ax.errorbar(p, r, yerr=[[r-ci[0]], [ci[1]-r]],
                            fmt="none", color=color, capsize=4)

    # Paper 1B line
    paper_rhos = [PAPER_1B[p] for p in PROXY_SIZES]
    ax.plot(PROXY_SIZES, paper_rhos, "s--", color=COLORS["1b_paper"],
            linewidth=2, markersize=8, label="T=1B (Olmix paper, Figure 3)")

    ax.axhline(THRESHOLD, color="red", linestyle=":", linewidth=1.5, alpha=0.8)
    ax.text(max(PROXY_SIZES)*0.98, THRESHOLD+0.01,
            f"ρ={THRESHOLD}", color="red", ha="right", fontsize=9)

    ax.set_xlabel("Proxy Model Size (M parameters)", fontsize=11)
    ax.set_ylabel("Spearman Rank Correlation (ρ)", fontsize=11)
    ax.set_xscale("log")
    ax.set_ylim(0.4, 1.02)
    ax.set_xticks(PROXY_SIZES)
    ax.set_xticklabels([f"{p}M" for p in PROXY_SIZES])
    ax.set_title("Does the 15M Proxy Threshold Hold for Smaller Targets?",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig

def fig3_scaling_law(matrix, fits):
    """Fitted scaling law surface vs observations."""
    best_fit = fits.get("ratio", {})
    params   = best_fit.get("params", {})

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    # Left: ratio surface
    ax = axes[0]
    if params:
        alpha, beta = params["alpha"], params["beta"]
        p_range = np.linspace(0.5, 35, 100)
        for t, color, label in [(60, COLORS["60m"], "T=60M"),
                                 (150, COLORS["150m"], "T=150M"),
                                 (1000, COLORS["1b_paper"], "T=1B")]:
            pred = 1 - np.exp(-alpha * (p_range / t) ** beta)
            ax.plot(p_range, pred, "-", color=color, label=f"Fit: {label}", alpha=0.7)

    # Overlay observations
    for t, color in [(60, COLORS["60m"]), (150, COLORS["150m"])]:
        for p in PROXY_SIZES:
            k = f"P{p}M_T{t}M"
            if k in matrix:
                ax.scatter(p, matrix[k]["spearman_r"], color=color, s=80, zorder=5)
    for p in PROXY_SIZES:
        ax.scatter(p, PAPER_1B[p], color=COLORS["1b_paper"], s=80, marker="s", zorder=5)

    ax.axhline(THRESHOLD, color="red", linestyle=":", linewidth=1, alpha=0.7)
    ax.set_xscale("log"); ax.set_xlim(0.5, 40); ax.set_ylim(0.4, 1.02)
    ax.set_xlabel("Proxy Size (M params)"); ax.set_ylabel("Spearman ρ")
    ax.set_title(f"Ratio Model Fit\n"
                 f"ρ = 1-exp(-α(P/T)^β), R²={best_fit.get('r2', 'N/A'):.3f}",
                 fontsize=10)
    ax.legend(fontsize=8)

    # Right: P* threshold vs T
    ax = axes[1]
    t_range = np.logspace(np.log10(50), np.log10(1200), 100)
    if params:
        alpha, beta = params["alpha"], params["beta"]
        # P* = T * (-log(1-threshold)/alpha)^(1/beta)
        p_star = t_range * ((-np.log(1 - THRESHOLD)) / alpha) ** (1 / beta)
        ax.plot(t_range, p_star, "-", color="navy",
                label=f"Ratio model (β={beta:.2f})")

    # Observed approximate thresholds
    for t, color, label in [(60, COLORS["60m"], "T=60M"),
                             (150, COLORS["150m"], "T=150M"),
                             (1000, COLORS["1b_paper"], "T=1B (paper)")]:
        rhos = {}
        for p in PROXY_SIZES:
            k = f"P{p}M_T{t}M" if t != 1000 else f"P{p}M_T1000M_paper"
            if k in matrix:
                rhos[p] = matrix[k]["spearman_r"]
            elif t == 1000 and p in PAPER_1B:
                rhos[p] = PAPER_1B[p]
        # Find approximate P* by interpolation
        sorted_p = sorted(rhos.keys())
        for i in range(len(sorted_p)-1):
            if rhos[sorted_p[i]] < THRESHOLD <= rhos[sorted_p[i+1]]:
                p_star_obs = (sorted_p[i] + sorted_p[i+1]) / 2
                ax.scatter(t, p_star_obs, color=color, s=100, zorder=5, label=label)
                break

    ax.axhline(15, color="gray", linestyle="--", alpha=0.5,
               label="15M (paper's recommendation)")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Target Model Size (M params)")
    ax.set_ylabel("Min Reliable Proxy Size P* (M params)")
    ax.set_title(f"Minimum Proxy Size vs Target Size\n(ρ ≥ {THRESHOLD} threshold)",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthetic", action="store_true")
    args = parser.parse_args()

    Path("results/figures").mkdir(parents=True, exist_ok=True)

    matrix = synth_matrix() if args.synthetic else load_or_synthetic(
        "results/correlation_matrix.json", synth_matrix)
    fits   = synth_fits()   if args.synthetic else load_or_synthetic(
        "results/scaling_law_fits.json", synth_fits)

    for name, fig in [("fig1_heatmap",    plt.figure()),
                       ("fig2_threshold",   fig2_threshold(matrix)),
                       ("fig3_scaling_law", fig3_scaling_law(matrix, fits))]:
        if name == "fig1_heatmap":
            fig, ax = plt.subplots(figsize=(6, 4.5))
            fig1_heatmap(matrix, ax)
            plt.tight_layout()

        for ext in ["pdf", "png"]:
            path = f"results/figures/{name}.{ext}"
            fig.savefig(path, bbox_inches="tight", dpi=200 if ext=="png" else None)
        plt.close(fig)
        print(f"Saved results/figures/{name}.pdf/.png")

    if args.synthetic:
        print("\n[SYNTHETIC] Figures generated with fake data. Pipeline verified.")

if __name__ == "__main__":
    main()
```

---

### setup.sh

```bash
#!/bin/bash
set -e
echo "Setting up proxy_scaling environment..."

pip install -r requirements.txt

# Clone OLMo
if [ ! -d "OLMo" ]; then
    git clone https://github.com/allenai/OLMo
    cd OLMo && pip install -e ".[train]" && cd ..
fi

# Clone OLMES
if [ ! -d "olmes" ]; then
    git clone https://github.com/allenai/olmes
    cd olmes && pip install -e . && cd ..
fi

# Download Dolma 2 tokenizer (cache it)
python3 -c "
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('allenai/dolma2-tokenizer', trust_remote_code=True)
print(f'Tokenizer ready: vocab_size={tok.vocab_size}')
"

mkdir -p data results/eval results/figures checkpoints run_configs

echo ""
echo "Setup complete. Run in order:"
echo "  1. python3 scripts/00_explore_olmix_data.py"
echo "  2. python3 scripts/01_download_olmix_data.py"
echo "  3. python3 scripts/02_select_mixtures.py   [commit outputs to git]"
echo "  4. bash scripts/03_prepare_dclm_data.sh"
echo "  5. python3 scripts/04_generate_run_configs.py"
echo "  [rent cloud GPU]"
echo "  6. bash scripts/05_smoke_test.sh"
echo "  7. bash scripts/06_run_all_training.sh      [inside tmux]"
echo "  8. bash scripts/07_run_evaluation.sh"
echo "  9. python3 scripts/08_collect_results.py"
echo " 10. python3 scripts/09_compute_correlations.py"
echo " 11. python3 scripts/10_fit_scaling_law.py"
echo " 12. python3 scripts/11_make_figures.py"
```

### requirements.txt

```
torch>=2.0
transformers>=4.40
huggingface_hub>=0.23
datasets>=2.19
numpy>=1.24
pandas>=2.0
scipy>=1.10
matplotlib>=3.7
seaborn>=0.12
pyyaml>=6.0
wandb>=0.16
```

---

## Cloud GPU Instructions

### Phase 2: Smoke test (~$6)
```bash
git clone <repo> proxy_scaling && cd proxy_scaling
bash setup.sh
bash scripts/03_prepare_dclm_data.sh --smoke-test
bash scripts/05_smoke_test.sh
```
All checks must print PASS before continuing.

### Phase 3: Full experiment (~$160)
```bash
tmux new -s exp
bash scripts/03_prepare_dclm_data.sh
bash scripts/06_run_all_training.sh     # ~40 hrs, leave running
# After training completes:
bash scripts/07_run_evaluation.sh       # ~8 hrs
python3 scripts/08_collect_results.py
python3 scripts/09_compute_correlations.py
python3 scripts/10_fit_scaling_law.py
python3 scripts/11_make_figures.py
```

Monitor:
```bash
python3 -c "
import json
log = json.load(open('results/training_log.json'))
c = sum(1 for r in log if r['status']=='completed')
f = sum(1 for r in log if r['status']=='failed')
print(f'Completed: {c}/68, Failed: {f}')
for r in log[-3:]:
    print(f'  {r[\"run_name\"]}: {r[\"status\"]} ({r[\"elapsed_min\"]} min)')
"
```

Download results:
```bash
rsync -avz <instance-ip>:~/proxy_scaling/results/ ./results/
rsync -avz <instance-ip>:~/proxy_scaling/data/shared_mixtures*.json ./data/
rsync -avz <instance-ip>:~/proxy_scaling/data/proxy_30m_bpb.csv ./data/
```

---

## What the Results Will Tell You

**If ρ(1M, 60M) ≈ ρ(1M, 1B) ≈ 0.73 and ρ(15M, 60M) ≈ 0.89:**
The threshold is roughly constant regardless of target size. The paper's 15M
recommendation generalizes. The ratio model will fit poorly (β ≈ 0).

**If ρ(1M, 60M) > 0.85 (above threshold) but ρ(1M, 1B) = 0.73 (below):**
The threshold is lower for smaller targets — even a 1M proxy suffices for 60M
models. The ratio model will fit well (β > 0). Practitioners training small models
can use much cheaper proxies.

**If ρ(15M, 60M) < 0.85 (below threshold) but ρ(15M, 1B) = 0.89 (above):**
The threshold is higher for smaller targets — more proxy capacity is needed to
predict small model rankings. Unexpected but interesting.

Either way the result is publishable. The paper provides only one column of this
matrix; any new column is a contribution.

---

## Limitations to Acknowledge

1. **Proxy token budget**: we use 2x Chinchilla for 1M/15M proxies vs the paper's
   5x Chinchilla for 30M proxies. This may depress correlation values for our proxies.
   Acknowledge and note that the relative pattern (does correlation jump at 15M?)
   should still be detectable.

2. **Target token budget**: 1x Chinchilla targets may be noisier than the paper's
   targets. This increases variance in our correlation estimates.

3. **8 mixtures for 150M**: the 150M correlation estimates are based on 8 paired
   runs, giving wide confidence intervals (~±0.15-0.20). Report CIs prominently.

4. **Single architecture**: OLMo 2 only. Thresholds may differ for other architectures.

5. **Paper's 1B column**: taken from reported values, not recomputed. Slight
   methodological inconsistency since the paper uses 5x Chinchilla targets and
   a different proxy training budget than ours.