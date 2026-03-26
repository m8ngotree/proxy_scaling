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
missing = [d for d in domains if not os.path.exists(f'data/domains/{d}/train.npy')]
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
    cfg["max_duration"] = "10M tokens"  # Override to minimal budget
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
    echo "ALL PASS - safe to run: bash scripts/06_run_all_training.sh"
else
    echo "FAILURES DETECTED - fix before proceeding"
    exit 1
fi
