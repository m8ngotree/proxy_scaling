#!/bin/bash
echo "Starting all training runs. Run inside tmux:"
echo "  tmux new -s training && bash scripts/06_run_all_training.sh"
echo ""

python3 - << 'PYEOF'
import json, subprocess, time
from pathlib import Path

manifest = json.load(open("run_configs/manifest.json"))
log_path = Path("results/training_log.json")
log = json.loads(log_path.read_text()) if log_path.exists() else []
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
          f"mix{run['mixture_id']:02d}, {run['token_budget']/1e9:.1f}B tokens)")
    t0 = time.time()
    try:
        subprocess.run(
            ["python3", "OLMo/scripts/train.py", run["config_path"]],
            check=True,
        )
        status = "completed"
    except subprocess.CalledProcessError:
        status = "failed"
    elapsed = (time.time() - t0) / 60

    log.append({
        "run_name": name,
        "run_type": run["run_type"],
        "mixture_id": run["mixture_id"],
        "status": status,
        "elapsed_min": round(elapsed, 1),
    })
    log_path.write_text(json.dumps(log, indent=2))
    print(f"  -> {status} in {elapsed:.1f} min")

done = sum(1 for r in log if r["status"] == "completed")
failed = sum(1 for r in log if r["status"] == "failed")
print(f"\nDone. Completed: {done}, Failed: {failed}")
if failed:
    print("Failed runs:")
    for r in log:
        if r["status"] == "failed":
            print(f"  {r['run_name']}")
PYEOF
