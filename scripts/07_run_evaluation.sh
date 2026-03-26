#!/bin/bash
set -e

mkdir -p results/eval

python3 - << 'PYEOF'
import json, subprocess
from pathlib import Path

manifest = json.load(open("run_configs/manifest.json"))

for run in manifest:
    name = run["run_name"]
    ckpt = Path(f"checkpoints/{name}")
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
        # Maybe the checkpoint is the directory itself (no step subdirs)
        latest = str(ckpt)
    else:
        latest = str(steps[-1])

    print(f"Evaluating: {name} from {latest}")
    tasks = (
        "arc_challenge::olmes arc_easy::olmes hellaswag::olmes "
        "winogrande::olmes piqa::olmes csqa::olmes "
        "minerva_math_algebra::olmes mmlu::olmes"
    )
    cmd = (
        f"olmes --model {latest} --task {tasks} "
        f"--output-dir results/eval/{name}/"
    )
    try:
        subprocess.run(cmd, shell=True, check=True)
        print(f"  -> done")
    except subprocess.CalledProcessError as e:
        print(f"  -> FAILED: {e}")
PYEOF
