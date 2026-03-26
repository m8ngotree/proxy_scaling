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
    print(f"Found {len(bin_files)} .bin files - may be pre-tokenized data")
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

# Main data preparation - download DCLM and tokenize if needed
python3 scripts/tokenize_dclm_domains.py \
    --tokens-per-domain "$TOKENS_PER_DOMAIN" \
    --output-dir data/domains/

echo "=== Data preparation complete ==="
