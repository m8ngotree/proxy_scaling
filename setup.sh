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
