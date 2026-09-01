#!/bin/bash
#SBATCH --job-name=merge-bayes-lora
#SBATCH --partition=thickstun
#SBATCH -A thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=logs/merge_bayes_lora_%j.out
#SBATCH --error=logs/merge_bayes_lora_%j.out

set -euo pipefail

# Chained after train_asap_bayes_lora.sbatch: PEFT saves only the adapter, but
# the eval chain (scripts/fullsong_ours.slurm -> full_song_rollout.py ->
# evaluate_muster_asap.load_model) needs a normal HF directory. merge_lora.py
# loads the base + adapter, merges in eval() mode -- i.e. with the POSTERIOR
# MEAN mu, since lora_B.weight IS mu -- and writes a plain checkpoint.
# CPU-only; no GPU is needed to merge.

cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1

ADAPTER="${ADAPTER:-run_bayes_lora_r512/final}"
BASE="${BASE:-run_paper_split_v2/checkpoint-2500}"
OUT="${MERGED:-run_bayes_lora_r512/merged}"

echo "Host: $(hostname)"
echo "Start: $(date)"
# Prefer the requested adapter, but fall back to the newest checkpoint-N if the
# run ended without writing final/ (e.g. wall-clock kill).
if [ ! -f "$ADAPTER/adapter_config.json" ]; then
    ADAPTER=$(ls -d run_bayes_lora_r512/checkpoint-* 2>/dev/null \
        | sort -t- -k2 -n | tail -1)
    echo "final/ missing; falling back to $ADAPTER"
fi
[ -f "$ADAPTER/adapter_config.json" ] || { echo "no adapter found"; exit 1; }

python merge_lora.py --base "$BASE" --adapter "$ADAPTER" --output "$OUT"
echo "End: $(date)"
