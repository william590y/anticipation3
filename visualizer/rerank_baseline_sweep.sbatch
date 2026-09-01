#!/bin/bash
#SBATCH --job-name=rerank-base
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-19%4
#SBATCH --output=logs/rerank_base_%A_%a.out
#SBATCH --error=logs/rerank_base_%A_%a.out

# Baseline alpha/beta sweep: beam search (width 5) maximizing
#   alpha * logp_FT + (1-alpha) * logp_base,  alpha in {0.1,0.3,0.5,0.7,0.9}
# on all 24 viz windows, RAW (unfiltered) performance input, under the
# lowest-val-loss non-LoRA FT checkpoint (run_paper_split_v2/checkpoint-2500).
# 5 settings x 4 window-shards = 20 tasks, %4 = the 4 allotted ellis GPUs.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1

ALPHAS=(0.1 0.3 0.5 0.7 0.9)
S=$((SLURM_ARRAY_TASK_ID / 4))
K=$((SLURM_ARRAY_TASK_ID % 4))
A="${ALPHAS[$S]}"
B=$(python -c "print(round(1.0 - $A, 1))")
KEYS=$(python -c "
import json
raw = open('visualizer/data_slim.js').read()
d = json.loads(raw[raw.index('=')+1:].strip().rstrip(';'))
print(','.join(d['example_order'][$K::4]))
")
AA=$(python -c "print(int($A*100))")
echo "setting alpha=$A beta=$B  shard $K  keys: $KEYS"
mkdir -p visualizer/rerank_shards
python visualizer/rerank_viz_rollout.py \
  --checkpoint run_paper_split_v2/checkpoint-2500 \
  --variant raw --num-beams 5 \
  --alpha "$A" --beta "$B" --weights none \
  --example-keys "$KEYS" \
  --output "visualizer/rerank_shards/base_a$(printf '%02d' "$AA")_s${K}.json"
