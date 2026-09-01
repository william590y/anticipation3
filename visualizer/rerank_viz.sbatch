#!/bin/bash
#SBATCH --job-name=rerank-viz
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --array=0-7%4
#SBATCH --output=logs/rerank_viz_%A_%a.out
#SBATCH --error=logs/rerank_viz_%A_%a.out

# Beam-search rerank decode (alpha*logp_FT + beta*logp_base + gamma*q_phi) on
# the 24 viz windows, 8 shards capped at 2 concurrent GPUs (%2 = the two ellis
# GPUs allotted to the reranking task). Requires nbest_data/decode_weights.json
# from nbest/fit_weights.py. Merge afterwards:
#   python visualizer/merge_rerank_rollouts.py --shards visualizer/rerank_shards/shard_*.json
#   python visualizer/compute_f1.py --data visualizer/data.js
#   python visualizer/compute_sequence_ppl.py --data visualizer/data.js --checkpoint run_paper_split_v2/checkpoint-7500
#   python visualizer/split_visualizer_payload.py
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1

WEIGHTS="${WEIGHTS:-nbest_data/decode_weights.json}"
NUM_BEAMS="${NUM_BEAMS:-8}"
CKPT="${CKPT:-run_paper_split_v2/checkpoint-2500}"
VARIANT="${VARIANT:-raw}"
test -f "$WEIGHTS" || { echo "missing $WEIGHTS"; exit 1; }

KEYS=$(python -c "
import json, sys
raw = open('visualizer/data_slim.js').read()
d = json.loads(raw[raw.index('=')+1:].strip().rstrip(';'))
order = d['example_order']
print(','.join(order[int('$SLURM_ARRAY_TASK_ID')::8]))
")
echo "shard $SLURM_ARRAY_TASK_ID keys: $KEYS"
mkdir -p visualizer/rerank_shards
python visualizer/rerank_viz_rollout.py \
  --example-keys "$KEYS" --num-beams "$NUM_BEAMS" --weights "$WEIGHTS" \
  --checkpoint "$CKPT" --variant "$VARIANT" \
  --output "visualizer/rerank_shards/shard_$(printf '%02d' "$SLURM_ARRAY_TASK_ID").json"
