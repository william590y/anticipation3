#!/bin/bash
#SBATCH --job-name=viz-vl-beam
#SBATCH --partition=thickstun
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:1
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/viz_valloss_beams_%A_%a.out
#SBATCH --error=logs/viz_valloss_beams_%A_%a.out

set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

NUM_SHARDS=7
SHARD_INDEX="${SLURM_ARRAY_TASK_ID:?}"
OUT="visualizer/valloss_beam_shards/shard_$(printf '%02d' "$SHARD_INDEX").json"

# Slice example_order for this shard
KEYS=$(python - <<PY
import json
from pathlib import Path
text = Path('visualizer/data.js').read_text()
data = json.loads(text[len('window.VISUALIZER_DATA = '):].rstrip().rstrip(';'))
order = data['example_order']
keys = order[${SHARD_INDEX}::${NUM_SHARDS}]
print(','.join(keys))
PY
)

mkdir -p logs visualizer/valloss_beam_shards
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname) shard=$SHARD_INDEX keys=$KEYS"
echo "Start: $(date)"

python visualizer/precompute_beams.py \
  --data visualizer/data.js \
  --output "$OUT" \
  --example-keys "$KEYS" \
  --groups rollouts_valloss,rollouts_lora_valloss \
  --modes token \
  --beam-min 1 \
  --beam-max 10 \
  --variants filtered,filtered_seeded,raw,raw_seeded

echo "End: $(date)"
