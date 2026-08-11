#!/bin/bash
#SBATCH --job-name=viz-bmfull
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/viz_bmfull_%j.out
#SBATCH --error=logs/viz_bmfull_%j.out

# FULL beam precompute: beams 1..10 × filtered/raw × plain/seeded × base+LoRA.
# Job name viz-bmfull avoids collisions with partial viz-beams experiments.
set -euo pipefail

cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

DATA="${DATA:-visualizer/data.js}"
SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-8}"
OUT_DIR="${OUT_DIR:-visualizer/beam_shards}"
OUT="${OUT_DIR}/full_shard_$(printf '%02d' "$SHARD_INDEX").json"

mkdir -p logs "$OUT_DIR"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "FULL beams 1-10 all variants  Shard ${SHARD_INDEX}/${NUM_SHARDS} -> ${OUT}"
echo "Start: $(date)"
nvidia-smi -L || true

KEYS="$(python - "$DATA" "$SHARD_INDEX" "$NUM_SHARDS" <<'PY'
import json, sys
from pathlib import Path
data_path, shard_index, num_shards = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
text = Path(data_path).read_text()
body = text[len("window.VISUALIZER_DATA = "):].rstrip().rstrip(";")
data = json.loads(body)
order = data.get("example_order") or list(data["examples"])
n = len(order)
start = (n * shard_index) // num_shards
end = (n * (shard_index + 1)) // num_shards
print(",".join(order[start:end]))
PY
)"

echo "Example keys: $KEYS"
[[ -n "$KEYS" ]] || { echo '{"examples":{}}' > "$OUT"; exit 0; }

python visualizer/precompute_beams.py \
    --data "$DATA" \
    --output "$OUT" \
    --example-keys "$KEYS" \
    --beam-min 1 \
    --beam-max 10 \
    --modes token \
    --variants filtered,filtered_seeded,raw,raw_seeded

echo "End: $(date)"
