#!/bin/bash
#SBATCH --job-name=viz-beams
#SBATCH --partition=thickstun
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/viz_beams_%j.out
#SBATCH --error=logs/viz_beams_%j.out
#
# Beam-search precompute on thickstun ONLY (never gen-ai-large).
# Default config (user request 2026-08-11): beam=5 only, raw unseeded only.
#
#   for i in $(seq 0 6); do
#     sbatch --export=ALL,SHARD_INDEX=$i,NUM_SHARDS=7 visualizer/precompute_beams.sbatch
#   done
set -euo pipefail

cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

HOST="$(hostname -s 2>/dev/null || hostname)"
case "$HOST" in
  genai-large*|gen-ai-large*)
    echo "REFUSING to run on banned host $HOST" >&2
    exit 1
    ;;
esac

# Hard-pin partition intent (belt and suspenders vs sbatch overrides).
if [[ "${SLURM_JOB_PARTITION:-}" != "thickstun" ]]; then
  echo "REFUSING partition=${SLURM_JOB_PARTITION:-unset}; expected thickstun" >&2
  exit 1
fi

SHARD_INDEX="${SHARD_INDEX:-0}"
NUM_SHARDS="${NUM_SHARDS:-7}"
DATA="${DATA:-visualizer/data.js}"
OUT_DIR="${OUT_DIR:-visualizer/beam_shards}"
BEAM_MIN="${BEAM_MIN:-5}"
BEAM_MAX="${BEAM_MAX:-5}"
VARIANTS="${VARIANTS:-raw}"

mkdir -p logs "$OUT_DIR"
OUT="${OUT_DIR}/shard_$(printf '%02d' "$SHARD_INDEX").json"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

echo "Host: $HOST"
echo "Partition: ${SLURM_JOB_PARTITION}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
echo "Shard ${SHARD_INDEX}/${NUM_SHARDS} -> ${OUT}"
echo "Beams ${BEAM_MIN}-${BEAM_MAX} variants=${VARIANTS}"
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
keys = [eid for i, eid in enumerate(order) if i % num_shards == shard_index]
print(",".join(keys))
PY
)"

echo "Example keys: $KEYS"
if [[ -z "$KEYS" ]]; then
  echo "No example keys for this shard — writing empty shard."
  echo '{"examples":{}}' > "$OUT"
  exit 0
fi

python visualizer/precompute_beams.py \
    --data "$DATA" \
    --output "$OUT" \
    --example-keys "$KEYS" \
    --beam-min "$BEAM_MIN" \
    --beam-max "$BEAM_MAX" \
    --variants "$VARIANTS"

echo "End: $(date)"
