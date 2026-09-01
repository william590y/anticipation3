#!/bin/bash
#SBATCH --job-name=viz-masked
#SBATCH --partition=ellis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# Two independent shard workers (no NCCL). 24 GB 3090s are enough for one
# music-medium AR rollout, so do not pin A6000s.
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/viz_masked_%j.out
#SBATCH --error=logs/viz_masked_%j.out

# Attach run_paper_split_v2_masked AR rollouts to visualizer/data.js.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

CHECKPOINT="${CHECKPOINT:-run_paper_split_v2_masked/checkpoint-20000}"
DATA="${DATA:-visualizer/data.js}"
OUT_DIR="${OUT_DIR:-visualizer/masked_shards}"
NUM_SHARDS=2

mkdir -p logs "$OUT_DIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Checkpoint: $CHECKPOINT"
echo "Start: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

NGPU=$(nvidia-smi -L | wc -l)
if [[ "$NGPU" -lt 2 ]]; then
  echo "Need 2 GPUs, got ${NGPU}" >&2
  exit 1
fi

# Map SLURM's (possibly non-contiguous) visible devices onto shard 0/1.
mapfile -t DEVS < <(python - <<'PY'
import os
raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
ids = [x.strip() for x in raw.split(",") if x.strip()]
print("\n".join(ids[:2] if ids else ["0", "1"]))
PY
)

fail=0
pids=()
for shard in 0 1; do
  out="${OUT_DIR}/shard_$(printf '%02d' "$shard").json"
  echo "Launch shard ${shard} on GPU ${DEVS[$shard]} -> ${out}"
  CUDA_VISIBLE_DEVICES="${DEVS[$shard]}" python visualizer/precompute_masked_rollouts.py \
      --data "$DATA" \
      --checkpoint "$CHECKPOINT" \
      --shard-index "$shard" \
      --num-shards "$NUM_SHARDS" \
      --output "$out" &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "shard pid $pid failed" >&2
    fail=1
  fi
done
if [[ "$fail" -ne 0 ]]; then
  exit 1
fi

echo "== merge =="
python visualizer/merge_masked_rollouts.py --data "$DATA" \
    --shards "${OUT_DIR}/shard_00.json" "${OUT_DIR}/shard_01.json"

echo "== F1 =="
python visualizer/compute_f1.py --data "$DATA"

STAMP=$(date +%Y%m%d%H%M)
sed -i -E "s|data\\.js\\?v=[^\"]+|data.js?v=${STAMP}|" visualizer/visualizer.html || true

python - <<'PY'
import json
from pathlib import Path
text = Path("visualizer/data.js").read_text()
data = json.loads(text[text.index("{"): text.rindex("}") + 1])
missing = []
for eid in data["example_order"]:
    block = data["examples"][eid].get("rollouts_masked") or {}
    raw = block.get("raw") or {}
    if "pred_score" not in raw:
        missing.append(f"{eid}.raw")
        continue
    if "f1" not in raw:
        missing.append(f"{eid}.raw.f1")
print("missing", len(missing), missing[:8])
print("checkpoint_masked", data.get("checkpoint_masked"))
if missing:
    raise SystemExit(1)
print("OK")
PY

echo "Done: $(date)"
