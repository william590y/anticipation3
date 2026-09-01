#!/bin/bash
#SBATCH --job-name=viz-m40k-3090
#SBATCH --partition=ellis
#SBATCH --nodes=1
#SBATCH --nodelist=ellis-compute-01
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:3
#SBATCH --mem=64G
#SBATCH --time=3:00:00
#SBATCH --output=logs/viz_masked_40k_3090_%j.out
#SBATCH --error=logs/viz_masked_40k_3090_%j.out

# 3×3090 shard of both 40k masked checkpoints (global ranks 0-2 of 5).
# Sized to currently free Ellis 3090s (do not wait for a 4th).
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

DATA="${DATA:-visualizer/data.js}"
OUT_DIR="${OUT_DIR:-visualizer/masked_40k_shards}"
RANK_OFFSET="${RANK_OFFSET:-0}"
LOCAL_RANKS="${LOCAL_RANKS:-3}"
WORLD_SIZE="${WORLD_SIZE:-5}"

mkdir -p logs "$OUT_DIR"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID

echo "Host: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Ranks ${RANK_OFFSET}..$((RANK_OFFSET + LOCAL_RANKS - 1)) / ${WORLD_SIZE}"
echo "Start: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

mapfile -t DEVS < <(python - <<PY
import os
raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
ids = [x.strip() for x in raw.split(",") if x.strip()]
need = int("${LOCAL_RANKS}")
if ids:
    print("\n".join(ids[:need]))
else:
    print("\n".join(str(i) for i in range(need)))
PY
)

if [[ "${#DEVS[@]}" -lt "$LOCAL_RANKS" ]]; then
  echo "Need ${LOCAL_RANKS} GPUs, got ${#DEVS[@]}" >&2
  exit 1
fi

fail=0
pids=()
for local in $(seq 0 $((LOCAL_RANKS - 1))); do
  rank=$((RANK_OFFSET + local))
  out="${OUT_DIR}/shard_$(printf '%02d' "$rank").json"
  echo "Launch rank ${rank} on GPU ${DEVS[$local]} -> ${out}"
  CUDA_VISIBLE_DEVICES="${DEVS[$local]}" python visualizer/precompute_masked_40k_rollouts.py \
      --data "$DATA" \
      --rank "$rank" \
      --world-size "$WORLD_SIZE" \
      --output "$out" &
  pids+=($!)
done

for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    echo "rank pid $pid failed" >&2
    fail=1
  fi
done
if [[ "$fail" -ne 0 ]]; then
  exit 1
fi
echo "3090 shards done: $(date)"
