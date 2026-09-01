#!/bin/bash
#SBATCH -p default_partition
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -J nb-fp32
#SBATCH -o /home/wjl86/anticipation3/logs/nb_fp32_%A_%a.out

# fp32 regeneration of the N-best shards (task 9).
#
# WHY: the original shards were decoded under bf16 autocast, while the
# visualizer's own greedy row is fp32 -- so the F1 table mixed numerics.
# Measured on the 24 viz windows (visualizer/dtype_ab_viz.py): bf16 agrees
# with fp32 on only 63.3% of greedy score tokens. The F1 gap is within noise
# at that sample size and its sign disagrees with a 208-window measurement,
# so this is about removing a confound, not about bf16 being "wrong".
#
# fp32 is not slower here: --fast-decode (anticipation/fast_decode.py) is a
# bit-identical fp32 reimplementation worth ~1.75x at these batch sizes, and
# it also makes ~200 sequences fit on a 48GB card, which the old DynamicCache
# path could not. Net throughput beats the bf16 run it replaces.
#
# Tasks 0-3 = the 9-candidate set (3 train shards + val), matching
# nbest_data/unf_*.pt exactly except for dtype: stride 3, num_shards 3,
# n_sampled 8, NO augmentation.
# Tasks 4-10 = the 32-candidate set (6 train shards + val), matching
# nbest_data/unf32_*.pt: stride 3, num_shards 6, n_sampled 32, augmented
# train shards, unaugmented val.
#
# ALWAYS submit via this file, never --wrap: SLURM expands $VARIABLES in a
# wrap string in the OUTER shell, which once made every task of an array run
# the val branch instead of its own train shard (arrays 369449 and 370733).
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1

T=$SLURM_ARRAY_TASK_ID

# Auto-size the batch to whatever card we land on. fp32 activations are twice
# bf16's, so the old window_batch 10 (330 sequences) OOMs a 24GB 3090 even
# with the static cache; pinning a specific 48GB node loses to reservations.
# Submit with --constraint="a6000|6000ada|a100|a40" -- a bare --gres=gpu:1
# schedules onto 11GB TitanX/1080Ti nodes, where even window_batch 2 OOMs
# (array 470107 lost all 11 tasks that way).
VRAM=$(python -c "import torch;print(int(torch.cuda.get_device_properties(0).total_memory/2**30))")
if   [ "$VRAM" -ge 70 ]; then WB32=10; WB9=36
elif [ "$VRAM" -ge 40 ]; then WB32=6;  WB9=22
elif [ "$VRAM" -ge 20 ]; then WB32=2;  WB9=8
else
  echo "GPU too small (${VRAM}GiB) for an fp32 decode; refusing rather than OOMing"
  exit 1
fi
echo "GPU has ${VRAM}GiB -> window_batch ${WB32} (32-cand) / ${WB9} (9-cand)"

COMMON="--stride 3 --score-chunk 32 --fp32 --fast-decode --save-every 25 --resume"
COMMON="$COMMON --checkpoint run_paper_split_v2/checkpoint-2500"

if [ "$T" -lt 3 ]; then                      # 9-cand train shards
  python -m nbest.generate_nbest --token-file data/train_paper_unfiltered.txt \
    --shard-index "$T" --num-shards 3 --n-sampled 8 --window-batch "$WB9" \
    $COMMON --output "nbest_data/fp32_9_train_shard0${T}.pt"
elif [ "$T" -eq 3 ]; then                    # 9-cand val
  python -m nbest.generate_nbest --token-file data/val_paper_unfiltered.txt \
    --shard-index 0 --num-shards 1 --n-sampled 8 --window-batch "$WB9" \
    $COMMON --output "nbest_data/fp32_9_val_shard00.pt"
elif [ "$T" -lt 10 ]; then                   # 32-cand train shards (augmented)
  S=$(( T - 4 ))
  python -m nbest.generate_nbest --token-file data/train_paper_unfiltered.txt \
    --shard-index "$S" --num-shards 6 --n-sampled 32 --window-batch "$WB32" \
    --augment $COMMON --output "nbest_data/fp32_32_train_shard0${S}.pt"
else                                         # 32-cand val (unaugmented)
  python -m nbest.generate_nbest --token-file data/val_paper_unfiltered.txt \
    --shard-index 0 --num-shards 1 --n-sampled 32 --window-batch "$WB32" \
    $COMMON --output "nbest_data/fp32_32_val_shard00.pt"
fi
echo FP32_GEN_OK
