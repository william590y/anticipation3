#!/bin/bash
#SBATCH -p thickstun
#SBATCH -A thickstun
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -J nb-bigpool
#SBATCH -o /home/wjl86/anticipation3/logs/nb_bigpool_%A_%a.out

# Task 7: repeat the pairwise32 experiment at 64 and 128 candidates.
#
# DESIGN -- CONSTANT WINDOWS. Every arm sees the SAME 35,908 windows (the
# full 6-shard stride-3 split, identical to unf32_*), so the only thing that
# changes between arms is pool depth:
#
#   pairwise32  35,908 windows x  33 = 1.18M candidates   (existing)
#   64-cand     35,908 windows x  65 = 2.33M candidates
#   128-cand    35,908 windows x 129 = 4.63M candidates
#
# This makes the comparison per-window exact -- the same window with a deeper
# pool -- at the cost of the arms no longer being compute-matched (the 128
# arm generates ~4x pairwise32). A deeper pool should lift the ORACLE ceiling
# on identical windows; whether a reranker can reach it is the question.
# Pairs per window grow quadratically (528 -> 2080 -> 8256), so the pairwise
# loss gets far more signal per window as well.
#
# fp32 + --fast-decode for the same reason as generate_fp32.slurm: bit-identical
# to the fp32 baseline and fast enough that precision costs nothing.
# ALWAYS submit via this file, never --wrap (SLURM expands $VARS in the outer
# shell; that silently ruined two earlier arrays).
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1

T=$SLURM_ARRAY_TASK_ID

VRAM=$(python -c "import torch;print(int(torch.cuda.get_device_properties(0).total_memory/2**30))")
# ~200 sequences in flight is the fp32 sweet spot on a 48GB card; scale the
# window batch so that n_sampled+1 candidates per window stays near it.
if   [ "$VRAM" -ge 70 ]; then SEQ=330
elif [ "$VRAM" -ge 40 ]; then SEQ=198
elif [ "$VRAM" -ge 20 ]; then SEQ=66
else echo "GPU too small (${VRAM}GiB) for an fp32 decode; refusing"; exit 1; fi

COMMON="--token-file data/train_paper_unfiltered.txt --stride 3 --num-shards 6"
COMMON="$COMMON --score-chunk 32 --fp32 --fast-decode --save-every 25 --resume"
COMMON="$COMMON --augment --checkpoint run_paper_split_v2/checkpoint-2500"

# Tasks 0-5 = 64-cand shards 0-5; tasks 6-11 = 128-cand shards 0-5. Both
# arms therefore cover all 35,908 windows, same as unf32_*.
if [ "$T" -lt 6 ]; then                       # 64-candidate arm: shards 0-5
  N=64; WB=$(( SEQ / (N + 1) ))
  # Refuse rather than clamp to 1: one window of 129 fp32 sequences does not
  # fit a 24GB card, and clamping would just OOM after the model loads.
  [ "$WB" -lt 1 ] && { echo "card too small for ${N}-cand"; exit 1; }
  echo "64-cand: ${VRAM}GiB -> window_batch ${WB} ($(( WB * (N+1) )) seqs)"
  python -m nbest.generate_nbest $COMMON --shard-index "$T" \
    --n-sampled "$N" --window-batch "$WB" \
    --output "nbest_data/bp64_train_shard0${T}.pt"
else                                          # 128-candidate arm: shards 0-5
  S=$(( T - 6 )); N=128; WB=$(( SEQ / (N + 1) ))
  [ "$WB" -lt 1 ] && { echo "card too small for ${N}-cand"; exit 1; }
  echo "128-cand: ${VRAM}GiB -> window_batch ${WB} ($(( WB * (N+1) )) seqs)"
  python -m nbest.generate_nbest $COMMON --shard-index "$S" \
    --n-sampled "$N" --window-batch "$WB" \
    --output "nbest_data/bp128_train_shard0${S}.pt"
fi
echo BIGPOOL_GEN_OK
