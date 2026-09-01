#!/bin/bash
#SBATCH -J testeval
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -c 8
#SBATCH -t 12:00:00
#SBATCH --requeue
#SBATCH -A thickstun
#SBATCH -o /home/wjl86/anticipation3/logs/testeval_%j.out
# Pools over the PAPER-SPLIT TEST FILE, to check whether the 24-window table's
# pairwise32 / pairwise32feat result is a fluctuation.
#
# STRIDE 150 IS NOT ARBITRARY. tokenize-asap-sliding.py advances its window by
# ONE NOTE per line (`for start_idx in range(num_items)`), and a packed window
# holds 138 score notes -- so lines closer than 138 apart share most of their
# content and are not independent draws. 150 guarantees disjoint windows.
# 177,011 test lines / 150 = ~1,180 windows.
#
# bf16 autocast (the default) ON PURPOSE: this is a replication of the bf16
# table's rows, so it must share their numerics. The fp32 question is a
# separate axis, handled by the fp32 pipeline.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;n=torch.cuda.get_device_name(0);print(f'{n} {t:.1f}GiB');assert t>=20,f'refusing {n}'"

S=nbest_data/test9_stride150.pt
# --checkpoint is explicit: generate_nbest's DEFAULT_CKPT is still 7500, while
# every shard and every table row is checkpoint-2500.
# Size to the card handed over: thickstun/ellis/default mix 48GB and 24GB.
WB=$(python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;print(24 if t>=40 else 8)")
FB=$(python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;print(48 if t>=40 else 16)")
echo "window_batch=$WB feature_batch=$FB"
# --resume makes a preemption cost one checkpoint interval, not the whole shard.
python3 -u -m nbest.generate_nbest \
    --token-file data/test_paper.txt --stride 150 --n-sampled 8 \
    --checkpoint run_paper_split_v2/checkpoint-2500 \
    --window-batch "$WB" --output "$S" --save-every 50 --resume
echo "=== pools done; adding per-token features for pairwise32feat ==="
python3 -u -m nbest.add_token_features \
    --shard "$S" --output nbest_data/test9feat_stride150.pt \
    --batch "$FB" --score-chunk 96
echo TESTEVAL_OK
