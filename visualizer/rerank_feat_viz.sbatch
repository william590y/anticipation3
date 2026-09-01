#!/bin/bash
#SBATCH --job-name=rerank-feat-viz
#SBATCH --partition=ellis
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --output=/home/wjl86/anticipation3/logs/%x_%j.out
#SBATCH --error=/home/wjl86/anticipation3/logs/%x_%j.err

# The three Table-B methods that had no row in the 24-window F1 table:
#   pairwise32feat_0822  (token_features=2)  -> rollouts_rerank_pw32feat
#   duel32_0822          (token_features=2)  -> rollouts_duel32
#   genrm_0822           (the AMT itself)    -> rollouts_genrm
# ListT5 is deliberately EXCLUDED: it collapsed to a constant permutation
# (crowned-slot histogram {0: 120}), so its row would report which arbitrary
# index the collapse landed on, not a selection method.
#
# The first two were previously unscoreable: rerank_sample_viz.py passed no
# `feats`, and Reranker/DuelComparator default feats=None and silently skip
# the feature pathway. That is now plumbed (and refuses to run without it).
#
# ONE process, ALL 24 keys, ONE pool shared by every selector -- the pool is
# seeded per window (order.index(key)), so this reproduces the pool the
# existing rows were selected from. --output is the fitted alpha/beta/gamma
# shard, kept as a POOL-IDENTITY CHECK against the merged all_v3.json rather
# than for merging.
#
# Deliberately NOT --fp32: every existing selection row predates that flag and
# was decoded under bf16 autocast (no shard carries a decode_dtype key). New
# rows must share numerics with the rows they will sit beside.
set -euo pipefail
cd /home/wjl86/anticipation3
export PYTHONPATH=/home/wjl86/anticipation3:${PYTHONPATH:-}

nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -c "
import torch; n=torch.cuda.get_device_name(0)
free,total=torch.cuda.mem_get_info()
print(f'device {n}  {total/2**30:.1f} GiB')
assert total/2**30 >= 20, f'refusing: {n} has only {total/2**30:.1f} GiB'
torch.zeros(8,device='cuda',dtype=torch.bfloat16) @ torch.zeros(8,8,device='cuda',dtype=torch.bfloat16)
print('bf16 matmul OK')
"

OUT=visualizer/rerank_feat_shards
python3 -u visualizer/rerank_sample_viz.py \
    --checkpoint run_paper_split_v2/checkpoint-2500 \
    --variant raw --n-sampled 8 --seed 0 \
    --output              $OUT/fitted_poolcheck.json \
    --qsel-reranker       run_nbest_reranker/pairwise32feat_0822/final.pt \
    --output-qsel         $OUT/qsel_pw32feat.json \
    --duel-ckpt           run_nbest_reranker/duel32_0822/final.pt \
    --output-tournament   $OUT/tournament_duel32.json \
    --tournament-seed 0 \
    --genrm-ckpt          run_nbest_reranker/genrm_0822/final.pt \
    --output-genrm        $OUT/genrm.json

echo "=== decode done ==="
ls -la $OUT/
