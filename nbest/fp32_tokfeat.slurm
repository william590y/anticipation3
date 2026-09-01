#!/bin/bash
#SBATCH -J fp32tokfeat
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 10:00:00
#SBATCH --requeue
#SBATCH -A thickstun
#SBATCH --array=0-6%3
#SBATCH -o /home/wjl86/anticipation3/logs/fp32tokfeat_%A_%a.out
# Per-token features in FULL precision over the 33-candidate fp32 shards.
# Task 6 = val. --fp32 covers all three precision paths (FT autocast, base
# WEIGHT dtype, feature storage) and rewrites cand_logp_base, which in the
# existing fp32_* shards was computed with a bf16 base model.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;n=torch.cuda.get_device_name(0);print(f'{n} {t:.1f}GiB');assert t>=20,f'refusing {n}'"
T=$SLURM_ARRAY_TASK_ID
if [ "$T" -lt 6 ]; then IN="nbest_data/fp32_32_train_shard0${T}.pt"; OUT="nbest_data/fp32feat_train_shard0${T}.pt";
else IN="nbest_data/fp32_32_val_shard00.pt"; OUT="nbest_data/fp32feat_val_shard00.pt"; fi
test -f "$IN" || { echo "missing $IN"; exit 1; }
# Size the batch to the card actually handed over. default_partition mixes
# 24GB and 48GB cards, and a fixed batch 48 assumes the 48GB one.
B=$(python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;print(48 if t>=40 else 16)")
echo "batch=$B"
python3 -m nbest.add_token_features --shard "$IN" --output "$OUT" --batch "$B" --score-chunk 96 --fp32
echo FP32TOKFEAT_OK
