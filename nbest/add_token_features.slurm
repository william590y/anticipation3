#!/bin/bash
#SBATCH -p ellis
#SBATCH -A thickstun
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 8:00:00
#SBATCH --array=0-6%2
#SBATCH -J tokfeat32
#SBATCH -o logs/tokfeat32_%A_%a.out

# Per-token log-prob features for the 32-candidate shards (array task 6 = val).
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1

T=$SLURM_ARRAY_TASK_ID
if [ "$T" -lt 6 ]; then
  IN="nbest_data/unf32_train_shard0${T}.pt"
  OUT="nbest_data/tokfeat32_train_shard0${T}.pt"
else
  IN="nbest_data/unf32_val_shard00.pt"
  OUT="nbest_data/tokfeat32_val_shard00.pt"
fi
python -m nbest.add_token_features --shard "$IN" --output "$OUT" --batch 48 --score-chunk 96
echo TOKFEAT_OK
