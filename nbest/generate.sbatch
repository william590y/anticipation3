#!/bin/bash
#SBATCH --job-name=nbest-gen
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=24:00:00
#SBATCH --array=0-2%2
#SBATCH --output=logs/nbest_gen_%A_%a.out
#SBATCH --error=logs/nbest_gen_%A_%a.out
# N-best candidate generation: tasks 0,1 = train shards (stride 40, 2 shards);
# task 2 = val (stride 100). %2 caps usage at the 2 allotted ellis GPUs.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
T="$SLURM_ARRAY_TASK_ID"
if [ "$T" -lt 2 ]; then
  python -m nbest.generate_nbest --token-file data/train_paper.txt \
    --stride 40 --shard-index "$T" --num-shards 2 --n-sampled 8 \
    --window-batch 12 --score-chunk 8 \
    --output "nbest_data/train_shard0${T}.pt"
else
  python -m nbest.generate_nbest --token-file data/val_paper.txt \
    --stride 100 --shard-index 0 --num-shards 1 --n-sampled 8 \
    --window-batch 12 --score-chunk 8 \
    --output nbest_data/val_shard00.pt
fi
