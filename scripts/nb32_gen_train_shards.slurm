#!/bin/bash
#SBATCH -p ellis
#SBATCH -A thickstun
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=48G
#SBATCH -c 8
#SBATCH -t 20:00:00
#SBATCH --array=0-5%6
#SBATCH -J nb32-gen3
#SBATCH -o logs/nb32_gen3_%A_%a.out

# Augmented 32+1-candidate N-best train shards (val shard already on disk
# from 370733; every task of that array ran the val branch due to a --wrap
# quoting bug, hence this script file).
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1

T=$SLURM_ARRAY_TASK_ID
python -m nbest.generate_nbest \
  --token-file data/train_paper_unfiltered.txt \
  --stride 3 --shard-index "$T" --num-shards 6 \
  --n-sampled 32 --window-batch 10 --score-chunk 32 \
  --augment \
  --checkpoint run_paper_split_v2/checkpoint-2500 \
  --output "nbest_data/unf32_train_shard0${T}.pt"
echo GEN_TASK_OK
