#!/bin/bash
#SBATCH --job-name=diffdraft-kd
#SBATCH --partition=default_partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=48G
#SBATCH --time=6:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/diffdraft_kd_%A_%a.out
#SBATCH --error=logs/diffdraft_kd_%A_%a.out
# Teacher greedy rollouts for the diffusion drafter's KD corpus.
#
# Tasks 0-3 = train shards of data/train_paper.txt; task 4 = the validation
# corpus from data/val_paper.txt. A6000s (48 GB) rather than the ellis 3090s:
# the target runs in fp32 and the rollout's DynamicCache doubles transiently
# every step, so batch 96 needs ~30 GB, and the ellis 3090s are reserved for the
# other agents' bench-*/gate-* timing jobs (a co-tenant job would poison their
# wall-clock numbers, and mine).
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
# See bench/bench_decode.sbatch: the growing DynamicCache holds old+new cache at
# once, so without expandable_segments the freed blocks fragment away.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

if [ "${SLURM_ARRAY_TASK_ID}" -eq 4 ]; then
  python gen_diffdraft_kd.py \
    --token-file data/val_paper.txt \
    --num-windows 2048 --batch-size 96 \
    --shard 0 --num-shards 1 \
    --out data/diffdraft_kd_val
else
  python gen_diffdraft_kd.py \
    --token-file data/train_paper.txt \
    --num-windows 40000 --batch-size 96 \
    --shard "${SLURM_ARRAY_TASK_ID}" --num-shards 4 \
    --out data/diffdraft_kd_train
fi
