#!/bin/bash
#SBATCH --job-name=diffdraft-train
#SBATCH --partition=default_partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/diffdraft_train_%j.out
#SBATCH --error=logs/diffdraft_train_%j.out
# Masked-diffusion drafter, distilled from the target's greedy rollouts.
#
# Single A6000: the model is 6 x 1024 (75M non-embedding) and the batch is short
# sequences, so this is not worth the multi-GPU launch overhead, and the ellis
# 3090s belong to the other agents' timing runs.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python train_diffdraft.py "$@"
