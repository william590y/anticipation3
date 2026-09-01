#!/bin/bash
#SBATCH --job-name=ppo-select-best
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/ppo_select_best_%j.out
#SBATCH --error=logs/ppo_select_best_%j.out

set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

: "${PPO_RUN_DIR:?PPO_RUN_DIR is required}"
: "${PPO_MANIFEST:?PPO_MANIFEST is required}"
: "${PPO_TRAIN_JOB_ID:?PPO_TRAIN_JOB_ID is required}"

python -u visualizer/select_ppo_best.py \
    --run-dir "${PPO_RUN_DIR}" \
    --output "${PPO_MANIFEST}" \
    --train-job-id "${PPO_TRAIN_JOB_ID}" \
    --learning-rate "${PPO_LR:-3e-7}" \
    --ppo-epochs "${PPO_EPOCHS:-2}" \
    --gamma "${PPO_GAMMA:-1.0}" \
    --gae-lambda "${PPO_GAE_LAMBDA:-0.95}" \
    --target-kl "${PPO_TARGET_KL:-0.02}"
