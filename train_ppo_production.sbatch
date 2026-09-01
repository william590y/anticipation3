#!/bin/bash
#SBATCH --job-name=ppo-corrected
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:3
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/ppo_corrected_%j.out
#SBATCH --error=logs/ppo_corrected_%j.out

set -euo pipefail

cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# The submission wrapper always supplies a fresh absolute run directory.  Every
# hyperparameter can be overridden in the wrapper's environment after the probe
# comparison; the defaults are the conservative stable_v2 Ellis arm.
: "${PPO_RUN_DIR:?PPO_RUN_DIR must name a fresh PPO output directory}"
# PPO_LR is the chain's canonical name; PPO_LEARNING_RATE remains a supported
# compatibility alias.  If both are set, PPO_LR wins.
PPO_LR="${PPO_LR:-${PPO_LEARNING_RATE:-3e-7}}"
PPO_EPOCHS="${PPO_EPOCHS:-2}"
PPO_GAMMA="${PPO_GAMMA:-1.0}"
PPO_GAE_LAMBDA="${PPO_GAE_LAMBDA:-0.95}"
PPO_TARGET_KL="${PPO_TARGET_KL:-0.02}"
PPO_MAX_STEPS="${PPO_MAX_STEPS:-6000}"
PPO_EVAL_STEPS="${PPO_EVAL_STEPS:-25}"
PPO_SAVE_STEPS="${PPO_SAVE_STEPS:-10000}"
PPO_PROMPTS_PER_STEP="${PPO_PROMPTS_PER_STEP:-8}"
PPO_MICRO_BATCH="${PPO_MICRO_BATCH:-4}"
PPO_VALUE_LR="${PPO_VALUE_LR:-3e-5}"
PPO_VALUE_WARMUP_STEPS="${PPO_VALUE_WARMUP_STEPS:-10}"
PPO_WANDB_MODE="${PPO_WANDB_MODE:-online}"
PPO_WANDB_RUN_NAME="${PPO_WANDB_RUN_NAME:-ppo_corrected_${SLURM_JOB_ID}}"

echo "PPO production job ${SLURM_JOB_ID} on $(hostname); start $(date --iso-8601=seconds)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
echo "run_dir=${PPO_RUN_DIR} lr=${PPO_LR} epochs=${PPO_EPOCHS} gamma=${PPO_GAMMA} lambda=${PPO_GAE_LAMBDA} target_kl=${PPO_TARGET_KL}"

python -u -m torch.distributed.run --standalone --nproc_per_node=3 \
    train_ppo.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --init_checkpoint run_paper_split_v2/checkpoint-2500 \
    --output_dir "${PPO_RUN_DIR}" \
    --wandb_run_name "${PPO_WANDB_RUN_NAME}" \
    --wandb_mode "${PPO_WANDB_MODE}" \
    --learning_rate "${PPO_LR}" \
    --max_steps "${PPO_MAX_STEPS}" \
    --prompts_per_step "${PPO_PROMPTS_PER_STEP}" \
    --micro_batch "${PPO_MICRO_BATCH}" \
    --rollout_temperature 0.7 \
    --ppo_epochs "${PPO_EPOCHS}" \
    --gamma "${PPO_GAMMA}" \
    --gae_lambda "${PPO_GAE_LAMBDA}" \
    --clip_eps 0.2 \
    --vf_coef 0.5 \
    --value_lr "${PPO_VALUE_LR}" \
    --value_warmup_steps "${PPO_VALUE_WARMUP_STEPS}" \
    --target_kl "${PPO_TARGET_KL}" \
    --advantage_clip 5.0 \
    --max_grad_norm 1.0 \
    --eval_steps "${PPO_EVAL_STEPS}" \
    --save_steps "${PPO_SAVE_STEPS}" \
    --eval_windows 96 \
    --val_batch_size 8

echo "PPO production job ${SLURM_JOB_ID} end $(date --iso-8601=seconds)"
