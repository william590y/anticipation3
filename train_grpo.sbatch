#!/bin/bash
#SBATCH --job-name=onpolicy-grpo
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/onpolicy_grpo_%j.out
#SBATCH --error=logs/onpolicy_grpo_%j.out

set -euo pipefail

cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "Host: $(hostname)"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Start: $(date)"

# GRPO on run_paper_split_v2/checkpoint-2500. Replaces the diverging run_grpo
# job (wandb z4eviw60) which used reward = -(rollout CE):
#
#   train/reward_mean went -10.8 -> -19.2 (i.e. CE 10.8 -> 19.2)
#   val/tf_loss_score 0.266 -> 2.35 (left the teacher-forced basin)
#   val/ar_loss 10.76 -> 15.04 (onset CE 22 -> 30)
#   group_reward_std 3.08 -> 0.86 (within-group diversity collapsing)
#   kl_coef was 0, so nothing held the policy to the init checkpoint
#
# New reward is the quantity we actually want to move:
#
#   reward(rollout) = onset_acc + duration_acc + pitch_acc   in [0, 3]
#
# exact token match, no 10 ms bin tolerance -- the same three scores logged as
# val/ar_*_accuracy. Logged to wandb as train/REWARD and val/REWARD.
#
# Hyperparameters retuned for the new scale and the observed divergence:
#   lr 3e-6 -> 1e-6. Even with unit-variance advantages the 3e-6 CE run was
#     already walking off the TF basin; the accuracy-sum reward also has much
#     smaller within-group std (eight T=1 rollouts of one window often agree),
#     which inflates standardised advantages, so a lower LR is required.
#   kl_coef 0 -> 0.05. The k3 KL to the frozen init policy is the trust region
#     the previous run lacked; L2-anchor alone (lambda=1e5) did not stop the
#     functional drift (anchor_term stayed ~3e-4 while TF loss rose 9x).
#   advantage_eps 1e-4 -> 1e-3. Floor on the group std so a near-tie among the
#     eight rollouts cannot send advantages to 1/eps.
#   max_grad_norm 2.0 -> 1.0. Extra clip headroom; the CE run's pre-clip norms
#     were healthy (~0.5-1.3) but the new reward's advantage scale is unknown
#     a priori.
#
# 4 GPUs: 4 windows x 8 rollouts = 32 rollouts/rank x 4 ranks = 128 rollouts/step
# over 16 distinct windows. Same layout as the previous job.

python -m torch.distributed.run --standalone --nproc_per_node=4 \
    train_grpo.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --init_checkpoint run_paper_split_v2/checkpoint-2500 \
    --output_dir ./run_grpo_acc_reward \
    --wandb_run_name run_grpo_acc_reward \
    --wandb_mode online \
    --reward accuracy \
    --learning_rate 1e-6 \
    --kl_coef 0.05 \
    --advantage_eps 1e-3 \
    --max_grad_norm 1.0 \
    --max_steps 5750 \
    --prompts_per_step 4 \
    --group_size 8 \
    --micro_batch 4 \
    --rollout_temperature 1.0 \
    --clip_eps 0.2 \
    --eval_steps 25 \
    --save_steps 250 \
    --eval_windows 96 \
    --val_batch_size 8

echo "End: $(date)"
