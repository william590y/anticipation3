#!/bin/bash
#SBATCH --job-name=onpolicy-crpo
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:3
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/onpolicy_crpo_%j.out
#SBATCH --error=logs/onpolicy_crpo_%j.out

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

# CRPO (Wu et al., arXiv:2607.28026) on run_paper_split_v2/checkpoint-2500.
# Replaces the diverging run_crpo job (wandb jncw6zeg) which used reward =
# -(rollout CE) AND the paper's raw lambda=5 / tau=1:
#
#   train/reward_mean -16.7 -> -28.3 (CE 16.7 -> 28.3)
#   val/ar_loss 10.76 -> 23.44, val/tf_loss_score 0.266 -> 2.43
#   val/ar_pitch_accuracy 82.0% -> 67.0%  (collapsing)
#   val/ar_onset_accuracy 13.6% -> 2.5%   (collapsing)
#   pre-clip grad_norm 16 -> 518 by step 50, still 100-400 afterwards
#   KL+ 11.8 -> 22.9 and KL- 8.6 -> 14.5 (both moving AWAY from the teacher)
#
# Two independent bugs stacked:
#   1. CE reward is tail-dominated (same failure as GRPO / plain distillation).
#   2. tau=1 makes Eq. 11's softmax a hard max: sim = -KL runs 6-14 nats here
#      (an LLM token KL is <1), so exp(sim/1) spans ~e^20 and essentially all
#      the gradient goes into the repulsive term. lambda=5 then lets that
#      term dominate L_GRPO.
#
# New reward, shared with the GRPO arm:
#
#   reward(rollout) = onset_acc + duration_acc + pitch_acc   in [0, 3]
#
# exact token match, logged to wandb as train/REWARD and val/REWARD.
#
# Hyperparameters:
#   lr 3e-6 -> 1e-6. Same reason as GRPO: the CE run was already leaving the
#     TF basin, and the accuracy-sum reward has smaller group std so
#     standardised advantages get larger.
#   tau 1.0 -> 10.0. Puts tau on the scale of sim = -KL so the softmax is a
#     soft distribution over positions, not a hard max over two extremes.
#   lambda 5 -> 2. Rebalances L_CRPO against L_GRPO; at 5 the contrastive
#     term was the whole update (grad_norm hundreds, TF loss 3x worse than
#     GRPO-alone at the same step).
#   advantage_eps 1e-4 -> 1e-3, max_grad_norm 2 -> 1. Same floors as GRPO.
#     CRPO does not load a frozen KL reference (the EMA teacher is already a
#     full extra copy on a 48 GB card); the L2 anchor + lower LR + working
#     tau/lambda are the trust region.
#
# 3 GPUs: 2 windows x 8 rollouts = 16 rollouts/rank x 3 ranks = 48 rollouts/step
# over 6 distinct windows. A CRPO group cannot be split across backward passes
# (Eq. 11's softmax spans the whole group), so --micro_batch is unused.

python -m torch.distributed.run --standalone --nproc_per_node=3 \
    train_crpo.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --init_checkpoint run_paper_split_v2/checkpoint-2500 \
    --output_dir ./run_crpo_acc_reward \
    --wandb_run_name run_crpo_acc_reward \
    --wandb_mode online \
    --reward accuracy \
    --learning_rate 1e-6 \
    --advantage_eps 1e-3 \
    --max_grad_norm 1.0 \
    --max_steps 6900 \
    --prompts_per_step 2 \
    --group_size 8 \
    --rollout_temperature 1.0 \
    --crpo_lambda 2.0 \
    --crpo_temperature 10.0 \
    --positive_fraction 0.3 \
    --top_k 100 \
    --ema_teacher_alpha 0.1 \
    --clip_eps 0.2 \
    --eval_steps 25 \
    --save_steps 250 \
    --eval_windows 96 \
    --val_batch_size 8

echo "End: $(date)"
