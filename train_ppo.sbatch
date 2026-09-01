#!/bin/bash
#SBATCH --job-name=onpolicy-ppo
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:3
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/onpolicy_ppo_%j.out
#SBATCH --error=logs/onpolicy_ppo_%j.out

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

# Token-level PPO on run_paper_split_v2/checkpoint-2500 -- the same starting weights
# as the GRPO arm, so the two differ only in how the reward is delivered and who
# assigns credit:
#
#   GRPO  one scalar per 414-token rollout, spread uniformly over every token;
#         the baseline is the mean of a group of 8 samples of the same window.
#   PPO   +1 / -1 on EACH token as it is generated, a learned critic as the
#         baseline, GAE(gamma=1, lambda=0.95) propagating the downstream cost of a
#         bad token back to the token that caused it.
#
# The reward is indexed to the action that earned it: rewards[:, t] scores
# positions[t], the token just emitted, and the bootstrap reads V(s_{t+1}). That
# is what lets the advantage separate "this token was wrong" from "this token was
# fine but the rollout drifted after it" -- which matters here specifically
# because pitch is a copy of a control already in context (verified: the score
# pitch equals the paired control's pitch in 55200/55200 body slots), so a wrong
# onset does not just cost its own token, it desynchronises the copy for every
# slot that follows.
#
# The critic is a 1024->256->1 MLP on DETACHED final hidden states: the value loss
# trains the head alone and never reshapes the policy trunk, which keeps the DDP
# graph to two independently-reduced modules under one backward and stops a fresh
# critic from perturbing a converged policy. --value_warmup_steps 10 updates only
# the critic first, so the zero-init head is not injecting noise into every
# advantage. During warmup the policy optimiser does NOT step at all: stepping Adam
# on a near-zero gradient collapses its second moment, and the first real gradient
# then lands ~30x oversized (observed as approx_kl 0.003 -> 4.2e8 at the exact step
# warmup ended).
#
# Two trust-region valves, because a fresh critic is dangerous here. approx_kl is a
# MEAN over 3312 tokens, so the observed 1.6e5 was ONE token whose log-prob moved
# ~20 nats, not a broad policy shift: with gamma=1 over 414 steps the returns span
# +/-414, a critic at explained_variance ~0.03 emits outlier predictions, and after
# whitening a lone outlier among 3312 tokens sits at ~sqrt(3312) = 57 sigma.
#   --advantage_clip 5.0  bounds the per-token update whatever the critic does.
#   --target_kl 0.05      abandons the remaining inner epochs if it runs away anyway.
# --value_lr is also lowered to 3e-5 so the critic chases those large targets less
# violently. Watch train/explained_variance: at or below 0 means the critic is not
# yet a useful baseline and the advantages are mostly raw reward.
#
# Sampling: training rolls out at T=0.7, validation is greedy (evaluate_autoregressive
# hardcodes temperature 0). Exact-match accuracy is far more temperature-sensitive
# than cross-entropy -- the smoke run at T=1.0 scored 0.61-0.85 on training rollouts
# against 2.51-2.58 greedy on validation, almost all of it sampling noise on tokens
# greedy gets right.
#
# 3 GPUs: 8 windows/rank x 3 ranks = 24 rollouts/step, one rollout per window (PPO
# needs no group). 4 PPO epochs per rollout batch, since rollouts dominate wall clock
# and the clip is what keeps the reuse honest.

python -m torch.distributed.run --standalone --nproc_per_node=3 \
    train_ppo.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --init_checkpoint run_paper_split_v2/checkpoint-2500 \
    --output_dir ./run_ppo \
    --wandb_run_name run_ppo \
    --wandb_mode online \
    --learning_rate 1e-6 \
    --max_steps 6000 \
    --prompts_per_step 8 \
    --micro_batch 4 \
    --rollout_temperature 0.7 \
    --ppo_epochs 4 \
    --gamma 1.0 \
    --gae_lambda 0.95 \
    --clip_eps 0.2 \
    --vf_coef 0.5 \
    --value_lr 3e-5 \
    --value_warmup_steps 10 \
    --target_kl 0.05 \
    --advantage_clip 5.0 \
    --max_grad_norm 1.0 \
    --eval_steps 25 \
    --save_steps 250 \
    --eval_windows 96 \
    --val_batch_size 8

echo "End: $(date)"
