#!/bin/bash
#SBATCH --job-name=onpolicy-distill
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:3
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/onpolicy_distill_%j.out
#SBATCH --error=logs/onpolicy_distill_%j.out

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

# On-policy distillation of run_paper_split_v2/checkpoint-2500 -- the lowest
# teacher-forced validation loss (1.2180) non-LoRA checkpoint, i.e. the visualizer's
# "best val loss" base model. The model rolls itself out over the 138 body score
# slots (controls teacher-forced, exactly as at inference) and is graded at every
# generated token by the cross-entropy of the ground-truth next token given its own
# prefix. Headline metric: val/ar_loss, the same cross-entropy measured on a fixed
# 96-window validation subset along a greedy rollout.
#
# 3 GPUs: 16 windows/rank x 3 ranks = 48 rollouts per optimizer step. (Effective
# batch 64 is not reachable on 3 ranks -- 64 = 2^6 is not divisible by 3 -- so the
# per-rank batch is kept and the global one lands at 48.)
#
# The rollout is 414 sequential KV-cached forwards, so it is launch-bound rather than
# compute-bound -- measured on one RTX 6000 Ada, rolling out 8, 16 and 24 sequences
# all take ~4s (3.0 / 3.9 / 4.9 GiB peak). The update pass is the memory-bound half
# (14.3 GiB at micro_batch 4, 23.8 at 8), so it is chunked separately. Measured
# ~9.8 s/step amortized with the GRPO run sharing the node => ~18 h for 6600 steps.
#
# Rollout temperature 1.0: measured over the constrained score distribution, greedy
# and T=1 rollouts land at the same CE (val 1.085 vs 1.112) because most score
# tokens are near-deterministic given the aligned control -- so sampling costs
# nothing in state quality and gives GRPO the within-group spread it needs. Both
# runs therefore train on the same rollout distribution.
#
# Augmentation: transposition and tempo scaling from the original fine-tune are kept;
# the noise injections (per-onset/duration jitter, score-embedding masking) are off.
#
# LEARNING RATE: 1e-7. First launched at 1e-5, then 1e-6 -- a 10x cut barely moved
# the failure (step-25 pitch accuracy 19.5% at 1e-5 vs 20.5% at 1e-6), so this is not
# a step-size problem. At 1e-5 the model collapsed inside 25 steps -- val/ar_loss fell
# 10.76 -> 3.65 while val/tf_loss_score rose 0.266 -> 3.105 and AR pitch accuracy fell
# 81.97% -> 19.49%; at 1e-6 the same 25 steps gave tf 1.669 and 20.47%. The mean CE over a
# rollout is tail-dominated (median per-token CE ~0, p99 ~40 nats) and, once the
# rollout has drifted, the ground-truth onset is frequently unreachable from the
# timeline the model has already emitted -- so the cheapest way to lower the mean is
# to flatten the predictive distribution everywhere, which is exactly what it did.
# Watch val/tf_loss_score and val/ar_pitch_accuracy, not val/ar_loss alone: ar_loss
# going down while those two go the wrong way is this failure, not progress.

python -m torch.distributed.run --standalone --nproc_per_node=3 \
    train_onpolicy_distill.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --init_checkpoint run_paper_split_v2/checkpoint-2500 \
    --output_dir ./run_onpolicy_distill \
    --wandb_run_name run_onpolicy_distill \
    --wandb_mode online \
    --learning_rate 1e-7 \
    --max_steps 6500 \
    --prompts_per_step 16 \
    --micro_batch 4 \
    --rollout_temperature 1.0 \
    --eval_steps 25 \
    --save_steps 250 \
    --eval_windows 96 \
    --val_batch_size 8

echo "End: $(date)"
