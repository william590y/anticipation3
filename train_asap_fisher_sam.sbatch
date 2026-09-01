#!/bin/bash
#SBATCH --job-name=fisher-sam
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/home/wjl86/anticipation3/logs/fisher_sam_%j.out
#
# EXPERIMENT 1 -- FISHER SAM FINE-TUNE (2026-08-30)
# =================================================
# Sharpness-aware minimization asks for weights whose whole NEIGHBOURHOOD has
# low loss, by taking the gradient at the locally worst point instead of at w:
#
#     min_w  max_{||e|| <= rho}  L(w + e)
#
# Plain SAM (Foret et al., ICLR 2021) measures that neighbourhood with a
# Euclidean ball, so the ascent is e_w = rho * g/||g||_2 -- every coordinate is
# equally "far". Fisher SAM (Kim et al., ICML 2022, "Fisher SAM: Information
# Geometry and Sharpness Aware Minimisation") replaces the ball with the
# ellipsoid of the diagonal empirical Fisher (squared gradients), which is the
# natural metric on the model's distribution rather than on its coordinates:
#
#     u_i = g_i / (1 + eta * g_i^2)
#     e_w = rho * u / ||u||_2
#           = rho * (g_i/(1+eta g_i^2)) / sqrt( sum_j g_j^2/(1+eta g_j^2)^2 )
#
# Sharp coordinates (large |g_i|) get damped, flat ones keep their share of the
# budget. eta = 0 collapses the formula back to plain SAM exactly, which is how
# sam.py's self-test pins the implementation down. rho 0.05 / eta 1.0 are the
# paper's defaults. Implementation + self-test: sam.py; wiring: train.py
# (`--optimizer fisher_sam`, `sam is None` keeps adamw/muon untouched).
#
# WHY m-SHARPNESS (per-micro-batch), not per-effective-batch
# ----------------------------------------------------------
# SAM needs TWO forward/backwards per batch: g at w, then g' at w + e_w. With
# gradient accumulation there is a choice of where the "batch" boundary sits.
# We perturb per MICRO-batch (m = 4 sequences here, see the factorization note
# below): for each micro-batch we zero the grads, backprop g_i, ascend, zero
# again, backprop g'_i at the perturbed point, restore the weights bitwise, and
# hand-accumulate g'_i into a private fp32 buffer that is copied into .grad at
# the accumulation boundary. That private buffer is the whole reason for the
# manual bookkeeping: the ascent direction has to be built from THIS
# micro-batch's gradient alone, so .grad cannot simultaneously hold the running
# sum of the earlier micro-batches. The alternative (accumulate the whole
# window, perturb once, replay all 16 micro-batches) costs a third pass over
# the window and is the WORSE end of the knob in the SAM literature -- Foret et
# al. report that smaller m generalizes better ("m-sharpness"), and the ICML
# Fisher-SAM experiments use the per-micro-batch form as well.
#
# COST: 2 forward/backward passes per micro-batch, so expect ~2x the usual s/it
# (plus a few % for the ascent/restore elementwise work). 10k steps at eff.
# batch 64 -- the same optimization budget as the maskft/randreplace arms, so
# the arm stays directly comparable to them.
#
# WHY 4 x 16 AND NOT 8 x 8 (measured, not guessed)
# ------------------------------------------------
# The effective batch is the invariant (64), the factorization is not. SAM
# needs two full-size fp32 buffers that plain AdamW does not: the pre-ascent
# weights (bitwise restore) and the m-sharpness gradient accumulator, i.e.
# +2 x 1372 MiB on this 359.7M-param model. Measured on ellis-compute-02
# (smoke job 937455, identical recipe): at batch 8 x accum 8 this run occupies
# 27.0 GB of an A6000 -- more than a 3090 physically has (24.6 GB), so 8 x 8
# OOMs here. The reference point is the concurrent adamw arm randreplace15
# (job 937360, batch 8 x accum 8 on a 3090): 20,956 MiB peak allocated, which
# leaves 2.6 GB of headroom against SAM's 2.7 GB of extra buffers.
# Halving the micro-batch halves the activation/logit transient (the dominant
# term at seq 1020 x vocab 55028) and buys ~7 GB, while 4 x 16 = 64 keeps the
# effective batch exactly where every other arm has it. This is the same trade
# train_asap_paper_maskft_muon_ellis.sbatch made when it moved to 24GB cards.
# Side effect, and a favourable one: m-sharpness is now measured on m=4
# instead of m=8, and smaller m is the better-generalizing end of that knob.
#
# HARDWARE: ONE 3090 via a TYPED gres. A bare `gpu:1` on ellis gets handed
# whatever card the node feels like (job 572394 was handed an RTX 2080 Ti under
# a --constraint string), so the type goes in the gres and a runtime preflight
# asserts a real bf16 matmul on a device whose name contains 3090. Host RAM is
# 16G: TokenizedDataset only keeps line offsets and seeks per item, so the
# 12 GB train_paper.txt never lands in RAM. Single GPU is mandatory here --
# train.py refuses fisher_sam with world_size > 1 because the m-sharpness
# accumulator is rank-local.
#
# EFFECTIVE BATCH 64 = 4 x 16 x 1, the invariant every arm in this repo shares.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "Host: $(hostname)"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Start: $(date)"

# Runtime preflight: a real bf16 matmul on a real 3090. NVML failures on a
# node-wide-dead GPU (ma-compute-02 / kuleshov-compute-01 / badfellow class of
# failure) surface here in seconds instead of as a bogus "setup doesn't support
# bf16" hours in, and the name check catches a mis-handed card.
python -c "
import torch
assert torch.cuda.is_available(), 'no CUDA device'
assert torch.cuda.device_count() == 1, f'expected 1 GPU, got {torch.cuda.device_count()}'
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
x = torch.randn(512, 512, device='cuda', dtype=torch.bfloat16)
y = (x @ x).float().sum().item()
assert torch.isfinite(torch.tensor(y)), 'bf16 matmul produced a non-finite result'
assert '3090' in name, f'refusing non-3090 card: {name}'
print(f'preflight OK: {name}, {total/2**30:.1f} GiB total / {free/2**30:.1f} GiB free, bf16 matmul live')
"

# Fisher SAM fine-tune of the paper-split checkpoint. Everything except the
# optimizer matches train_asap_paper_maskft.sbatch's recipe (same init, same
# data, same effective batch, same 10k steps) so the comparison is clean.
python -u train.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --model_name run_paper_split_v2/checkpoint-2500 \
    --output_dir ./run_paper_split_v2_fisher_sam \
    --wandb_run_name run_paper_split_v2_fisher_sam \
    --wandb_mode online \
    --optimizer fisher_sam \
    --sam_rho 0.05 \
    --sam_eta 1.0 \
    --max_steps 10000 \
    --batch_size 4 \
    --val_batch_size 8 \
    --gradient_accumulation_steps 16 \
    --save_steps 1000 \
    --eval_steps 500

echo "End: $(date)"
echo FISHER_SAM_DONE
