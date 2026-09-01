#!/bin/bash
#SBATCH --job-name=bayes-lora-r512
#SBATCH --partition=ellis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=logs/bayes_lora_%j.out
#SBATCH --error=logs/bayes_lora_%j.out

set -euo pipefail

# EXPERIMENT 3 -- HIGH-RANK BAYESIAN LoRA (r=512).
#
# METHOD. Ordinary LoRA learns a point estimate of the low-rank update
# Delta_W = (alpha/r) B A. This arm instead does mean-field variational
# inference (Bayes-by-Backprop, Blundell et al., ICML 2015) over the adapter:
# every entry of every lora_B matrix gets a factorized Gaussian posterior
# q(B) = N(mu, sigma^2) with sigma = softplus(rho). During TRAINING each forward
# pass draws a fresh reparameterized sample B = mu + sigma * eps, eps ~ N(0, I),
# so the gradient flows into both mu and rho. During EVAL (validation, the
# autoregressive decode, and merging) the posterior MEAN mu is used, which makes
# every downstream artifact the deterministic LoRA model. The objective is the
# KL WEIGHT (2026-08-30): the principled ELBO (kl_weight=1) was MEASURED at
# step 0 to put KL_term/CE = 9.6e3 -- 62.9M lora_B posteriors x 6.41 nats each,
# divided by the 33,663 minibatches per epoch. At that weight the KL's steady
# pull on mu dominates the task signal and the arm degenerates into a strong
# adapter weight-decay run rather than a test of posterior uncertainty. Using
# the tempered/cold-posterior setting kl_weight=1e-4 (the measured parity point,
# 1.04e-4) keeps KL and CE comparable so the result is interpretable against
# run_paper_split_lora_r512. Collapse remains visible via train/posterior_sigma_*.
# ELBO: cross-entropy plus kl_weight * KL(q || N(0, prior_std^2)) / M, where M is
# the number of optimizer steps per epoch (the standard minibatch weighting).
#
# WHY THE POSTERIOR IS ON B ONLY (NOT A). Delta_W = (alpha/r) B A is invariant to
# (A, B) -> (cA, B/c) and to any invertible mixing of the r rank directions, so a
# posterior over both factors is non-identifiable: many (q_A, q_B) pairs induce
# the same distribution over Delta_W and the KL term is free to move along that
# degenerate direction instead of being informed by the data. Restricting the
# posterior to B keeps the map rho -> distribution over Delta_W injective given
# A. B is additionally the factor LoRA zero-initializes, so with a tiny
# init_sigma (1e-4) training starts *at* the deterministic LoRA solution and the
# posterior width has to be earned. See bayes_lora.py for the implementation and
# its self-test.
#
# SETTINGS. Rank/alpha (r=512, alpha=1024, targets c_attn,c_proj, dropout 0.05)
# mirror the existing high-rank arm train_lora_highrank_paper.sbatch so this is
# comparable to run_paper_split_lora_r512. Initialised from the supervised
# checkpoint run_paper_split_v2/checkpoint-2500 and trained on the external
# paper split (train_paper / val_paper; the 59-performance TEST split is held
# out). Effective batch stays at the repo-wide 64.
#
# BATCH SPLIT: 4 x 16, not 8 x 8. Measured on an A6000 with these exact
# settings (smoke job 937437): batch_size=8 peaks at 33.6 GiB of GPU memory,
# which does not fit a 24 GiB 3090. batch_size=4 x grad_accum=16 keeps the
# effective batch at 64 and is the same split the ORIGINAL single-GPU r=512 run
# used (17.9 GiB peak allocated, job 821113; the Bayesian rho parameters and the
# sampled-weight activations add roughly 2 GiB on top).
#
# HARDWARE. ONE RTX 3090 via a TYPED gres request: `--constraint` selects the
# NODE and has handed us a 2080 Ti before (job 572394), whereas
# gpu:nvidia_geforce_rtx_3090:1 asks for the card itself. The runtime preflight
# below still runs a REAL bf16 matmul and asserts the device name, because
# node-wide-dead NVML (ma-compute-02 / kuleshov-compute-01 / badfellow) turns
# into a misleading "setup doesn't support bf16" error minutes later otherwise.

# Run from the repo so the local `anticipation` package and `data/` paths resolve.
cd /home/wjl86/anticipation3

# Activate the base conda environment used for interactive runs.
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

# Avoid CPU thread oversubscription from the dataloader/eval workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Start: $(date)"

# --- runtime preflight: a real bf16 matmul on the card we were actually given ---
python - <<'PREFLIGHT'
import torch
assert torch.cuda.is_available(), "no CUDA device visible"
assert torch.cuda.device_count() == 1, f"expected 1 GPU, got {torch.cuda.device_count()}"
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
print(f"preflight: {name}, {total/2**30:.1f} GiB total, {free/2**30:.1f} GiB free")
assert "3090" in name, f"refusing to run on {name}: wanted an RTX 3090"
assert total / 2**30 >= 20, f"refusing {name} with only {total/2**30:.1f} GiB"
a = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16)
c = (a @ b).float()
assert torch.isfinite(c).all(), "bf16 matmul produced non-finite values"
print(f"preflight: bf16 matmul OK (checksum {c.sum().item():.3f})")
PREFLIGHT

# Bayesian LoRA r=512 from the supervised paper-split checkpoint.
python train_lora.py \
    --model_name run_paper_split_v2/checkpoint-2500 \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --output_dir ./run_bayes_lora_r512 \
    --wandb_run_name run_bayes_lora_r512 \
    --wandb_mode online \
    --max_steps 10000 \
    --original_weight_l2 0 \
    --lora_r 512 \
    --lora_alpha 1024 \
    --lora_dropout 0.05 \
    --lora_target_modules c_attn,c_proj \
    --bayesian \
    --kl_weight 1e-4 \
    --prior_std 0.1 \
    --init_sigma 1e-4 \
    --batch_size 4 \
    --gradient_accumulation_steps 16 \
    --val_batch_size 4 \
    --save_steps 1000 \
    --eval_steps 500

echo "End: $(date)"
