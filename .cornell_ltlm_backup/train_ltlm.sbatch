#!/bin/bash
#SBATCH --job-name=train-ltlm
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# thickstun is a single 8x RTX 6000 Ada (48GB) node. Pin the type so a
# future partition mix cannot silently land this on a smaller card.
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:4
#SBATCH --mem=128G
#SBATCH --time=infinite
#SBATCH --output=logs/train_ltlm_%j.out
#SBATCH --error=logs/train_ltlm_%j.out

set -euo pipefail

# Run from the repo so the local `anticipation` package and `data/` paths resolve.
cd /home/wjl86/anticipation3

source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

mkdir -p logs

RUN_DIR="${RUN_DIR:-./run_ltlm}"
MAX_STEPS="${MAX_STEPS:-20000}"
WANDB_MODE="${WANDB_MODE:-online}"
# Rate knob on the plan. Under --elbo_reduction mean the KL is averaged over
# n_thoughts*z_dim = 98304 latent elements while the NLL is averaged over ~1019
# tokens, so kl_weight=1 underweights KL by ~E/T = 96x relative to the true
# ELBO. At 1.0 fast learning has ~98k effectively unregularised parameters to
# fit a single window with, and it uses them: the oracle plan becomes so
# window-specific that nothing which has not seen the score can reproduce it
# (run_ltlm, job 52067: oracle AR duration 68.7% while prefix and diffusion
# both collapsed to 0.53%, with cosine-to-oracle decaying 0.038 -> 0.021).
# kl_weight=96 is true-ELBO equivalence.
KL_WEIGHT="${KL_WEIGHT:-1.0}"

echo "Host: $(hostname)"
echo "run_dir=${RUN_DIR} kl_weight=${KL_WEIGHT} max_steps=${MAX_STEPS}"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: $(which python)"
echo "Job ${SLURM_JOB_ID:-none}; start $(date --iso-8601=seconds)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# Effective batch 64: batch_size=4 x grad_accum=4 x 4 GPUs. Same global batch
# as train_asap_paper.sbatch (4 x 8 x 2) and train_plan_lm.sbatch (4 x 4 x 4).
# NCCL P2P is fine on thickstun (the ellis-compute-02 hang does not apply).
GPUS=$(nvidia-smi --list-gpus | wc -l)
if [[ "${GPUS}" -ne 4 ]]; then
    echo "Expected 4 GPUs (effective batch 4 x 4 x 4 = 64), got ${GPUS}." >&2
    exit 1
fi

python -u -m torch.distributed.run --standalone --nproc_per_node=4 \
    train_ltlm.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --output_dir "${RUN_DIR}" \
    --wandb_run_name "ltlm_${SLURM_JOB_ID:-local}" \
    --wandb_mode "${WANDB_MODE}" \
    --max_steps "${MAX_STEPS}" \
    --batch_size 4 \
    --val_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --thoughts_per_layer 4 \
    --mcmc_steps 16 \
    --inference_method adamVI \
    --kl_weight "${KL_WEIGHT}" \
    --diffusion_hidden 1024

# NO performance-token masking anywhere in this run, which is what keeps
# val/loss directly comparable with run_paper_split_v2 and run_plan_lm:
#   * the ELBO's likelihood covers the whole window (--posterior_score_only is
#     available but off, matching the paper);
#   * the decoder's train/val loss covers the whole window
#     (--loss_mask_performance_tokens, train.py's flag, is likewise off).
# The score-only variants are one flag each if this is revisited; the prefix
# validation arm is unaffected either way because it builds its observation
# from input_ids (ltlm.prefix_observation_labels).
#
# --diffusion_hidden 1024: must be >= z_dim (the GPT-2 hidden size). The
#   denoiser's z_out is a single linear, so a narrower model cannot represent
#   the full epsilon and DDIM carries the remainder into the sampled plan.

echo "Job ${SLURM_JOB_ID:-none} end $(date --iso-8601=seconds)"
