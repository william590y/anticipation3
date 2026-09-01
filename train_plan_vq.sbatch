#!/bin/bash
#SBATCH --job-name=plan-vq
#SBATCH --partition=ellis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
# The GPU TYPE is pinned, not just the count: ellis mixes a 5x3090 node with an
# 8xA6000 node, so a plain `gpu:4` can land on 24 GB cards. Override both this
# and the partition together (submit_plan_chain.sh passes --gres).
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --output=logs/plan_vq_%j.out
#SBATCH --error=logs/plan_vq_%j.out

set -euo pipefail

cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# GPU peer-to-peer is broken on ellis-compute-02: a bare 2-rank NCCL all_reduce
# hangs there forever (P2P_LEVEL=SYS, SHM_DISABLE and IB_DISABLE do not help).
# The hang is in the FIRST collective, so accelerate's opening
# wait_for_everyone() never returns and the job burns 4 GPUs at 100% util
# printing nothing -- it looks like slow startup, not a deadlock.
export NCCL_P2P_DISABLE=1

RUN_DIR="${RUN_DIR:-./run_plan_vq}"
NUM_CODES="${NUM_CODES:-8}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-512}"
MAX_STEPS="${MAX_STEPS:-20000}"

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID}; start $(date --iso-8601=seconds)"
echo "run_dir=${RUN_DIR} num_codes=${NUM_CODES} codebook_size=${CODEBOOK_SIZE}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# --nproc_per_node below is hardcoded to 4, so a --gres override that changes the
# count would silently leave GPUs idle (or oversubscribe them).
GPUS=$(nvidia-smi --list-gpus | wc -l)
if [[ "${GPUS}" -ne 4 ]]; then
    echo "Expected 4 GPUs (--nproc_per_node=4), got ${GPUS}." >&2
    exit 1
fi

# Stage 1: quantize FROZEN PRETRAINED PIANOBART features into a score plan.
#
# The score encoder is PianoBART (ICME 2024), frozen; only the performance
# encoder, the cross-attention pooling queries, the codebook and the
# performance-conditioned decoder are trained.
#
# Trains on the FILTERED paper split (data/train_paper.txt), the same windows
# and the same augmentation train.py uses.
#
# Effective batch 64 (16 x 4 GPUs). Nothing downstream depends on this number
# the way stage 2's 64 does -- these are fresh weights, not a fine-tune.
python -u -m torch.distributed.run --standalone --nproc_per_node=4 \
    train_plan_vq.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --output_dir "${RUN_DIR}" \
    --wandb_run_name "plan_vq_${SLURM_JOB_ID}" \
    --wandb_mode online \
    --num_codes "${NUM_CODES}" \
    --codebook_size "${CODEBOOK_SIZE}" \
    --code_dim 32 \
    --d_model 512 \
    --n_head 8 \
    --perf_layers 2 \
    --encoder_layers 2 \
    --decoder_layers 4 \
    --batch_size 16 \
    --val_batch_size 16 \
    --learning_rate 3e-4 \
    --warmup_steps 500 \
    --max_steps "${MAX_STEPS}" \
    --eval_steps 500 \
    --save_steps 2000 \
    --eval_samples 512 \
    --piano_roll_examples 4 \
    --num_workers 6

echo "Job ${SLURM_JOB_ID} end $(date --iso-8601=seconds)"
