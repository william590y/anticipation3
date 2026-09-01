#!/bin/bash
#SBATCH --job-name=train-paper-masked-40k
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:4
#SBATCH --mem=128G
#SBATCH --time=infinite
#SBATCH --output=logs/train_paper_masked_40k_%j.out
#SBATCH --error=logs/train_paper_masked_40k_%j.out

set -euo pipefail

# Run from the working tree (not a git clone) so this job picks up the
# compile / SDPA / fused-Adam / once-per-step L2 changes in train.py.
cd /home/wjl86/anticipation3

source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600
# Node-local inductor/triton cache: NFS autotune dirs hang on exit and slow compile.
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/torchinductor-${USER}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton-${USER}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR" logs

echo "Host: $(hostname)"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Start: $(date --iso-8601=seconds)"

GPUS=$(nvidia-smi --list-gpus | wc -l)
if [[ "${GPUS}" -ne 4 ]]; then
    echo "Expected 4 GPUs (effective batch 4 x 4 x 4 = 64), got ${GPUS}." >&2
    exit 1
fi

# 40k-step continuation of the masked paper-split FT, same algorithm as
# train_asap_paper_masked.sbatch:
#   * --loss_mask_performance_tokens
#   * paper-split train/val files
#   * effective batch 64: batch_size=4 x grad_accum=4 x 4 GPUs
#     (was 4 x 8 x 2). Do NOT keep accum=8; that would double the batch.
#   * lr cosine 3e-5 -> 3e-6 over 40000 optimizer steps, original_weight_l2 1e5
# New output_dir so we do not overwrite run_paper_split_v2_masked.
# Kernel-only speedups (compile / SDPA / fused Adam / TF32 / last-microbatch L2)
# are train.py defaults; they do not change the optimizer or loss.
python -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --output_dir ./run_paper_split_v2_masked_40k \
    --wandb_run_name run_paper_split_v2_masked_40k \
    --wandb_mode online \
    --max_steps 40000 \
    --batch_size 4 \
    --val_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --loss_mask_performance_tokens \
    --compile \
    --compile_mode default \
    --attn_implementation sdpa

echo "End: $(date --iso-8601=seconds)"
