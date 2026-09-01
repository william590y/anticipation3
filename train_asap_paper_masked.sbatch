#!/bin/bash
#SBATCH --job-name=train-paper-masked
#SBATCH --partition=ellis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
# The GPU TYPE is pinned, not just the count: ellis mixes a 5x3090 node (24 GB)
# with an 8xA6000 node (48 GB), and a plain `gpu:2` can land on the 3090s.
# This run exists to be hyperparameter-identical to run_paper_split_v2, so it
# cannot absorb the batch-size cut a 24 GB card would force -- it queues for
# A6000s instead.
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --mem=64G
#SBATCH --time=infinite
#SBATCH --output=logs/train_paper_masked_%j.out
#SBATCH --error=logs/train_paper_masked_%j.out

set -euo pipefail

cd /home/wjl86/anticipation3

source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# GPU peer-to-peer is broken on ellis-compute-02 -- the A6000 node this run is
# pinned to. A bare 2-rank NCCL all_reduce hangs there forever (P2P_LEVEL=SYS,
# SHM_DISABLE and IB_DISABLE do not help), and the hang is in the FIRST
# collective, so the job burns both GPUs at 100% util printing nothing. That
# reads as slow startup, not a deadlock.
export NCCL_P2P_DISABLE=1

echo "Host: $(hostname)"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Start: $(date --iso-8601=seconds)"

# The 24 GB failure mode is an OOM some way into training, not at startup, so
# check the card up front rather than discovering it an hour in.
if nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '$1 < 40000 {found=1} END {exit !found}'; then
    echo "Refusing to run: a GPU with <40 GB was allocated, which cannot hold the" >&2
    echo "run_paper_split_v2 batch. Wait for A6000s rather than changing the batch." >&2
    exit 1
fi

GPUS=$(nvidia-smi --list-gpus | wc -l)
if [[ "${GPUS}" -ne 2 ]]; then
    echo "Expected 2 GPUs (effective batch 4 x 8 x 2 = 64), got ${GPUS}." >&2
    exit 1
fi

# The masked-loss twin of run_paper_split_v2 (train_asap_paper.sbatch).
#
# EXACTLY ONE difference from that run: --loss_mask_performance_tokens, which
# sets control-triplet labels to -100 so the decoder's train AND val loss are
# scored only on the tokens it actually generates. Every other knob is the
# original's: paper-split files, max_steps 20000, batch_size 4 x grad_accum 8 x
# 2 GPUs = effective batch 64, and train.py's defaults for lr (3e-5 cosine to
# 3e-6), original_weight_l2 1e5, max_grad_norm 2.0, eval subsets 500/100, and
# the augmentation (onset/dur jitter 0.05, transpose +/-12, tempo +/-0.2).
#
# This is the baseline the masked-loss LTLM run (run_ltlm) is compared against;
# run_paper_split_v2's own val/loss averages over the controls and so is NOT
# comparable to either.
python -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --output_dir ./run_paper_split_v2_masked \
    --wandb_run_name run_paper_split_v2_masked \
    --wandb_mode online \
    --max_steps 20000 \
    --batch_size 4 \
    --val_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --loss_mask_performance_tokens

echo "End: $(date --iso-8601=seconds)"
