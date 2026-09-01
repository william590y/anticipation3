#!/bin/bash
#SBATCH --job-name=train-asap-paper
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=infinite
#SBATCH --output=logs/train_paper_%j.out
#SBATCH --error=logs/train_paper_%j.out

set -euo pipefail

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

echo "Host: $(hostname)"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Start: $(date)"

# Full fine-tune on the EXTERNAL ASAP split shared by Beyer&Dai (ISMIR 2024,
# MIDI2ScoreTransformer) and Zeng+ (ICLR 2026, joint-apt-epr) -- the two papers'
# split code is byte-identical (see make_paper_split.py / data/paper_split.txt).
# Trains on the paper TRAIN split, validates on the paper VALIDATION split
# (the 59-performance/14-piece TEST split is held out). 0.5s beat interval.
#
# HARDWARE-ONLY CHANGE from the original 1-GPU ellis run: moved to thickstun
# (RTX 6000 Ada 48GB, uncontended) on 2 GPUs. The optimization is deliberately
# left bit-for-bit identical:
#   * effective batch stays 64: batch_size=4 x grad_accum=8 x 2 GPUs (was
#     4 x 16 x 1). accelerate does NOT split batches, so per-rank batch stays 4
#     and train.py's global effective batch (bs x ga x num_processes) is 64.
#   * max_steps=20000 unchanged, and the LR schedule is a plain LambdaLR keyed to
#     the optimizer-step count -- it is created AFTER accelerator.prepare() and is
#     never prepared, so it is not stepped once-per-process (the usual accelerate
#     multi-GPU LR gotcha does not apply here).
#   * eval subsets (500 teacher-forced / 100 autoregressive) are both divisible by
#     2, so _resolve_eval_subset_size returns them unchanged at world_size=2.
# NOT 3 GPUs (as first considered): 64 = 2^6 is not divisible by 3, so no integer
# (batch_size, grad_accum) preserves the effective batch on 3 ranks.
#
# Uses torchrun rather than `accelerate launch` (same as train_asap_v2.sbatch).
python -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --output_dir ./run_paper_split_v2 \
    --wandb_run_name run_paper_split_v2 \
    --wandb_mode online \
    --max_steps 20000 \
    --batch_size 4 \
    --val_batch_size 4 \
    --gradient_accumulation_steps 8

echo "End: $(date)"
