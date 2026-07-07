#!/bin/bash
#SBATCH --job-name=train-asap-v2
#SBATCH --partition=ellis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:nvidia_rtx_a6000:2
#SBATCH --mem=32G
#SBATCH --time=infinite
#SBATCH --output=logs/train_v2_%j.out
#SBATCH --error=logs/train_v2_%j.out

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

# Rerun under a new output_dir/run name (run_nodummy_v2) to get a complete,
# uncontaminated checkpoint set -- the original run_nodummy/checkpoint-
# {2500,5000,7500} were overwritten by an unrelated forgotten job (741623) that
# shared the same output_dir for ~15 hours before being discovered and killed.
#
# Moved to ellis (2x RTX A6000 48GB) instead of the gpu/default_partition A100
# pool: the first attempt at this rerun (job 745057, on nlplarge-compute-01 via
# the `gpu` partition) got PREEMPTED ~28 minutes in with zero checkpoints saved
# -- that pool is priority-preemptible backfill (see slurm-partition-scheduling
# memory) and train.py has no checkpoint-resume, so a preemption there silently
# restarts from step 0. ellis is a first-class partition (not reached only via
# backfill), so it isn't subject to that risk. 2 GPUs (not 4) since that's what
# was free: batch_size=8/GPU (calibrated-safe, ~24GB peak, comfortable on a
# 48GB card) x grad_accum=4 x 2 GPUs = effective batch 64, matching the
# original run's config.
python -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
    --data_file data/train_normalized.txt \
    --val_file data/test_normalized.txt \
    --output_dir ./run_nodummy_v2 \
    --wandb_run_name run_nodummy_v2 \
    --wandb_mode online \
    --max_steps 20000 \
    --batch_size 8 \
    --val_batch_size 8 \
    --gradient_accumulation_steps 4

echo "End: $(date)"
