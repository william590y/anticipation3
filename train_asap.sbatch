#!/bin/bash
#SBATCH --job-name=train-asap
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4
#SBATCH --mem=32G
#SBATCH --time=infinite
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.out

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

# 4-way data-parallel fine-tune on the no-dummy retokenized data, 4x A100-80GB
# (SXM4/NVLink) on nlplarge-compute-01. Metrics stream to wandb "anticipation-asap".
#
# Launched via torchrun rather than `accelerate launch`: this box has a stale
# ~/.cache/huggingface/accelerate/default_config.yaml pinned to DeepSpeed ZeRO-2
# with its own gradient_accumulation_steps=16, which `accelerate launch` would
# silently apply instead of our settings. train.py's Accelerator() auto-detects
# torchrun's env vars (RANK/WORLD_SIZE/LOCAL_RANK) directly, bypassing that config.
#
# Effective batch 64, no gradient accumulation: 4 GPUs x batch_size=16 x
# grad_accum=1. This is the fastest arrangement for effective=64 given 4 free
# GPUs -- every optimizer step is exactly one synchronized forward/backward per
# GPU, no sequential accumulation rounds. batch_size=16/GPU has ~2x headroom
# below the calibrated-safe reference (batch=8 measured ~24GB peak on a 24GB
# card; 16 extrapolates to ~46GB, well under this card's 80GB).
python -m torch.distributed.run --standalone --nproc_per_node=4 train.py \
    --data_file data/train_normalized.txt \
    --val_file data/test_normalized.txt \
    --output_dir ./run_nodummy \
    --wandb_run_name run_nodummy \
    --wandb_mode online \
    --max_steps 20000 \
    --batch_size 16 \
    --val_batch_size 16 \
    --gradient_accumulation_steps 1

echo "End: $(date)"
