#!/bin/bash
#SBATCH --job-name=bench-diffdraft
#SBATCH --partition=default_partition
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=logs/bench_diffdraft_%j.out
#SBATCH --error=logs/bench_diffdraft_%j.out
# Diffusion-drafter correctness gate + speed table.
#
# A6000 rather than one of the ellis 3090s for two reasons: the other agents own
# the 3090s for their own timing runs (a co-tenant would poison both sets of
# numbers), and the target's fp32 KV cache is ~19 GB at batch 96, which simply
# does not fit in 24 GB -- the baseline column has to be measurable at every
# batch size in the table, in the same process, on the same card.
#
# Arguments go on the sbatch command line, NOT through --export (SLURM splits an
# --export value on commas; see bench/bench_decode.sbatch for the run that cost).
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python bench/bench_diffdraft.py "$@"
