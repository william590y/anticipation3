#!/bin/bash
#SBATCH --job-name=dd-bench-ellis
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/bench_diffdraft_%j.out
#SBATCH --error=logs/bench_diffdraft_%j.out
# Same benchmark as bench/bench_diffdraft.sbatch, on ellis instead of
# default_partition. The A6000 runs on default_partition were PREEMPTED twice
# (jobs 464486/464487 mid-training, 467752 mid-speed-table), which is fatal for a
# benchmark: half a speed table is worse than none, because the batch sizes are
# measured in one process against one baseline and a partial table invites
# comparing numbers that were never measured together.
#
# A 3090 is 24 GB, so the fp32 target's KV cache caps the batch sweep lower than
# the A6000 did -- the harness catches OutOfMemoryError per row and prints "OOM"
# rather than losing the whole table.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python bench/bench_diffdraft.py "$@"
