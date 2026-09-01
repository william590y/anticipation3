#!/bin/bash
#SBATCH --job-name=diffdraft-ellis
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=6
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --mem=48G
#SBATCH --time=8:00:00
#SBATCH --output=logs/diffdraft_train_%j.out
#SBATCH --error=logs/diffdraft_train_%j.out
# Same trainer as train_diffdraft.sbatch, on the ellis partition instead.
#
# The default_partition A6000 runs (jobs 464486/464487) were PREEMPTED at 44
# minutes -- default_partition is preemptible and a 14k-step run is a big enough
# target to lose. ellis is this group's own allocation, so it is not, and
# ellis-compute-01 freed four of its five 3090s when the other agents' bench jobs
# drained. A 3090 is ~2x slower than the A6000 and has 24 GB rather than 48, so
# the batch is smaller here; `--max-loss-tokens` already bounds the head+CE
# transient, which is what actually decides whether this fits.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python train_diffdraft.py "$@"
