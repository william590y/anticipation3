#!/bin/bash
#SBATCH --job-name=viz-paper-seeds
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-3%4
#SBATCH --output=visualizer/logs/viz_paper_seeds_%A_%a.out
#SBATCH --error=visualizer/logs/viz_paper_seeds_%A_%a.out

set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate paperpipe

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export PYTHONUNBUFFERED=1

: "${PAPER_SHARD_ROOT:?PAPER_SHARD_ROOT is required}"
PAPER_DATA_FILE="${PAPER_DATA_FILE:-/home/wjl86/anticipation3/visualizer/data.js}"
array_id="${SLURM_ARRAY_JOB_ID:?SLURM_ARRAY_JOB_ID is required}"
task_id="${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID is required}"
shard_dir="${PAPER_SHARD_ROOT}/${array_id}"
mkdir -p "${shard_dir}" visualizer/logs

python -u visualizer/precompute_paper_seed_rollouts.py \
  --data "${PAPER_DATA_FILE}" \
  --shard-index "${task_id}" \
  --num-shards 4 \
  --output "${shard_dir}/$(printf 'shard_%02d.json' "${task_id}")"
