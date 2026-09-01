#!/bin/bash
#SBATCH --job-name=viz-paper-publish
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=visualizer/logs/viz_paper_publish_%j.out
#SBATCH --error=visualizer/logs/viz_paper_publish_%j.out

set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export PYTHONUNBUFFERED=1

: "${PAPER_SHARD_DIR:?PAPER_SHARD_DIR is required}"
PAPER_DATA_FILE="${PAPER_DATA_FILE:-/home/wjl86/anticipation3/visualizer/data.js}"

python -u visualizer/publish_paper_seed_rollouts.py \
  --data "${PAPER_DATA_FILE}" \
  --shards "${PAPER_SHARD_DIR}" \
  --html /home/wjl86/anticipation3/visualizer/visualizer.html \
  --lock "${PAPER_DATA_FILE}.lock"
