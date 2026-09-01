#!/bin/bash
#SBATCH --job-name=viz-ppo-publish
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/viz_ppo_publish_%j.out
#SBATCH --error=logs/viz_ppo_publish_%j.out

set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

: "${PPO_MANIFEST:?PPO_MANIFEST is required}"
: "${PPO_SHARD_DIR:?PPO_SHARD_DIR is required}"
PPO_DATA_FILE="${PPO_DATA_FILE:-/home/wjl86/anticipation3/visualizer/data.js}"

python -u visualizer/publish_ppo_rollouts.py \
    --data "${PPO_DATA_FILE}" \
    --html /home/wjl86/anticipation3/visualizer/visualizer.html \
    --manifest "${PPO_MANIFEST}" \
    --shard-dir "${PPO_SHARD_DIR}" \
    --lock "${PPO_DATA_FILE}.lock"
