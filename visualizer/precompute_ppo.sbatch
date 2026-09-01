#!/bin/bash
#SBATCH --job-name=viz-ppo
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-7%8
#SBATCH --output=logs/viz_ppo_%A_%a.out
#SBATCH --error=logs/viz_ppo_%A_%a.out

set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

: "${PPO_MANIFEST:?PPO_MANIFEST is required}"
: "${PPO_SHARD_DIR:?PPO_SHARD_DIR is required}"
PPO_DATA_FILE="${PPO_DATA_FILE:-/home/wjl86/anticipation3/visualizer/data.js}"

mkdir -p "${PPO_SHARD_DIR}"
shard_name="$(printf 'shard_%02d.json' "${SLURM_ARRAY_TASK_ID}")"
python -u visualizer/precompute_ppo_rollouts.py \
    --data "${PPO_DATA_FILE}" \
    --manifest "${PPO_MANIFEST}" \
    --shard-index "${SLURM_ARRAY_TASK_ID}" \
    --num-shards 8 \
    --output "${PPO_SHARD_DIR}/${shard_name}"
