#!/bin/bash
#SBATCH --job-name=viz-ckpt-f1m
#SBATCH --partition=thickstun
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/viz_ckpt_f1_merge_%j.out
#SBATCH --error=logs/viz_ckpt_f1_merge_%j.out

set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

SHARDS=(visualizer/ckpt_f1_shards/shard_{00..06}.json)
for s in "${SHARDS[@]}"; do
  [[ -f "$s" ]] || { echo "MISSING $s" >&2; exit 1; }
done

python visualizer/merge_ckpt_f1.py --shards "${SHARDS[@]}" --output visualizer/ckpt_f1_compare.json
echo "Done: $(date)"
