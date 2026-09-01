#!/bin/bash
#SBATCH --job-name=viz-vl-bmrg
#SBATCH --partition=thickstun
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz_valloss_beam_merge_%j.out
#SBATCH --error=logs/viz_valloss_beam_merge_%j.out

set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

SHARDS=(visualizer/valloss_beam_shards/shard_{00..06}.json)
for s in "${SHARDS[@]}"; do
  [[ -f "$s" ]] || { echo "MISSING $s" >&2; exit 1; }
done

# Flat token shards: {group: {variant: {"5": {pred_score}}}}
python visualizer/merge_beam_shards.py --data visualizer/data.js --shards "${SHARDS[@]}"
STAMP=$(date +%Y%m%d%H%M)
sed -i -E "s|data\\.js\\?v=[^\"]+|data.js?v=${STAMP}|" visualizer/visualizer.html || true
echo "Beam merge done: $(date)"
