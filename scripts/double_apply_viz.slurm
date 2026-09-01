#!/bin/bash
#SBATCH -p thickstun
#SBATCH -A thickstun
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH -c 4
#SBATCH -t 1:30:00
#SBATCH --array=0-1
#SBATCH -J dapply-viz
#SBATCH -o logs/dapply_viz_%A_%a.out

# Double-application experiment: greedy decode, feed the generated score back
# in as the performance stream, decode again; F1 vs GT on the viz windows.
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1

T=$SLURM_ARRAY_TASK_ID
python visualizer/double_apply_viz.py \
  --variant raw \
  --checkpoint run_paper_split_v2/checkpoint-2500 \
  --shard-index "$T" --num-shards 2 \
  --output "visualizer/double_apply_shards/shard_${T}.json"
echo DAPPLY_TASK_OK
