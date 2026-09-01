#!/bin/bash
#SBATCH -J papervalS
#SBATCH -p default_partition
#SBATCH -A thickstun
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|6000ada|a100|a40|3090|a5500|a5000|l4"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 1:30:00
#SBATCH --array=0-11
#SBATCH -o /home/wjl86/anticipation3/logs/papervalS_%A_%a.out
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate paperpipe
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;assert t>=10"
S=$(printf "%02d" "$SLURM_ARRAY_TASK_ID")
python3 -u visualizer/run_paper_models.py --data "visualizer/data_valset_shard${S}.js"
echo PAPERVALS_OK
