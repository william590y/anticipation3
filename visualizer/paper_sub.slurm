#!/bin/bash
#SBATCH -J papersub
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -c 8
#SBATCH -t 2:00:00
#SBATCH -A thickstun
#SBATCH -o /home/wjl86/anticipation3/logs/papersub_%j.out
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate paperpipe
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python3 -u visualizer/run_paper_models.py --data visualizer/data_testset_sub.js --only paper2
conda activate base
python3 -u visualizer/compute_f1.py --data visualizer/data_testset_sub.js
echo PAPERSUB_OK
