#!/bin/bash
#SBATCH -J pf-viz24
#SBATCH -p ellis,default_partition
#SBATCH -A thickstun
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|6000ada|a100|a40|3090|a5500|a5000|l4"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow
#SBATCH -c 2
#SBATCH --mem=24G
#SBATCH -t 0:30:00
#SBATCH -o /home/wjl86/anticipation3/logs/pfviz24_%j.out
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;assert t>=10"
python3 -u visualizer/precompute_pitch_forced.py
echo PFVIZ24_OK
