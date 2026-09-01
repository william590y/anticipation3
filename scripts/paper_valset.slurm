#!/bin/bash
#SBATCH -J paperval
#SBATCH -p thickstun,ellis
#SBATCH -A thickstun
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|6000ada|a100|a40|3090"
#SBATCH -c 8
#SBATCH --mem=96G
#SBATCH -t 12:00:00
#SBATCH -o /home/wjl86/anticipation3/logs/paperval_%j.out
# Both papers sliced onto the stride-150 VAL windows (mirrors paper_testset.slurm).
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate paperpipe
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;n=torch.cuda.get_device_name(0);print(f'{n} {t:.1f}GiB');assert t>=10,f'refusing {n}'"
python3 -u visualizer/run_paper_models.py --data visualizer/data_valset.js
conda activate base
python3 -u visualizer/compute_f1.py --data visualizer/data_valset.js
echo PAPERVAL_OK
