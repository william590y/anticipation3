#!/bin/bash
#SBATCH -J scalemax-f1
#SBATCH -p thickstun
#SBATCH -A thickstun
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o /home/wjl86/anticipation3/logs/scalemax_%j.out
# CPU-only: no --gres on purpose -- the matcher is pure Python and the GPUs
# there are busy; 32 cores x multiprocessing covers 1180 windows x 3 models
# x 13 scales x 2 metric families.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1
python3 -u nbest/eval_scalemax_f1.py --procs 32
echo SCALEMAX_OK
