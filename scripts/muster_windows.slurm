#!/bin/bash
#SBATCH -J muster-win
#SBATCH -p thickstun
#SBATCH -A thickstun
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 3:00:00
#SBATCH -o /home/wjl86/anticipation3/logs/muster_win_%j.out
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1
python3 -u nbest/eval_muster_windows.py --procs 32
echo MUSTERWIN_OK
