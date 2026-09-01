#!/bin/bash
#SBATCH -J muster-fs
#SBATCH -p thickstun
#SBATCH -A thickstun
#SBATCH -c 160
#SBATCH --mem=100G
#SBATCH -t 4:00:00
#SBATCH -o /home/wjl86/anticipation3/logs/muster_fs_%j.out
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1
python nbest/eval_muster_fullsong_pc.py --procs 160
echo MUSTER_PC_DONE
