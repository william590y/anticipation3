#!/bin/bash
#SBATCH -J fs-eval
#SBATCH -p thickstun
#SBATCH -A thickstun
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 2:00:00
#SBATCH -o /home/wjl86/anticipation3/logs/fseval_%j.out
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1
python3 -u nbest/eval_fullsong.py --procs 32 --splits "${SPLITS:-test}" \
    --out "${OUT:-nbest_data/fullsong_eval.json}"
echo FSEVAL_OK
