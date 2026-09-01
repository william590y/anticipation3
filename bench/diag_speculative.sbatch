#!/bin/bash
#SBATCH --job-name=diag-spec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_rtx_a6000:1
#SBATCH --exclude=ellis-compute-01,ellis-compute-02
#SBATCH --mem=48G
#SBATCH --time=00:30:00
#SBATCH --output=logs/diag_spec_%j.out
#SBATCH --error=logs/diag_spec_%j.out
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python -m bench.diag_speculative ${DIAG_ARGS:-}
