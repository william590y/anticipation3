#!/bin/bash
#SBATCH --job-name=unfiltered-data
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/unfiltered_data_%j.out
#SBATCH --error=logs/unfiltered_data_%j.out

set -euo pipefail

cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

TRAIN_MAX="${TRAIN_MAX:-150000}"
TRAIN_STRIDE="${TRAIN_STRIDE:-20}"
VAL_MAX="${VAL_MAX:-8000}"
VAL_STRIDE="${VAL_STRIDE:-5}"

echo "unfiltered data build ${SLURM_JOB_ID} on $(hostname); start $(date --iso-8601=seconds)"

python -u build_unfiltered_windows.py \
    --input data/val_paper.txt \
    --output data/val_paper_unfiltered.txt \
    --stride "${VAL_STRIDE}" \
    --max-windows "${VAL_MAX}"

python -u build_unfiltered_windows.py \
    --input data/train_paper.txt \
    --output data/train_paper_unfiltered.txt \
    --stride "${TRAIN_STRIDE}" \
    --max-windows "${TRAIN_MAX}"

echo "unfiltered data build ${SLURM_JOB_ID} end $(date --iso-8601=seconds)"
wc -l data/val_paper_unfiltered.txt data/train_paper_unfiltered.txt
