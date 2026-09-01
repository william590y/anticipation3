#!/bin/bash
#SBATCH --job-name=tokenize-asap-paper
#SBATCH --partition=ellis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/tokenize_paper_%j.out
#SBATCH --error=logs/tokenize_paper_%j.out

set -euo pipefail

# Run from the repo so the local `anticipation` package and `data/` paths resolve.
cd /home/wjl86/anticipation3

source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Host: $(hostname)"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "Python: $(which python)"
echo "Start: $(date)"

# Tokenize the ASAP corpus using the EXTERNAL train/validation/test split shared
# by Beyer&Dai (ISMIR 2024, MIDI2ScoreTransformer) and Zeng+ (ICLR 2026,
# joint-apt-epr) -- see make_paper_split.py / data/paper_split.txt. Default 0.5s
# beat interval, so the default 0.5s aligned-stream cache is reused (already fully
# populated: this is a fast cache-hit packing pass, no re-alignment). Performances
# absent from the manifest (the ~100 the papers drop) are excluded from all splits.
python tokenize-asap-sliding.py \
    --workers "${SLURM_CPUS_PER_TASK:-32}" \
    --split-input data/paper_split.txt \
    --train-output data/train_paper.txt \
    --val-output data/val_paper.txt \
    --test-output data/test_paper.txt

echo "End: $(date)"
