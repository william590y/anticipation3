#!/bin/bash
#SBATCH --job-name=dataset-dur-hist
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/dataset_dur_hist_%j.out
#SBATCH --error=logs/dataset_dur_hist_%j.out

# Histogram of ground-truth score-note durations over the tokenized corpus.
# Chained after tokenization; also serves as a check that the fractional-duration
# fix removed the double-quantization artifacts (e.g. the old bin-23 peak).
set -euo pipefail

cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

echo "Host: $(hostname)"
echo "Start: $(date)"

python analysis/duration_histogram.py dataset \
    --files data/train_normalized.txt data/test_normalized.txt \
    --output analysis/duration_hist_dataset

echo "End: $(date)"
