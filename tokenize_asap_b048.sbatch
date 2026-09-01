#!/bin/bash
#SBATCH --job-name=tokenize-asap-b048
#SBATCH --partition=default_partition
#SBATCH --nodelist=jjs533-compute-02
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=256G
#SBATCH --time=36:00:00
#SBATCH --output=logs/tokenize_b048_%j.out
#SBATCH --error=logs/tokenize_b048_%j.out

set -euo pipefail

# Run from the repo so the local `anticipation` package and `data/` paths resolve.
cd /home/wjl86/anticipation3

# Activate the base conda environment used for interactive runs.
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

# Keep nested BLAS/thread pools from oversubscribing the worker processes.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

echo "Host: $(hostname)"
echo "SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"
echo "Python: $(which python)"
echo "Start: $(date)"

# Retokenize at a 0.48s normalized score beat interval (default is 0.5s) with the
# SAME train/test split (split_by_score is a deterministic function of the fixed
# seed=42 + the always-identical set of valid ASAP pieces, so no split file needs
# to be reused/loaded -- re-running it reproduces the same membership). Writes to
# beat-interval-specific output paths and a DEDICATED aligned-stream cache dir so
# the existing data/train_normalized.txt, data/test_normalized.txt, and the 0.5s
# data/asap_aligned_stream_cache (used by the visualizer and everything else) are
# left completely untouched -- the cache filename is keyed only on the performance
# MIDI path, not the beat interval, so sharing the default cache dir across beat
# intervals would silently overwrite its entries in place (tokenize-asap-sliding.py
# now refuses to run that way; see its --cache-dir help).
python tokenize-asap-sliding.py \
    --workers "${SLURM_CPUS_PER_TASK:-128}" \
    --beat-interval 0.48 \
    --cache-dir data/asap_aligned_stream_cache_b048 \
    --train-output data/train_normalized_b048.txt \
    --test-output data/test_normalized_b048.txt \
    --split-file data/normalized_split_b048.txt

echo "End: $(date)"
