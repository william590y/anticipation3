#!/bin/bash
#SBATCH --job-name=tokenize-asap
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=100
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=logs/tokenize_%j.out
#SBATCH --error=logs/tokenize_%j.out

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

# Retokenize with the no-dummy interleaving (unmatched perf notes dropped) and
# the 90/10 split. Reuses data/asap_aligned_stream_cache when fingerprints match.
python tokenize-asap-sliding.py --workers "${SLURM_CPUS_PER_TASK:-100}"

echo "End: $(date)"
