#!/bin/bash
#SBATCH --job-name=viz-precompute
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=logs/viz_precompute_%j.out
#SBATCH --error=logs/viz_precompute_%j.out

# Precompute visualizer data. Override via --export, e.g.:
#   sbatch --export=ALL,CHECKPOINT=run_nodummy/final,TEST_FILE=data/test_normalized.txt \
#          visualizer/precompute.sbatch
set -euo pipefail

cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

CHECKPOINT="${CHECKPOINT:-run_nodummy/final}"
TEST_FILE="${TEST_FILE:-data/test_normalized.txt}"
OUTPUT="${OUTPUT:-visualizer/data.js}"
NUM_EXAMPLES="${NUM_EXAMPLES:-24}"
EXTRA_FLAGS="${EXTRA_FLAGS:-}"

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-32}"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
echo "Checkpoint: $CHECKPOINT  Test file: $TEST_FILE  Output: $OUTPUT"
echo "Start: $(date)"

python visualizer/precompute_visualizer.py \
    --checkpoint "$CHECKPOINT" \
    --test-file "$TEST_FILE" \
    --num-examples "$NUM_EXAMPLES" \
    --output "$OUTPUT" \
    $EXTRA_FLAGS

echo "End: $(date)"
