#!/bin/bash
#SBATCH --job-name=model-dur-hist
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH --mem=16G
#SBATCH --time=03:00:00
#SBATCH --output=logs/model_dur_hist_%j.out
#SBATCH --error=logs/model_dur_hist_%j.out

# Histogram of the trained model's predicted score-note durations, sampled from
# full autoregressive rollouts over the validation set. Compare against
# analysis/duration_hist_dataset.png (ground truth) to see whether the model's
# output durations cluster on the same note-value gridlines.
#
# Override via --export, e.g.:
#   sbatch --export=ALL,CHECKPOINT=run_nodummy_lora_r512/final analysis/model_duration_hist.sbatch
set -euo pipefail

cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

CHECKPOINT="${CHECKPOINT:-run_nodummy/final}"
TEST_FILE="${TEST_FILE:-data/test_normalized.txt}"
NUM_EXAMPLES="${NUM_EXAMPLES:-300}"
OUTPUT="${OUTPUT:-analysis/duration_hist_model}"

echo "Host: $(hostname)"
echo "Checkpoint: $CHECKPOINT  Test file: $TEST_FILE  Windows: $NUM_EXAMPLES"
echo "Start: $(date)"

python analysis/duration_histogram.py model \
    --checkpoint "$CHECKPOINT" \
    --test-file "$TEST_FILE" \
    --num-examples "$NUM_EXAMPLES" \
    --output "$OUTPUT"

echo "End: $(date)"
