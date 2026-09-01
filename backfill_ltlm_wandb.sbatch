#!/bin/bash
#SBATCH --job-name=ltlm-wandb-backfill
#SBATCH --partition=ellis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --output=logs/backfill_ltlm_wandb_%j.out
#SBATCH --error=logs/backfill_ltlm_wandb_%j.out

# Sidecar for live Gaussian LTLM planner job 81973.
# The training process will NOT reload train_ltlm.py; this job reads
# /home/wjl86/anticipation3-planner/run_ltlm_planner/checkpoint-* and logs
# AR accuracies + piano rolls onto wandb run ri7dgjs2 using checkpoint_step
# as the chart x-axis (wandb drops decreasing step= on a live run).
#
# Compute on Ellis; upload by attaching to job 81973's wandb-core:
#   srun --jobid=81973 --overlap python backfill_ltlm_wandb.py --upload_only
#
# Launch (from the git repo, not the clone):
#   sbatch backfill_ltlm_wandb.sbatch
# Override:
#   sbatch --export=ALL,WATCH=0,STEPS=2500,5000 backfill_ltlm_wandb.sbatch

set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

mkdir -p logs
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export NCCL_P2P_DISABLE=1

OUTPUT_DIR="${OUTPUT_DIR:-/home/wjl86/anticipation3-planner/run_ltlm_planner}"
VAL_FILE="${VAL_FILE:-data/val_paper.txt}"
WANDB_RUN_ID="${WANDB_RUN_ID:-ri7dgjs2}"
WANDB_DIR="${WANDB_DIR:-/home/wjl86/anticipation3/logs/wandb_ltlm_planner_backfill}"
WATCH_JOB_ID="${WATCH_JOB_ID:-81973}"
WATCH="${WATCH:-1}"
STEPS="${STEPS:-}"
EVAL_MAX_SAMPLES="${EVAL_MAX_SAMPLES:-100}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-4}"
NO_WANDB="${NO_WANDB:-1}"

echo "Host: $(hostname)"
echo "Partition: ${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Output dir: $OUTPUT_DIR"
echo "wandb run: $WANDB_RUN_ID  wandb_dir=$WANDB_DIR  watch_job=$WATCH_JOB_ID watch=$WATCH"
echo "Start: $(date)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

if [[ "$(hostname)" == *ellis-compute-02* ]]; then
  export NCCL_P2P_DISABLE=1
fi

WATCH_FLAGS=()
if [[ "$WATCH" == "1" ]]; then
  WATCH_FLAGS+=(--watch --watch_job_id "$WATCH_JOB_ID")
fi
STEP_FLAGS=()
if [[ -n "$STEPS" ]]; then
  STEP_FLAGS+=(--steps "$STEPS")
fi
NO_WANDB_FLAGS=()
if [[ "$NO_WANDB" == "1" ]]; then
  NO_WANDB_FLAGS+=(--no_wandb)
fi

python backfill_ltlm_wandb.py \
    --output_dir "$OUTPUT_DIR" \
    --val_file "$VAL_FILE" \
    --wandb_run_id "$WANDB_RUN_ID" \
    --wandb_dir "$WANDB_DIR" \
    --eval_max_samples "$EVAL_MAX_SAMPLES" \
    --val_batch_size "$VAL_BATCH_SIZE" \
    --piano_roll_examples 3 \
    "${WATCH_FLAGS[@]}" \
    "${STEP_FLAGS[@]}" \
    "${NO_WANDB_FLAGS[@]}"

echo "End: $(date)"
