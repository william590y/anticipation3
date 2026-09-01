#!/bin/bash
#SBATCH --job-name=draft-kd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=zabih-compute-01
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/draft_kd_%j.out
#SBATCH --error=logs/draft_kd_%j.out
# Constrained-KL distillation of run_paper_split_v2/checkpoint-2500 into a
# shallow (K-block, full-width) draft for speculative decoding.
#
# Pinned to zabih-compute-01 (6x idle A6000) rather than the ellis 3090s: the
# whole exercise ends in a wall-clock comparison, ellis-compute-01 is currently
# running four other jobs (a contended GPU makes a launch-bound decode timing
# meaningless), and the draft must be trained and benchmarked on ONE GPU type.
# Override with SBATCH_NODE if that node fills up -- but then re-run the whole
# benchmark there too, never mix nodes across the comparison table.
#
# Env overrides: DRAFT_LAYERS, OUTPUT_DIR, MAX_STEPS, BATCH_SIZE, EXTRA_ARGS.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch; print('torch', torch.__version__)"

DRAFT_LAYERS="${DRAFT_LAYERS:-4}"
OUTPUT_DIR="${OUTPUT_DIR:-run_draft_L${DRAFT_LAYERS}}"
MAX_STEPS="${MAX_STEPS:-6000}"
BATCH_SIZE="${BATCH_SIZE:-8}"

python train_draft.py \
  --draft_layers "$DRAFT_LAYERS" \
  --output_dir "$OUTPUT_DIR" \
  --max_steps "$MAX_STEPS" \
  --batch_size "$BATCH_SIZE" \
  ${EXTRA_ARGS:-}
