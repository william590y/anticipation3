#!/bin/bash
#SBATCH --job-name=build-paper-viz
# Both partitions are A6000 / RTX-6000-Ada boxes; deliberately NOT the B200 box.
# thickstun-compute-01 currently reports all 8 GPUs allocated, so listing ellis
# too lets the job start on whichever frees a GPU first.
#SBATCH --partition=ellis,thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=logs/build_paper_viz_%j.out
#SBATCH --error=logs/build_paper_viz_%j.out

# Rebuild the visualizer on the paper split:
#   * our rollouts from the newly-trained paper-split checkpoints (highest AR
#     pitch accuracy), over 12 validation + 12 test windows
#   * rollouts from the two external papers' released weights
#   * note-level F1 (3 criteria) for all three models, replacing MUSTER
#
# Chained with --dependency=afterok on the two training jobs, so it only starts
# once both have finished successfully.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "Host: $(hostname)"; echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

MANIFEST=visualizer/paper_windows.json

# ---- 1. choose the best saved checkpoint from each run -----------------------
FT_LOG=$(ls -t logs/train_paper_*.out | head -1)
LORA_LOG=$(ls -t logs/train_lora_r512_paper_*.out | head -1)
echo "== selecting checkpoints =="
echo "full fine-tune (from $FT_LOG):"
CKPT=$(python visualizer/pick_best_checkpoint.py --log "$FT_LOG" \
        --output-dir run_paper_split_v2 | tail -1)
echo "LoRA (from $LORA_LOG):"
LORA_CKPT=$(python visualizer/pick_best_checkpoint.py --log "$LORA_LOG" \
        --output-dir run_paper_split_lora_r512 | tail -1)
echo "  -> checkpoint      = $CKPT"
echo "  -> lora checkpoint = $LORA_CKPT"

# ---- 2. our rollouts, validation and test windows ---------------------------
VAL_IDX=$(python -c "
import json;w=json.load(open('$MANIFEST'))['windows']
print(','.join(str(x['line_index']) for x in w if x['split']=='validation'))")
TEST_IDX=$(python -c "
import json;w=json.load(open('$MANIFEST'))['windows']
print(','.join(str(x['line_index']) for x in w if x['split']=='test'))")

echo "== [1/5] our rollouts: validation windows =="
python visualizer/precompute_visualizer.py \
    --checkpoint "$CKPT" --lora-checkpoint "$LORA_CKPT" \
    --test-file data/val_paper.txt --only-indices "$VAL_IDX" \
    --output visualizer/data_val_raw.js

echo "== [2/5] our rollouts: test windows =="
python visualizer/precompute_visualizer.py \
    --checkpoint "$CKPT" --lora-checkpoint "$LORA_CKPT" \
    --test-file data/test_paper.txt --only-indices "$TEST_IDX" \
    --output visualizer/data_test_raw.js

echo "== [3/5] merge into visualizer/data.js =="
python visualizer/merge_paper_viz.py --manifest "$MANIFEST" \
    --val-raw visualizer/data_val_raw.js \
    --test-raw visualizer/data_test_raw.js \
    --output visualizer/data.js

# ---- 3. external paper models (isolated env: custom music21 fork) -----------
echo "== [4/5] external paper-model rollouts =="
conda activate paperpipe
python -u visualizer/run_paper_models.py \
    --data visualizer/data.js --manifest "$MANIFEST" \
    --paper1-ckpt external/weights/joint_apt_epr.ckpt \
    --paper2-ckpt external/weights/MIDI2ScoreTF.ckpt \
  || echo "WARNING: external paper-model step failed; continuing with our rollouts only"
conda activate base

# ---- 4. F1 (replaces MUSTER) + real-XML sidecar -----------------------------
echo "== [5/5] note-level F1 for all models =="
python visualizer/compute_f1.py --data visualizer/data.js

echo "== real-XML sheet-music sidecar =="
python visualizer/precompute_scores_xml.py --data visualizer/data.js \
    --output visualizer/scores_xml.js

echo "Done. End: $(date)"
