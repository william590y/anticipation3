#!/bin/bash
# CPU post-process for the 0.48s-beat-interval viz dataset. Run AFTER the GPU job
# visualizer/precompute_b048.sbatch writes visualizer/data_b048_raw.js.
#
# Pipeline:
#   1. compute_muster.py  -> adds MUSTER scores (beat_seconds=0.48) to the raw file
#      in place (still keyed by b048 line index).
#   2. precompute_scores_xml.py -> real ASAP sheet-music sidecar for the b048 grid
#      (48 bins/beat, b048 cache), written keyed by b048 line index.
#   3. make_b048_dataset.py -> re-keys BOTH the data and the scores sidecar from the
#      b048 line indices to the matching 0.5s window ids (window_map_b048.json), so
#      the "beat grid" dropdown in visualizer.html can pair them 1:1 with the 0.5s
#      windows. Produces the final data_b048.js + scores_xml_b048.js.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

RAW=visualizer/data_b048_raw.js
RAW_SCORES=visualizer/scores_xml_b048_raw.js

echo "[1/3] MUSTER (beat_seconds=0.48)..."
python visualizer/compute_muster.py --data "$RAW" --beat-seconds 0.48

echo "[2/3] real-XML sidecar (48 bins/beat, b048 cache)..."
python visualizer/precompute_scores_xml.py \
    --data "$RAW" \
    --cache-dir data/asap_aligned_stream_cache_b048 \
    --bins-per-beat 48 \
    --output "$RAW_SCORES"

echo "[3/3] re-key data + scores to the 0.5s window ids..."
python visualizer/make_b048_dataset.py \
    --raw "$RAW" \
    --ref visualizer/data.js \
    --window-map visualizer/window_map_b048.json \
    --raw-scores "$RAW_SCORES" \
    --beat-seconds 0.48 \
    --output visualizer/data_b048.js \
    --output-scores visualizer/scores_xml_b048.js

echo "Done. visualizer.html now offers the 0.48 s/beat option."
