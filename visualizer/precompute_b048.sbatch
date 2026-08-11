#!/bin/bash
#SBATCH --job-name=viz-precompute-b048
#SBATCH --partition=ellis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=12:00:00
#SBATCH --output=logs/viz_precompute_b048_%j.out
#SBATCH --error=logs/viz_precompute_b048_%j.out

set -euo pipefail

cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export PYTHONUNBUFFERED=1

echo "Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Start: $(date)"

# Recompute the visualizer rollouts for the 0.48s-beat-interval tokenization,
# using the highest-pitch-accuracy checkpoints from each b048 run (checkpoint-20000
# for both the full fine-tune and the r=512 LoRA -- both are the top saved
# checkpoint by training-time AR pitch accuracy: 92.06% and 95.04% respectively).
#
#   --test-file / --cache-dir : the 0.48s tokenization + its DEDICATED aligned-stream
#       cache (control triplets are beat-independent, but the cache must match the
#       test file so locate_window can recover the raw streams).
#   --only-indices            : the 24 b048 line indices that reproduce EXACTLY the
#       same 24 pieces/windows shown in the 0.5s viz (visualizer/data.js), matched by
#       beat-independent control-token signature (order mirrors data.js example_order).
#       Output examples are keyed by these b048 line indices; a post-process step
#       re-keys them to the 0.5s window keys and stamps beat_seconds=0.48.
#
# Writes data_b048_raw.js (keyed by b048 line index); make_b048_dataset.py then
# produces the final visualizer/data_b048.js.
python visualizer/precompute_visualizer.py \
    --checkpoint run_nodummy_v2_b048/checkpoint-20000 \
    --lora-checkpoint run_nodummy_lora_r512_b048/checkpoint-20000 \
    --test-file data/test_normalized_b048.txt \
    --cache-dir data/asap_aligned_stream_cache_b048 \
    --only-indices 7,16175,75731,104067,104617,247800,272392,302362,1026,9,16181,28937,53862,40266,47945,75741,70234,104074,104625,247807,165834,272400,279148,302371 \
    --output visualizer/data_b048_raw.js

echo "End: $(date)"
