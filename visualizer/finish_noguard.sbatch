#!/bin/bash
#SBATCH --job-name=viz-noguard
#SBATCH --partition=ellis,thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=4:00:00
#SBATCH --output=logs/viz_noguard_%j.out
#SBATCH --error=logs/viz_noguard_%j.out

# Final visualizer build: no span guard, quarters->bins the only scaling applied.
# Runs BOTH models in one process on purpose -- that is the configuration where the
# repos' identically-named modules collide, and the sys.path isolation fix (paper 1's
# `dataset/` is a namespace package, which Python ranks below paper 2's dataset.py)
# has only been verified in isolation so far.
# Everything here needs a compute node: loading the 204MB data.js gets SIGKILLed by
# the login node's memory cap.
set -uo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
echo "Host: $(hostname)"; echo "Start: $(date)"

cp visualizer/data.js visualizer/data.js.bak_noguard

conda activate paperpipe
echo "== [1/3] both paper models, no span guard =="
python -u visualizer/run_paper_models.py --data visualizer/data.js \
    --manifest visualizer/paper_windows.json
conda activate base

echo "== [2/3] rebin (also recovers anything an earlier guarded run withheld) =="
python -u visualizer/rebin_paper_rollouts.py --data visualizer/data.js

echo "== [3/3] F1, all windows =="
python visualizer/compute_f1.py --data visualizer/data.js
echo "Done. End: $(date)"
