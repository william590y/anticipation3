#!/bin/bash
#SBATCH --job-name=finish-paper-viz
#SBATCH --partition=ellis,thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/finish_paper_viz_%j.out
#SBATCH --error=logs/finish_paper_viz_%j.out

# Resume of build_paper_viz.sbatch after job 845859 FAILED at its last two steps.
# Our rollouts (both models, 24 windows -- the 2.4h part) and paper 2's are already
# merged into visualizer/data.js, so this only redoes what did not survive:
#   * paper 1's rollouts: it never ran, because loading paper 2 first left
#     `models.embedding` cached from paper 2's repo (fixed in run_paper_models.py);
#   * F1: the run used the old single-table report, which pooled our four rollout
#     variants and averaged each model over a different window set;
#   * the MusicXML sidecar: it died on the final window's inexpressible durations
#     (now refused per-window instead of aborting the run).
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
echo "Host: $(hostname)"; echo "Start: $(date)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

cp visualizer/data.js visualizer/data.js.bak

echo "== [1/3] paper 1 rollouts =="
conda activate paperpipe
python -u visualizer/run_paper_models.py \
    --data visualizer/data.js --manifest visualizer/paper_windows.json \
    --only paper1 --debug \
  || echo "WARNING: paper-1 step failed; continuing"
conda activate base

echo "== [2/3] note-level F1 =="
python visualizer/compute_f1.py --data visualizer/data.js

echo "== [3/3] real-XML sheet-music sidecar =="
python visualizer/precompute_scores_xml.py --data visualizer/data.js \
    --output visualizer/scores_xml.js

echo "Done. End: $(date)"
