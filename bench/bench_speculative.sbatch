#!/bin/bash
#SBATCH --job-name=bench-spec
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --nodelist=zabih-compute-01
#SBATCH --mem=64G
#SBATCH --time=8:00:00
#SBATCH --output=logs/bench_spec_%j.out
#SBATCH --error=logs/bench_spec_%j.out
# Speculative-decoding benchmark. Same node as train_draft.sbatch on purpose:
# every number in the comparison table has to come from one GPU, and a
# launch-bound decode is extremely sensitive to who else is on the card.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
python -m bench.bench_speculative ${BENCH_ARGS:-}
