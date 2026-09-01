#!/bin/bash
#SBATCH --job-name=m40k-viz-ppl
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=2:00:00
#SBATCH --array=0-1
#SBATCH --output=logs/m40k_viz_ppl_%A_%a.out
#SBATCH --error=logs/m40k_viz_ppl_%A_%a.out
# GT vs generated interleaved-sequence PPL under run_paper_split_v2_masked_40k/
# checkpoint-40000 on the 24 viz windows; 2 GPUs = one split per array task.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
SPLITS=(validation test)
SPLIT="${SPLITS[$SLURM_ARRAY_TASK_ID]}"
python eval_masked40k_viz_ppl.py --split "$SPLIT" \
  --output "masked40k_viz_ppl/${SPLIT}.json"
