#!/bin/bash
#SBATCH --job-name=nbest-gen-unf
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --array=0-3%4
#SBATCH --output=logs/nbest_gen_unf_%A_%a.out
#SBATCH --error=logs/nbest_gen_unf_%A_%a.out
# Full-consistency N-best generation on the UNFILTERED (raw performance)
# paper-split windows, under the lowest-val-loss non-LoRA FT checkpoint
# (run_paper_split_v2/checkpoint-2500). Tasks 0-2 = train shards (stride 3
# over ~150k windows -> ~50k), task 3 = val (stride 3 over ~8k -> ~2.7k).
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
T="$SLURM_ARRAY_TASK_ID"
CKPT=run_paper_split_v2/checkpoint-2500
if [ "$T" -lt 3 ]; then
  python -m nbest.generate_nbest --token-file data/train_paper_unfiltered.txt \
    --stride 3 --shard-index "$T" --num-shards 3 --n-sampled 8 \
    --checkpoint "$CKPT" --window-batch 12 --score-chunk 8 \
    --output "nbest_data/unf_train_shard0${T}.pt"
else
  python -m nbest.generate_nbest --token-file data/val_paper_unfiltered.txt \
    --stride 3 --shard-index 0 --num-shards 1 --n-sampled 8 \
    --checkpoint "$CKPT" --window-batch 12 --score-chunk 8 \
    --output nbest_data/unf_val_shard00.pt
fi
