#!/bin/bash
#SBATCH --job-name=train-asap-v2-b048
#SBATCH --partition=ellis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=infinite
#SBATCH --output=logs/train_v2_b048_%j.out
#SBATCH --error=logs/train_v2_b048_%j.out

set -euo pipefail

# Run from the repo so the local `anticipation` package and `data/` paths resolve.
cd /home/wjl86/anticipation3

# Activate the base conda environment used for interactive runs.
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

# Avoid CPU thread oversubscription from the dataloader/eval workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

echo "Host: $(hostname)"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: $(which python)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Start: $(date)"

# run_nodummy_v2 (train_asap_v2.sbatch) used 2 GPUs: batch_size=8/GPU x
# grad_accum=4 x 2 GPUs = effective batch 64, max_steps=20000. This job runs on a
# single GPU instead. First attempt used batch_size=8/grad_accum=8 and OOM'd on
# ellis's 24GB RTX 3090 (L2 anchor term holds a frozen copy of the original
# weights alongside the trained copy, so it's tighter on memory than the nominal
# batch size suggests) -- dropped to batch_size=4/grad_accum=16 to keep the same
# effective batch size (4 x 16 x 1 = 64) while halving activation memory. On ellis
# (not thickstun) since thickstun's 8 GPUs were all in use with no ETA;
# cpus-per-task/mem trimmed to 4/16G since ellis nodes have tight CPU/RAM headroom
# even when a GPU is free, and train.py uses num_workers=0 so barely needs either.
# Trained on the 0.48s-beat-interval tokenization (tokenize_asap_b048.sbatch)
# instead of the default 0.5s data/train_normalized.txt.
python train.py \
    --data_file data/train_normalized_b048.txt \
    --val_file data/test_normalized_b048.txt \
    --output_dir ./run_nodummy_v2_b048 \
    --wandb_run_name run_nodummy_v2_b048 \
    --wandb_mode online \
    --max_steps 20000 \
    --batch_size 4 \
    --val_batch_size 4 \
    --gradient_accumulation_steps 16

echo "End: $(date)"
