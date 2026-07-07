#!/bin/bash
#SBATCH --job-name=train-asap-lora
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=1
#SBATCH --mem=128G
#SBATCH --time=infinite
#SBATCH --output=logs/train_lora_%j.out
#SBATCH --error=logs/train_lora_%j.out

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

# Single-GPU LoRA fine-tune. Metrics stream to wandb project "anticipation-asap".
# Heatmaps render via matplotlib: PNGs under run_lora/heatmaps/ and on the dashboard.
python train_lora.py \
    --data_file data/train_normalized.txt \
    --val_file data/test_normalized.txt \
    --output_dir ./run_lora \
    --wandb_run_name run_lora \
    --wandb_mode online \
    --lora_r 16 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --lora_target_modules c_attn,c_proj

echo "End: $(date)"
