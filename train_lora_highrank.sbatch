#!/bin/bash
#SBATCH --job-name=train-lora-r512
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:4
#SBATCH --mem=32G
#SBATCH --time=infinite
#SBATCH --output=logs/train_lora_r512_%j.out
#SBATCH --error=logs/train_lora_r512_%j.out

set -euo pipefail

# Run from the repo so the local `anticipation` package and `data/` paths resolve.
cd /home/wjl86/anticipation3

source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

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

# Same run as train_asap.sbatch (no-dummy data, 20k steps) but LoRA at an
# extremely high rank (r=512 on a 1024-dim model) and no L2 anchoring.
#
# Same 4x A100-80GB / torchrun setup as train_asap.sbatch (see its comments for
# why torchrun instead of `accelerate launch`). Effective batch 64, no grad
# accum: batch_size=16/GPU x grad_accum=1 x 4 GPUs. LoRA freezes the base
# weights (only adapter/embedding params get optimizer state), so this has
# more headroom than the full fine-tune at the same batch size.
python -m torch.distributed.run --standalone --nproc_per_node=4 train_lora.py \
    --data_file data/train_normalized.txt \
    --val_file data/test_normalized.txt \
    --output_dir ./run_nodummy_lora_r512 \
    --wandb_run_name run_nodummy_lora_r512 \
    --wandb_mode online \
    --max_steps 20000 \
    --original_weight_l2 0 \
    --lora_r 512 \
    --lora_alpha 1024 \
    --lora_dropout 0.05 \
    --lora_target_modules c_attn,c_proj \
    --batch_size 16 \
    --val_batch_size 16 \
    --gradient_accumulation_steps 1

echo "End: $(date)"
