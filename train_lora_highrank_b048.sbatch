#!/bin/bash
#SBATCH --job-name=train-lora-r512-b048
#SBATCH --partition=ellis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=16G
#SBATCH --time=infinite
#SBATCH --output=logs/train_lora_r512_b048_%j.out
#SBATCH --error=logs/train_lora_r512_b048_%j.out

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

# run_nodummy_lora_r512 (train_lora_highrank.sbatch) used 4 GPUs: LoRA
# r=512/alpha=1024 on c_attn,c_proj, no L2 anchor, batch_size=16/GPU x grad_accum=1
# x 4 GPUs = effective batch 64, max_steps=20000. This job runs on a single GPU
# instead. First attempt used batch_size=16/grad_accum=4 and OOM'd on ellis's 48GB
# A6000; batch_size=8/grad_accum=8 then also OOM'd on a 24GB 3090 -- dropped to
# batch_size=4/grad_accum=16 (matching the full fine-tune job) to keep the same
# effective batch size (4 x 16 x 1 = 64) while further cutting activation memory.
# On ellis (not thickstun)
# since thickstun's 8 GPUs were all in use with no ETA; cpus-per-task/mem trimmed
# to 4/16G since ellis nodes have tight CPU/RAM headroom even when a GPU is free,
# and train_lora.py uses num_workers=0 so barely needs either. Trained on the
# 0.48s-beat-interval tokenization (tokenize_asap_b048.sbatch) instead of the
# default 0.5s data/train_normalized.txt.
python train_lora.py \
    --data_file data/train_normalized_b048.txt \
    --val_file data/test_normalized_b048.txt \
    --output_dir ./run_nodummy_lora_r512_b048 \
    --wandb_run_name run_nodummy_lora_r512_b048 \
    --wandb_mode online \
    --max_steps 20000 \
    --original_weight_l2 0 \
    --lora_r 512 \
    --lora_alpha 1024 \
    --lora_dropout 0.05 \
    --lora_target_modules c_attn,c_proj \
    --batch_size 4 \
    --val_batch_size 4 \
    --gradient_accumulation_steps 16

echo "End: $(date)"
