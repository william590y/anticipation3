#!/bin/bash
#SBATCH --job-name=train-lora-r512-paper
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=64G
#SBATCH --time=infinite
#SBATCH --output=logs/train_lora_r512_paper_%j.out
#SBATCH --error=logs/train_lora_r512_paper_%j.out

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

# LoRA (r=512/alpha=1024 on c_attn,c_proj, no L2 anchor) on the EXTERNAL ASAP
# split shared by Beyer&Dai (ISMIR 2024, MIDI2ScoreTransformer) and Zeng+
# (ICLR 2026, joint-apt-epr) -- the two papers' split code is byte-identical
# (see make_paper_split.py / data/paper_split.txt). Trains on the paper TRAIN
# split, validates on the paper VALIDATION split (TEST is held out). 0.5s beat.
#
# HARDWARE-ONLY CHANGE from the original 1-GPU ellis run: moved to thickstun
# (RTX 6000 Ada 48GB, uncontended) on 2 GPUs, with the optimization left
# bit-for-bit identical -- effective batch stays 64 (batch_size=4 x grad_accum=8
# x 2 GPUs, was 4 x 16 x 1), max_steps=20000 unchanged, and all LoRA
# hyperparameters below are untouched. See train_asap_paper.sbatch's comment for
# why the LR schedule and eval subset sizes are unaffected by world_size=2, and
# why this is 2 GPUs and not 3 (64 = 2^6 is not divisible by 3).
#
# Uses torchrun rather than `accelerate launch` (same as train_lora_highrank.sbatch).
python -m torch.distributed.run --standalone --nproc_per_node=2 train_lora.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --output_dir ./run_paper_split_lora_r512 \
    --wandb_run_name run_paper_split_lora_r512 \
    --wandb_mode online \
    --max_steps 20000 \
    --original_weight_l2 0 \
    --lora_r 512 \
    --lora_alpha 1024 \
    --lora_dropout 0.05 \
    --lora_target_modules c_attn,c_proj \
    --batch_size 4 \
    --val_batch_size 4 \
    --gradient_accumulation_steps 8

echo "End: $(date)"
