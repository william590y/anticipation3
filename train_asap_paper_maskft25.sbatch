#!/bin/bash
#SBATCH --job-name=maskft25
#SBATCH --partition=thickstun
#SBATCH --account=thickstun
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/wjl86/anticipation3/logs/maskft25_%j.out
#
# CONTEXT-DROPOUT FINE-TUNE (2026-08-27). The whole-song diagnosis: the model
# has never seen imperfect score context (--mask_prob existed, class default
# 0.75, but NO run ever passed it), so its huge clean-context skill (42-68%
# tol1 continuing from a GT overlap) collapses to ~8% on its own context.
# Beyer & Dai train with 75% context dropout for exactly this
# (arXiv:2410.00210). This fine-tunes run_paper_split_v2/checkpoint-2500 with
# mask_prob 0.75, otherwise the original paper-split recipe.
#
# 2x RTX 6000 Ada (48GB) on thickstun.
# batch 16 x accum 2 x 2 ranks = effective 64 (batch 32 OOMs at 48GB) (the recipe invariant;
# 64=2^6 so 4 ranks preserve it where 3 cannot). NCCL_P2P_DISABLE=1 is
# MANDATORY on ellis (broken P2P on ellis-compute-02 hangs the first
# collective).
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
python -c "import torch;assert torch.cuda.device_count()==2;[torch.zeros(8,device=f'cuda:{i}',dtype=torch.bfloat16)@torch.zeros(8,8,device=f'cuda:{i}',dtype=torch.bfloat16) for i in range(2)];print('4x bf16 preflight OK')"
python -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --model_name run_paper_split_v2/checkpoint-2500 \
    --output_dir ./run_paper_split_v2_maskft25 \
    --wandb_run_name run_paper_split_v2_maskft25 \
    --wandb_mode online \
    --mask_prob 0.25 \
    --max_steps 10000 \
    --batch_size 16 \
    --val_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --save_steps 500 \
    --eval_steps 250
echo MASKFT_DONE
