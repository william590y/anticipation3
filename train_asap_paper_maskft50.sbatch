#!/bin/bash
#SBATCH --job-name=maskft50
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/wjl86/anticipation3/logs/maskft50_%j.out
#
# CONTEXT-DROPOUT FINE-TUNE (2026-08-27). The whole-song diagnosis: the model
# has never seen imperfect score context (--mask_prob existed, class default
# 0.75, but NO run ever passed it), so its huge clean-context skill (42-68%
# tol1 continuing from a GT overlap) collapses to ~8% on its own context.
# Beyer & Dai train with 75% context dropout for exactly this
# (arXiv:2410.00210). This fine-tunes run_paper_split_v2/checkpoint-2500 with
# mask_prob 0.75, otherwise the original paper-split recipe.
#
# 2x RTX 3090 (24GB) on ellis-compute-01 (thickstun was queued >1 day out).
# batch 8 x accum 4 x 2 ranks = effective 64 (the recipe invariant;
# 64=2^6 so 4 ranks preserve it where 3 cannot). NCCL_P2P_DISABLE=1 is
# MANDATORY on ellis (broken P2P on ellis-compute-02 hangs the first
# collective).
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
python -c "import torch;assert torch.cuda.device_count()==2;assert all('3090' in torch.cuda.get_device_name(i) for i in range(2)), torch.cuda.get_device_name(0);[torch.zeros(8,device=f'cuda:{i}',dtype=torch.bfloat16)@torch.zeros(8,8,device=f'cuda:{i}',dtype=torch.bfloat16) for i in range(2)];print('4x bf16 preflight OK')"
python -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --model_name run_paper_split_v2/checkpoint-2500 \
    --output_dir ./run_paper_split_v2_maskft50 \
    --wandb_run_name run_paper_split_v2_maskft50 \
    --wandb_mode online \
    --mask_prob 0.50 \
    --max_steps 10000 \
    --batch_size 8 \
    --val_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --save_steps 500 \
    --eval_steps 250
echo MASKFT_DONE
