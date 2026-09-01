#!/bin/bash
#SBATCH --job-name=mft-muon
#SBATCH --partition=thickstun
#SBATCH --account=thickstun
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/home/wjl86/anticipation3/logs/maskft_muon_%j.out
#
# MUON + INFINITE-COMPUTE-REGULARIZATION ARM (2026-08-29): the 0.75
# mask-dropout fine-tune redone with 40k steps (4x the original -- the small
# paper split makes this the papers' data-constrained regime), the Muon
# optimizer on hidden matrices (Newton-Schulz orthogonalized momentum,
# muon_lr 2e-3), and the strong decoupled weight decay of "Pre-training under
# infinite compute" (Kim et al. 2025): WD ~30x standard -> 3.0 on matrices,
# 0.3 on embeddings, 0 on norms/biases. AdamW 3e-5 drives the non-matrix
# params. 2x Ada, batch 16 x accum 2 x 2 = eff 64 (recipe invariant).
# Original context comment follows.
# CONTEXT-DROPOUT FINE-TUNE (2026-08-27). The whole-song diagnosis: the model
# has never seen imperfect score context (--mask_prob existed, class default
# 0.75, but NO run ever passed it), so its huge clean-context skill (42-68%
# tol1 continuing from a GT overlap) collapses to ~8% on its own context.
# Beyer & Dai train with 75% context dropout for exactly this
# (arXiv:2410.00210). This fine-tunes run_paper_split_v2/checkpoint-2500 with
# mask_prob 0.75, otherwise the original paper-split recipe.
#
# 4x A6000, batch 4 x accum 4 x 4 ranks = effective 64 (the recipe invariant;
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
    --output_dir ./run_paper_split_v2_maskft_muon40k \
    --wandb_run_name run_paper_split_v2_maskft_muon40k \
    --wandb_mode online \
    --mask_prob 0.75 \
    --max_steps 40000 \
    --batch_size 16 \
    --val_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --save_steps 2000 \
    --eval_steps 500 \
    --optimizer muon \
    --muon_lr 2e-3 \
    --muon_wd 3.0 \
    --emb_wd 0.3
echo MASKFT_DONE
