#!/bin/bash
#SBATCH --job-name=mft-muon-e
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=48:00:00
#SBATCH --output=/home/wjl86/anticipation3/logs/maskft_muon_e_%j.out
#
# MUON ARM v2 (2026-08-30). v1 (job 914086) ran all 40k steps but DESTROYED
# the model: two compounding faults. (a) muon_lr 2e-3 x wd 3.0 = 6e-3 decay
# per step -> weight-shrink timescale 167 steps against a 40k-step run, so the
# pretrained weights were annihilated in the first few hundred steps (loss
# 2.9 -> 57 by step 500). (b) train.py anchors to the ORIGINAL weights with
# --original_weight_l2 default 1e5, a regularizer that pulls toward the
# pretrained point while WD pulls toward zero: at the end AnchorTerm was 61.5
# of the 63.7 total loss (AnchorL2 6.1e-4 vs the healthy run's 6.2e-7), i.e.
# the objective was almost entirely "undo the weight decay" rather than the
# task. Final val loss 1.906 vs mask75's 1.245; val AR pitch 100 is a
# degenerate readout of a collapsed model, not a good number.
# MOVED TO ELLIS 3090s (2026-08-30): thickstun projected a ~2-day wait. The
# 24GB cards take batch 8/GPU, so grad_accum doubles to 4 and the effective
# batch stays 8 x 4 x 2 = 64 -- the recipe invariant every arm shares.
# v2: muon_lr 2e-4, muon_wd 0.5 (decay timescale ~10k steps, comparable to the
# 40k run, still 50x the 0.01 default), emb_wd 0.05, and the original-weight
# anchor DISABLED so decoupled WD is the sole regularizer -- which is what
# "Pre-training under infinite compute" actually prescribes.
#
# ORIGINAL v1 HEADER:
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
    --output_dir ./run_paper_split_v2_maskft_muon40k_v2 \
    --wandb_run_name run_paper_split_v2_maskft_muon40k_v2 \
    --wandb_mode online \
    --mask_prob 0.75 \
    --max_steps 40000 \
    --batch_size 8 \
    --val_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --save_steps 2000 \
    --eval_steps 500 \
    --optimizer muon \
    --muon_lr 2e-4 \
    --muon_wd 0.5 \
    --emb_wd 0.05 \
    --original_weight_l2 0
echo MASKFT_DONE
