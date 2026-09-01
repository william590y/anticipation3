#!/bin/bash
#SBATCH --job-name=randrep15
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/home/wjl86/anticipation3/logs/randreplace_%j.out
#
# EXPERIMENT 4 -- TRAINING-TIME TOKEN NOISE (2026-08-30)
#
# Paper: "Demystifying Training-Time Augmentation for Data-Constrained
# Language Model Pretraining", Chen et al., arXiv:2606.16246.
#
# FINDING BEING REPRODUCED: of the paper's three augmentation families
# (token-level noise, sequence permutation, target-offset prediction), the
# best INDIVIDUAL method is token-level *random replacement* at
# alpha_r = 15% -- val loss 3.841 @ epoch 28, beating <mask>-ing at the
# matched rate (3.910 @ 15%) and beating replacement at 5% (3.912) and 30%.
# (Their best OVERALL config stacks low-rate replacement + right-to-left +
# target-offset prediction, 4.015 -> 3.805; this run is the single-method arm.)
#
# MECHANISM (train.py TokenizedDataset, --rand_replace_prob): their Eq. 1 with
# alpha_m = 0. One uniform draw per BODY token; below alpha_r the INPUT token
# is replaced, otherwise kept. The LABEL SEQUENCE IS NEVER MODIFIED -- the
# model always predicts the original x_t, so this is a denoising objective,
# not a corrupted-target one. Corruption is restricted to positions >=
# ALTERNATING_START (192), so the 32-pair structural prefix is protected,
# matching the paper's "control tokens are never corrupted".
#
# DELIBERATE ADAPTATION, NOT THE PAPER'S LITERAL DRAW: the replacement token is
# sampled from the position's OWN role range (score onset / dur / pitch,
# control onset / dur / pitch; train._role_range), not uniformly from the whole
# 55028-token vocab. In this repo's packed alternating format the role of a
# position is fixed by (pos - 192) % 6, so a cross-range token is not
# "plausible but incorrect" -- it is structurally ILLEGAL and trivially
# detectable, and the model would learn to spot corruption instead of denoising
# it. `--rand_replace_whole_vocab` switches to the paper-literal draw if the
# comparison is ever wanted.
#
# VERIFIED EMPIRICALLY before submit (400 windows of data/train_paper.txt):
# realised body corruption 14.93% (expected 14.975% after same-token
# collisions), 0 corrupted positions < 192, 0 tokens outside their role range,
# labels byte-identical to the uncorrupted dataset's input_ids, and nothing
# corrupted with is_training=False.
#
# RECIPE: fine-tune run_paper_split_v2/checkpoint-2500 (the shared init for
# every post-training arm) on the paper split, 10k steps. 1x RTX 3090 (24GB),
# batch 8 x accum 8 x 1 rank = effective 64, the repo-wide invariant.
# Typed GRES (not --constraint) because a bare gpu:1 gets handed dead or
# ancient cards; the runtime preflight re-asserts the card and does a real bf16
# matmul, which is the only thing that catches a node-wide-dead NVML.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1
python -c "
import torch
assert torch.cuda.device_count() == 1, torch.cuda.device_count()
name = torch.cuda.get_device_name(0)
free, total = torch.cuda.mem_get_info()
x = torch.randn(64, 64, device='cuda', dtype=torch.bfloat16)
y = (x @ x).float().sum().item()
assert y == y, 'bf16 matmul produced NaN'
assert '3090' in name, f'refusing {name}'
print(f'preflight OK: {name} {total/2**30:.1f}GiB, bf16 matmul ran')
"
python train.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --model_name run_paper_split_v2/checkpoint-2500 \
    --output_dir ./run_paper_split_v2_randreplace15 \
    --wandb_run_name run_paper_split_v2_randreplace15 \
    --wandb_mode online \
    --rand_replace_prob 0.15 \
    --max_steps 10000 \
    --batch_size 8 \
    --val_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --save_steps 1000 \
    --eval_steps 500
echo RANDREPLACE_DONE
