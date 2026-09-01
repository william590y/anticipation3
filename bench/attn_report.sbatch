#!/bin/bash
#SBATCH --job-name=bench-attn
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:1
#SBATCH --mem=32G
#SBATCH --time=4:00:00
#SBATCH --output=logs/bench_attn_%j.out
#SBATCH --error=logs/bench_attn_%j.out
# Inference decode benchmark. One 3090 on ellis-compute-01 (2 of its 5 GPUs are
# free; the other three are running listt5-train/genrm-train/claude -- do not
# disturb).
#
# Arguments go on the sbatch command line, NOT through --export: SLURM splits an
# --export value on commas, so `--export=ALL,BENCH_ARGS="--batch-sizes 8,32,96"`
# silently delivers `--batch-sizes 8` and turns "32" and "96" into environment
# variable names. That cost one bogus baseline table (job 462074, which reported
# only batch 8 and defaulted the variant list). Use:
#   sbatch bench/bench_decode.sbatch --batch-sizes 8,32,96,198 --variants ...
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
# The baseline decode grows a DynamicCache by torch.cat every step, so it holds
# the old and new cache simultaneously -- that transient doubling is what pushes
# batch 96+ over a 24 GB card. expandable_segments keeps the freed blocks usable
# instead of fragmenting them away; it does not change any numerics.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python bench/attn_report.py "$@"
