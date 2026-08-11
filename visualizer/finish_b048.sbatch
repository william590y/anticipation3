#!/bin/bash
#SBATCH --job-name=finish-b048
#SBATCH --partition=ellis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=logs/finish_b048_%j.out
#SBATCH --error=logs/finish_b048_%j.out

# CPU post-process for the 0.48s viz dataset (MUSTER + real-XML sidecar + re-key).
# Run as a batch job so it survives login-session teardowns.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
bash visualizer/finish_b048.sh
