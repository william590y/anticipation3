#!/bin/bash
#SBATCH --job-name=train-ltlm-planner
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=infinite
#SBATCH --output=logs/train_ltlm_planner_%j.out
#SBATCH --error=logs/train_ltlm_planner_%j.out

set -euo pipefail

# Dedicated checkout so we do not clobber the unpushed Cornell train_ltlm.py.
SRC=/home/wjl86/anticipation3
WORKDIR=/home/wjl86/anticipation3-planner
BRANCH=cursor/thickstun-ltlm-sbatch-50bc

if [ ! -d "$WORKDIR/.git" ]; then
  git clone --branch "$BRANCH" https://github.com/william590y/anticipation3.git "$WORKDIR"
else
  git -C "$WORKDIR" fetch origin "$BRANCH"
  git -C "$WORKDIR" checkout "$BRANCH"
  git -C "$WORKDIR" pull --ff-only origin "$BRANCH"
fi

# Token files live in the original repo and are gitignored.
ln -sfn "$SRC/data" "$WORKDIR/data"
mkdir -p "$WORKDIR/logs" "$SRC/logs"

cd "$WORKDIR"

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
echo "Workdir: $PWD"
echo "Commit: $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Start: $(date)"

# Same per-GPU batch as the killed ltlm_63400/63401 jobs (4 x 4 x 4 GPUs = 64).
# torchrun rather than accelerate launch: this box has a stale accelerate
# default_config.yaml that would override gradient accumulation.
python -m torch.distributed.run --standalone --nproc_per_node=4 train_ltlm.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --output_dir ./run_ltlm_planner \
    --wandb_run_name ltlm_planner \
    --wandb_mode online \
    --max_steps 20000 \
    --batch_size 4 \
    --val_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --kl_weight 5 \
    --thoughts_per_layer 4 \
    --mcmc_steps 16

echo "End: $(date)"
