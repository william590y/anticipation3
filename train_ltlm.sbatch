#!/bin/bash
#SBATCH --job-name=train-ltlm-diffusion
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gpus=4
#SBATCH --mem=128G
#SBATCH --time=infinite
#SBATCH --output=logs/train_ltlm_diffusion_%j.out
#SBATCH --error=logs/train_ltlm_diffusion_%j.out

set -euo pipefail

# Isolated checkout of local main so the job picks up DDP + compile/SDPA
# without a GitHub push. Do not clone origin: that branch still has the
# unwrap-elbo desync that killed job 69260.
SRC=/home/wjl86/anticipation3
WORKDIR=/home/wjl86/anticipation3-diffusion

if [ ! -d "$WORKDIR/.git" ]; then
  git clone --local --branch main "$SRC" "$WORKDIR"
else
  git -C "$WORKDIR" fetch "$SRC" main
  git -C "$WORKDIR" checkout -B main FETCH_HEAD
fi

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
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=3600
export NCCL_TIMEOUT=3600
# Node-local inductor/triton cache: NFS autotune dirs hang on exit and slow compile.
export TORCHINDUCTOR_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/torchinductor-${USER}"
export TRITON_CACHE_DIR="${SLURM_TMPDIR:-/tmp}/triton-${USER}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

echo "Host: $(hostname)"
echo "SLURM_JOB_PARTITION=${SLURM_JOB_PARTITION:-unset}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Python: $(which python)"
echo "Workdir: $PWD"
echo "Commit: $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "Start: $(date)"

python -m torch.distributed.run --standalone --nproc_per_node=4 train_ltlm.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --output_dir ./run_ltlm_diffusion \
    --wandb_run_name ltlm_diffusion \
    --wandb_mode online \
    --planner diffusion \
    --max_steps 20000 \
    --batch_size 4 \
    --val_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --kl_weight 5 \
    --thoughts_per_layer 4 \
    --mcmc_steps 16 \
    --diffusion_timesteps 1000 \
    --diffusion_layers 3 \
    --ddim_steps 50 \
    --compile \
    --compile_mode default \
    --attn_implementation sdpa \
    --num_workers 4

echo "End: $(date)"
