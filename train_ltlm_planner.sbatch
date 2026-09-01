#!/bin/bash
#SBATCH --job-name=train-ltlm-planner
#SBATCH --partition=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:4
#SBATCH --mem=128G
#SBATCH --time=infinite
#SBATCH --output=logs/train_ltlm_planner_%j.out
#SBATCH --error=logs/train_ltlm_planner_%j.out

set -euo pipefail

# Restart of job 69260 (gaussian PerformancePlanner) on the DDP-fixed
# trainer. Isolated checkout of local main — do not clone origin, and do
# not reuse the old github branch that still has unwrap-elbo desync.
SRC=/home/wjl86/anticipation3
WORKDIR=/home/wjl86/anticipation3-planner

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

GPUS=$(nvidia-smi --list-gpus | wc -l)
if [[ "${GPUS}" -ne 4 ]]; then
    echo "Expected 4 GPUs (effective batch 4 x 4 x 4 = 64), got ${GPUS}." >&2
    exit 1
fi

# Same algorithm as crashed job 69260 / wandb ltlm_planner: gaussian
# planner, beta=5, 4 thoughts/layer, 16 AdamVI steps, effective batch 64.
# --planner gaussian is required: train_ltlm.py now defaults to diffusion.
python -m torch.distributed.run --standalone --nproc_per_node=4 train_ltlm.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --output_dir ./run_ltlm_planner \
    --wandb_run_name ltlm_planner \
    --wandb_mode online \
    --planner gaussian \
    --max_steps 20000 \
    --batch_size 4 \
    --val_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --kl_weight 5 \
    --thoughts_per_layer 4 \
    --mcmc_steps 16 \
    --compile \
    --compile_mode default \
    --attn_implementation sdpa \
    --num_workers 4

echo "End: $(date)"
