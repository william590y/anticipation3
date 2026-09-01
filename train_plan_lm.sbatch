#!/bin/bash
#SBATCH --job-name=plan-lm
#SBATCH --partition=ellis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
# The GPU TYPE is pinned, not just the count: ellis mixes a 5x3090 node with an
# 8xA6000 node, so a plain `gpu:4` can land on 24 GB cards. Override both this
# and the partition together (submit_plan_chain.sh passes --gres).
#SBATCH --gres=gpu:nvidia_rtx_a6000:4
#SBATCH --mem=128G
#SBATCH --time=infinite
#SBATCH --output=logs/plan_lm_%j.out
#SBATCH --error=logs/plan_lm_%j.out

set -euo pipefail

cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false

# GPU peer-to-peer is broken on ellis-compute-02: a bare 2-rank NCCL all_reduce
# hangs there forever (P2P_LEVEL=SYS, SHM_DISABLE and IB_DISABLE do not help).
# The hang is in the FIRST collective, so accelerate's opening
# wait_for_everyone() never returns and the job burns 4 GPUs at 100% util
# printing nothing -- it looks like slow startup, not a deadlock.
export NCCL_P2P_DISABLE=1

RUN_DIR="${RUN_DIR:-./run_plan_lm}"
VQ_CHECKPOINT="${VQ_CHECKPOINT:-./run_plan_vq/plan_vq_best.pt}"
PLAN_PLACEMENT="${PLAN_PLACEMENT:-front}"
MAX_STEPS="${MAX_STEPS:-20000}"

if [[ ! -f "${VQ_CHECKPOINT}" ]]; then
    echo "Stage-1 checkpoint ${VQ_CHECKPOINT} is missing; stage 2 cannot start." >&2
    exit 1
fi

echo "Host: $(hostname)"
echo "Job ${SLURM_JOB_ID}; start $(date --iso-8601=seconds)"
echo "run_dir=${RUN_DIR} vq=${VQ_CHECKPOINT} placement=${PLAN_PLACEMENT}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

# The effective batch of 64 is 4 x grad_accum 4 x 4 GPUs, so the GPU count is
# part of the recipe: a --gres override that changes it silently changes the
# optimization this run exists to compare against run_paper_split_v2.
GPUS=$(nvidia-smi --list-gpus | wc -l)
if [[ "${GPUS}" -ne 4 ]]; then
    echo "Expected 4 GPUs (effective batch 4 x 4 x 4 = 64), got ${GPUS}." >&2
    exit 1
fi

# Stage 2: the same fine-tune as train_asap_paper.sbatch, with the plan in front.
#
# The optimization is deliberately identical to run_paper_split_v2 so the only
# difference is the plan:
#   * effective batch stays 64 -- 4 x grad_accum 4 x 4 GPUs (was 4 x 8 x 2).
#   * max_steps 20000, lr 3e-5 cosine to 3e-6, weight anchor 1e5, clip 2.0.
#   * eval subsets 500 / 100 are both divisible by 4, so _resolve_eval_subset_size
#     returns them unchanged at world_size=4.
python -u -m torch.distributed.run --standalone --nproc_per_node=4 \
    train_plan_lm.py \
    --data_file data/train_paper.txt \
    --val_file data/val_paper.txt \
    --vq_checkpoint "${VQ_CHECKPOINT}" \
    --plan_placement "${PLAN_PLACEMENT}" \
    --output_dir "${RUN_DIR}" \
    --wandb_run_name "plan_lm_${SLURM_JOB_ID}" \
    --wandb_mode online \
    --max_steps "${MAX_STEPS}" \
    --batch_size 4 \
    --val_batch_size 4 \
    --gradient_accumulation_steps 4

echo "Job ${SLURM_JOB_ID} end $(date --iso-8601=seconds)"
