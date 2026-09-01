#!/bin/bash
# Submit the plan pipeline as one failure-safe chain:
#
#   stage 1  train_plan_vq.sbatch    4 GPUs  quantize frozen PianoBART features into a plan
#      | afterok
#   stage 2  train_plan_lm.sbatch    4 GPUs  the same fine-tune as run_paper_split_v2,
#                                            with the plan prefixed to every window
#
# Both stages use the FILTERED paper split (data/train_paper.txt / val_paper.txt).
#
# Stage 2 runs the stage-1 encoder inline (both are stock transformers), so it
# sees the plan for the window as *augmented* and there is no code-export step.
#
# Optional environment:
#   KILL_JOB         Job id to scancel first, freeing its GPUs for this chain.
#   PARTITION        Slurm partition (default: ellis).
#   GRES             GPU request (default: gpu:nvidia_rtx_a6000:4). The TYPE is
#                    pinned because ellis mixes a 5x3090 node with an 8xA6000
#                    node and a plain `gpu:4` can land on 24 GB cards. Override
#                    it together with PARTITION -- e.g.
#                    PARTITION=thickstun GRES=gpu:nvidia_rtx_6000_ada_generation:4
#                    -- and keep the count at 4: both sbatch scripts run
#                    --nproc_per_node=4 and stage 2's effective batch of 64 is
#                    4 x grad_accum 4 x 4 GPUs.
#   RUN_STAMP        Suffix for the two output directories.
#   NUM_CODES        Plan length, 4-16 (default 8).
#   CODEBOOK_SIZE    VQ codebook entries (default 512).
#   PLAN_PLACEMENT   front | after_prefix  (default front -- the literal spec).
#   VQ_MAX_STEPS     Stage-1 steps (default 20000).
#   LM_MAX_STEPS     Stage-2 steps (default 20000, matching run_paper_split_v2).

set -euo pipefail

repo_root="/home/wjl86/anticipation3"
cd "${repo_root}"
mkdir -p logs

PARTITION="${PARTITION:-ellis}"
GRES="${GRES:-gpu:nvidia_rtx_a6000:4}"
RUN_STAMP="${RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
NUM_CODES="${NUM_CODES:-8}"
CODEBOOK_SIZE="${CODEBOOK_SIZE:-512}"
PLAN_PLACEMENT="${PLAN_PLACEMENT:-front}"
VQ_MAX_STEPS="${VQ_MAX_STEPS:-20000}"
LM_MAX_STEPS="${LM_MAX_STEPS:-20000}"

VQ_RUN_DIR="$(realpath -m "${VQ_RUN_DIR:-${repo_root}/run_plan_vq_${RUN_STAMP}}")"
LM_RUN_DIR="$(realpath -m "${LM_RUN_DIR:-${repo_root}/run_plan_lm_${RUN_STAMP}}")"
VQ_CHECKPOINT="${VQ_RUN_DIR}/plan_vq_best.pt"

for value in "${VQ_RUN_DIR}" "${LM_RUN_DIR}" "${VQ_CHECKPOINT}"; do
    if [[ "${value}" == *,* ]]; then
        echo "Slurm export values may not contain commas: ${value}" >&2
        exit 2
    fi
done
for directory in "${VQ_RUN_DIR}" "${LM_RUN_DIR}"; do
    if [[ -e "${directory}" ]]; then
        echo "Refusing to reuse an existing run directory: ${directory}" >&2
        exit 2
    fi
done

if [[ -n "${KILL_JOB:-}" ]]; then
    echo "Cancelling job ${KILL_JOB} to free its GPUs..."
    scancel "${KILL_JOB}"
    # scancel returns before the allocation is actually released.
    for _ in $(seq 1 60); do
        squeue -j "${KILL_JOB}" -h -o "%i" 2>/dev/null | grep -q . || break
        sleep 2
    done
    if squeue -j "${KILL_JOB}" -h -o "%i" 2>/dev/null | grep -q .; then
        echo "WARNING: job ${KILL_JOB} is still in the queue; submitting anyway." >&2
    else
        echo "Job ${KILL_JOB} released."
    fi
fi

stage1=$(sbatch --parsable \
    --partition="${PARTITION}" --gres="${GRES}" \
    --export=ALL,RUN_DIR="${VQ_RUN_DIR}",NUM_CODES="${NUM_CODES}",CODEBOOK_SIZE="${CODEBOOK_SIZE}",MAX_STEPS="${VQ_MAX_STEPS}" \
    train_plan_vq.sbatch)
echo "stage 1 (plan VQ): job ${stage1}  -> ${VQ_RUN_DIR}"

stage2=$(sbatch --parsable \
    --partition="${PARTITION}" --gres="${GRES}" \
    --dependency=afterok:"${stage1}" --kill-on-invalid-dep=yes \
    --export=ALL,RUN_DIR="${LM_RUN_DIR}",VQ_CHECKPOINT="${VQ_CHECKPOINT}",PLAN_PLACEMENT="${PLAN_PLACEMENT}",MAX_STEPS="${LM_MAX_STEPS}" \
    train_plan_lm.sbatch)
echo "stage 2 (plan LM): job ${stage2}  -> ${LM_RUN_DIR}  (afterok:${stage1})"

cat <<EOF

plan: ${NUM_CODES} codes x ${CODEBOOK_SIZE}-entry codebook, placement=${PLAN_PLACEMENT}
score encoder: frozen pretrained PianoBART
slurm: partition=${PARTITION} gres=${GRES}
watch:  tail -f logs/plan_vq_${stage1}.out
        tail -f logs/plan_lm_${stage2}.out
EOF
