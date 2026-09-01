#!/bin/bash
set -euo pipefail
cd /home/wjl86/anticipation3

grpo_parent="${GRPO_PARENT_JOB_ID:-27571}"
if ! [[ "${grpo_parent}" =~ ^[0-9]+$ ]]; then
  echo "GRPO_PARENT_JOB_ID must be a numeric SLURM job id" >&2
  exit 2
fi
mkdir -p visualizer/grpo_final_shards visualizer/seed_backfill_shards visualizer/logs
if compgen -G 'visualizer/grpo_final_shards/*.json' >/dev/null || \
   compgen -G 'visualizer/seed_backfill_shards/*.json' >/dev/null; then
  echo "Refusing to submit with stale JSON in grpo_final_shards/ or seed_backfill_shards/." >&2
  echo "Move the prior outputs aside, then resubmit." >&2
  exit 3
fi

final_raw="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${grpo_parent}" visualizer/finalize_grpo_seed_shards.sbatch)"
final_job="${final_raw%%;*}"
backfill_raw="$(sbatch --parsable visualizer/precompute_seed_backfill.sbatch)"
backfill_job="${backfill_raw%%;*}"
publish_raw="$(sbatch --parsable --kill-on-invalid-dep=yes --dependency="afterok:${final_job}:${backfill_job}" visualizer/publish_seed_pipeline.sbatch)"
publish_job="${publish_raw%%;*}"

echo "GRPO source array: ${grpo_parent}"
echo "GRPO finalize array: ${final_job} (afterok:${grpo_parent})"
echo "Seed backfill array: ${backfill_job} (independent, 12 tasks / max 4 GPUs)"
echo "Atomic publish job: ${publish_job} (afterok:${final_job}:${backfill_job})"
echo "CHAIN=${grpo_parent}->${final_job} + ${backfill_job}->${publish_job}"
