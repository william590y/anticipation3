#!/bin/bash
set -euo pipefail
cd /home/wjl86/anticipation3

parent_job="${PAPER_PARENT_JOB_ID:-}"
if [[ -n "${parent_job}" ]] && ! [[ "${parent_job}" =~ ^[0-9]+$ ]]; then
  echo "PAPER_PARENT_JOB_ID must be a numeric Slurm job id" >&2
  exit 2
fi

shard_root="${PAPER_SHARD_ROOT:-/home/wjl86/anticipation3/visualizer/paper_seed_shards}"
data_file="${PAPER_DATA_FILE:-/home/wjl86/anticipation3/visualizer/data.js}"
mkdir -p "${shard_root}" visualizer/logs

array_args=(--parsable --export="ALL,PAPER_SHARD_ROOT=${shard_root},PAPER_DATA_FILE=${data_file}")
if [[ -n "${parent_job}" ]]; then
  array_args+=(--kill-on-invalid-dep=yes --dependency="afterok:${parent_job}")
fi
array_raw="$(sbatch "${array_args[@]}" visualizer/precompute_paper_seeds.sbatch)"
array_job="${array_raw%%;*}"
if ! [[ "${array_job}" =~ ^[0-9]+$ ]]; then
  echo "Could not parse paper array job id from: ${array_raw}" >&2
  exit 3
fi

shard_dir="${shard_root}/${array_job}"
publish_raw="$(sbatch --parsable --kill-on-invalid-dep=yes \
  --dependency="afterok:${array_job}" \
  --export="ALL,PAPER_SHARD_DIR=${shard_dir},PAPER_DATA_FILE=${data_file}" \
  visualizer/publish_paper_seeds.sbatch)"
publish_job="${publish_raw%%;*}"

if [[ -n "${parent_job}" ]]; then
  echo "Parent job: ${parent_job}"
  echo "Paper seed array: ${array_job} (afterok:${parent_job}; four Thickstun GPUs)"
else
  echo "Paper seed array: ${array_job} (four Thickstun GPUs)"
fi
echo "Paper shard directory: ${shard_dir}"
echo "Atomic publisher: ${publish_job} (afterok:${array_job})"
echo "CHAIN=${parent_job:+${parent_job}->}${array_job}->${publish_job}"
