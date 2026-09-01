#!/bin/bash
# Submit one failure-safe chain:
#   corrected PPO (3 GPUs) -> validate/select best -> PPO viz (4 GPUs) -> publish
#
# Optional environment:
#   BEGIN_TIME             Slurm --begin value for the training job. Empty = now.
#   PPO_RUN_DIR            Fresh output directory (default: timestamped directory).
#   PPO_LR, PPO_EPOCHS, PPO_GAMMA, PPO_GAE_LAMBDA, PPO_TARGET_KL
#                          Passed through and recorded in the selection manifest;
#                          defaults are the stable_v2 diagnostic-probe settings.

set -euo pipefail

repo_root="/home/wjl86/anticipation3"
cd "${repo_root}"
mkdir -p logs visualizer/ppo_shards

run_stamp="$(date +%Y%m%d_%H%M%S)_${BASHPID}"
PPO_RUN_DIR="${PPO_RUN_DIR:-${repo_root}/run_ppo_corrected_${run_stamp}}"
PPO_RUN_DIR="$(realpath -m "${PPO_RUN_DIR}")"
PPO_DATA_FILE="${PPO_DATA_FILE:-${repo_root}/visualizer/data.js}"
PPO_DATA_FILE="$(realpath -m "${PPO_DATA_FILE}")"
PPO_MANIFEST="${PPO_RUN_DIR}/selected_best_val.json"
PPO_LR="${PPO_LR:-${PPO_LEARNING_RATE:-3e-7}}"
PPO_EPOCHS="${PPO_EPOCHS:-2}"
PPO_GAMMA="${PPO_GAMMA:-1.0}"
PPO_GAE_LAMBDA="${PPO_GAE_LAMBDA:-0.95}"
PPO_TARGET_KL="${PPO_TARGET_KL:-0.02}"

for export_value in \
    "${PPO_RUN_DIR}" "${PPO_DATA_FILE}" "${PPO_MANIFEST}" \
    "${PPO_LR}" "${PPO_EPOCHS}" "${PPO_GAMMA}" "${PPO_GAE_LAMBDA}" "${PPO_TARGET_KL}"; do
    if [[ "${export_value}" == *,* ]]; then
        echo "Slurm export values may not contain commas: ${export_value}" >&2
        exit 2
    fi
done
if [[ -e "${PPO_RUN_DIR}" ]]; then
    echo "Refusing to reuse PPO_RUN_DIR; choose a fresh path: ${PPO_RUN_DIR}" >&2
    exit 2
fi
if [[ ! -f "${PPO_DATA_FILE}" ]]; then
    echo "Visualizer data file does not exist: ${PPO_DATA_FILE}" >&2
    exit 2
fi

train_submit=(
    --parsable
    "--export=ALL,PPO_RUN_DIR=${PPO_RUN_DIR},PPO_LR=${PPO_LR},PPO_EPOCHS=${PPO_EPOCHS},PPO_GAMMA=${PPO_GAMMA},PPO_GAE_LAMBDA=${PPO_GAE_LAMBDA},PPO_TARGET_KL=${PPO_TARGET_KL}"
)
if [[ -n "${BEGIN_TIME:-}" ]]; then
    train_submit+=("--begin=${BEGIN_TIME}")
fi
train_raw="$(sbatch "${train_submit[@]}" train_ppo_production.sbatch)"
train_job_id="${train_raw%%;*}"
if [[ ! "${train_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse PPO training job id from: ${train_raw}" >&2
    exit 1
fi

PPO_SHARD_DIR="${repo_root}/visualizer/ppo_shards/train_${train_job_id}"
if [[ -e "${PPO_SHARD_DIR}" ]]; then
    echo "Unexpected pre-existing train-ID shard directory: ${PPO_SHARD_DIR}" >&2
    exit 2
fi

select_raw="$(sbatch --parsable \
    "--dependency=afterok:${train_job_id}" \
    --kill-on-invalid-dep=yes \
    "--export=ALL,PPO_RUN_DIR=${PPO_RUN_DIR},PPO_MANIFEST=${PPO_MANIFEST},PPO_TRAIN_JOB_ID=${train_job_id},PPO_LR=${PPO_LR},PPO_EPOCHS=${PPO_EPOCHS},PPO_GAMMA=${PPO_GAMMA},PPO_GAE_LAMBDA=${PPO_GAE_LAMBDA},PPO_TARGET_KL=${PPO_TARGET_KL}" \
    visualizer/select_ppo_best.sbatch)"
select_job_id="${select_raw%%;*}"
if [[ ! "${select_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse PPO selector job id from: ${select_raw}" >&2
    exit 1
fi

rollout_raw="$(sbatch --parsable \
    "--dependency=afterok:${select_job_id}" \
    --kill-on-invalid-dep=yes \
    "--export=ALL,PPO_MANIFEST=${PPO_MANIFEST},PPO_SHARD_DIR=${PPO_SHARD_DIR},PPO_DATA_FILE=${PPO_DATA_FILE}" \
    visualizer/precompute_ppo.sbatch)"
rollout_job_id="${rollout_raw%%;*}"
if [[ ! "${rollout_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse PPO rollout-array job id from: ${rollout_raw}" >&2
    exit 1
fi

publish_raw="$(sbatch --parsable \
    "--dependency=afterok:${rollout_job_id}" \
    --kill-on-invalid-dep=yes \
    "--export=ALL,PPO_MANIFEST=${PPO_MANIFEST},PPO_SHARD_DIR=${PPO_SHARD_DIR},PPO_DATA_FILE=${PPO_DATA_FILE}" \
    visualizer/publish_ppo.sbatch)"
publish_job_id="${publish_raw%%;*}"
if [[ ! "${publish_job_id}" =~ ^[0-9]+$ ]]; then
    echo "Could not parse PPO publisher job id from: ${publish_raw}" >&2
    exit 1
fi

printf 'PPO_RUN_DIR=%s\n' "${PPO_RUN_DIR}"
printf 'PPO_MANIFEST=%s\n' "${PPO_MANIFEST}"
printf 'PPO_SHARD_DIR=%s\n' "${PPO_SHARD_DIR}"
printf 'PPO_HPARAMS=lr:%s,epochs:%s,gamma:%s,gae_lambda:%s,target_kl:%s\n' \
    "${PPO_LR}" "${PPO_EPOCHS}" "${PPO_GAMMA}" "${PPO_GAE_LAMBDA}" "${PPO_TARGET_KL}"
printf 'TRAIN_JOB_ID=%s\n' "${train_job_id}"
printf 'SELECT_JOB_ID=%s dependency=afterok:%s\n' "${select_job_id}" "${train_job_id}"
printf 'ROLLOUT_ARRAY_JOB_ID=%s dependency=afterok:%s tasks=0-3\n' "${rollout_job_id}" "${select_job_id}"
printf 'PUBLISH_JOB_ID=%s dependency=afterok:%s\n' "${publish_job_id}" "${rollout_job_id}"
if [[ -n "${BEGIN_TIME:-}" ]]; then
    printf 'TRAIN_BEGIN=%s\n' "${BEGIN_TIME}"
else
    printf 'TRAIN_BEGIN=immediate\n'
fi
