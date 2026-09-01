#!/bin/bash
#SBATCH -J fs-ours
#SBATCH -p ellis,default_partition
#SBATCH -A thickstun
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|6000ada|a100|a40|3090"
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 12:00:00
#SBATCH --array=0-7
#SBATCH -o /home/wjl86/anticipation3/logs/fs_ours_%A_%a.out
# Full-song sliding-window rollouts, SPLIT env selects test|val.
# File-gated: the driver skips outputs that exist, so a preemption resumes.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;n=torch.cuda.get_device_name(0);print(f'{n} {t:.1f}GiB');assert t>=10,f'refusing {n}'"
python3 -u full_song_rollout.py --split "${SPLIT:?}" --window-mode "${MODE:-slide}" \
    --overlap-source "${CTX:-pred}" \
    --window-overlap "${OVERLAP:-69}" \
    --checkpoint "${CKPT:-run_paper_split_v2/checkpoint-2500}" \
    --out-dir "${OUTDIR:-fullsong_rollouts}" \
    ${PFORCE:+--pitch-force} \
    ${TRACE:+--save-trace} \
    --shard-index "$SLURM_ARRAY_TASK_ID" --num-shards 8
