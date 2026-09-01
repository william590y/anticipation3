#!/bin/bash
#SBATCH -J fs-papers
#SBATCH -p ellis,default_partition
#SBATCH -A thickstun
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|6000ada|a100|a40|3090|a5500|a5000|l4"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow
#SBATCH -c 2
#SBATCH --mem=16G
#SBATCH -t 2:00:00
#SBATCH --array=0-15
#SBATCH -o /home/wjl86/anticipation3/logs/fs_papers_%A_%a.out
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate paperpipe
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;n=torch.cuda.get_device_name(0);print(f'{n} {t:.1f}GiB');assert t>=10,f'refusing {n}'"
# All shards zeng-first: the paper1-loader crash left Beyer 92/145 done and
# Zeng 0/145, so Zeng IS the remaining work; each shard sweeps its Beyer
# leftovers afterwards (file-gated skips make that cheap).
python3 -u visualizer/run_paper_fullsong.py --splits test,val \
    --shard-index "$SLURM_ARRAY_TASK_ID" --num-shards 16 --zeng-first
