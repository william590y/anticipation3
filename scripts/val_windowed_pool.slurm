#!/bin/bash
#SBATCH -J val-wpool
#SBATCH -p ellis,default_partition
#SBATCH -A thickstun
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|6000ada|a100|a40|3090"
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 8:00:00
#SBATCH -o /home/wjl86/anticipation3/logs/val_wpool_%j.out
# Windowed VAL coverage for the histograms: greedy + 1 sample per stride-150
# val window (n-sampled 1 -- histograms need only the greedy rollout).
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;n=torch.cuda.get_device_name(0);print(f'{n} {t:.1f}GiB');assert t>=10,f'refusing {n}'"
WB=$(python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;print(24 if t>=40 else 8)")
python3 -u -m nbest.generate_nbest \
    --token-file data/val_paper.txt --stride 150 --n-sampled 1 \
    --checkpoint run_paper_split_v2/checkpoint-2500 \
    --window-batch "$WB" --output nbest_data/val2_stride150.pt \
    --save-every 50 --resume
echo VALPOOL_OK
