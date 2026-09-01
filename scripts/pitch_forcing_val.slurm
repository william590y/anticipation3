#!/bin/bash
#SBATCH -J pforce-val
#SBATCH -p ellis,default_partition
#SBATCH -A thickstun
#SBATCH --requeue
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|6000ada|a100|a40|3090|a5500|a5000|l4"
#SBATCH --exclude=ma-compute-02,kuleshov-compute-01,badfellow
#SBATCH -c 2
#SBATCH --mem=32G
#SBATCH -t 4:00:00
#SBATCH -o /home/wjl86/anticipation3/logs/pforce_val_%j.out
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;assert t>=10"
B=$(python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;print(48 if t>=40 else 16)")
python3 -u nbest/pitch_forcing_eval.py --batch "$B" \
    --shard nbest_data/val2_stride150.pt \
    --payload visualizer/data_valset.js \
    --save-tokens nbest_data/pitch_forced_tokens_val.pt \
    --out nbest_data/pitch_forcing_eval_val.json
echo PFORCEVAL_OK
