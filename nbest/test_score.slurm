#!/bin/bash
#SBATCH -J testscore
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH -c 8
#SBATCH -t 4:00:00
#SBATCH --requeue
#SBATCH -A thickstun
#SBATCH -o /home/wjl86/anticipation3/logs/testscore_%j.out
# A FILE, not --wrap: sbatch --wrap runs the string under `sh`, where `source`
# does not exist (job 585106 died on `source: not found` in 4 seconds), and it
# also expands $VARIABLES in the submitting shell (the documented array trap).
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;n=torch.cuda.get_device_name(0);print(f'{n} {t:.1f}GiB');assert t>=20,f'refusing {n}'"
python3 -u nbest/eval_test_selectors.py \
    --shard nbest_data/test9feat_stride150.pt \
    --token-file data/test_paper.txt
echo TESTSCORE_OK
