#!/bin/bash
#SBATCH -J fp32feat
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH --array=0-1
#SBATCH -o /home/wjl86/anticipation3/logs/fp32feat_%A_%a.out
# The two token_features=2 selectors, on the fp32 feature shards.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;n=torch.cuda.get_device_name(0);print(f'{n} {t:.1f}GiB');assert t>=20,f'refusing {n}'"
ls nbest_data/fp32feat_train_shard*.pt >/dev/null || { echo "feature shards missing"; exit 1; }
case $SLURM_ARRAY_TASK_ID in
 0) python3 -m nbest.train_reranker_pairwise --shards 'nbest_data/fp32feat_train_shard*.pt' --run_dir run_nbest_reranker/pairwise32feat_fp32 --steps 20000 --windows_per_batch 2 --fp32 ;;
 1) python3 -m nbest.train_duel              --shards 'nbest_data/fp32feat_train_shard*.pt' --run_dir run_nbest_reranker/duel32_fp32        --steps 20000 --windows_per_batch 2 --pairs_per_window 8 --fp32 ;;
esac
echo FP32FEAT_OK
