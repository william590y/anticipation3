#!/bin/bash
#SBATCH -J fp32sel
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH --array=0-3
#SBATCH -o /home/wjl86/anticipation3/logs/fp32sel_%A_%a.out
# The four selectors that need no per-token features, retrained in fp32 on the
# fp32 shards. Launch args mirror the bf16 runs exactly (recovered from
# wandb-metadata.json) so dtype is the ONLY difference.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;n=torch.cuda.get_device_name(0);print(f'{n} {t:.1f}GiB');assert t>=20,f'refusing {n}'"
case $SLURM_ARRAY_TASK_ID in
 0) python3 -m nbest.train_reranker_pairwise --shards 'nbest_data/fp32_9_train_shard*.pt'  --run_dir run_nbest_reranker/pairwise_fp32   --steps 20000 --fp32 ;;
 1) python3 -m nbest.train_reranker_pairwise --shards 'nbest_data/fp32_32_train_shard*.pt' --run_dir run_nbest_reranker/pairwise32_fp32 --steps 20000 --windows_per_batch 2 --fp32 ;;
 2) python3 -m nbest.train_reranker_listwise --shards 'nbest_data/fp32_32_train_shard*.pt' --run_dir run_nbest_reranker/listwise32_fp32 --steps 20000 --windows_per_batch 2 --fp32 ;;
 3) python3 -m nbest.train_genrm            --shards 'nbest_data/fp32_32_train_shard*.pt' --run_dir run_nbest_reranker/genrm_fp32      --steps 8000  --cands_per_step 4 --fp32 ;;
esac
echo FP32SEL_OK
