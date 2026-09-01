#!/bin/bash
#SBATCH -J fp32ptfit
#SBATCH --gres=gpu:1
#SBATCH --mem=96G
#SBATCH -c 8
#SBATCH -t 24:00:00
#SBATCH -o /home/wjl86/anticipation3/logs/fp32ptfit_%j.out
# The fitted alpha/beta/gamma row needs a POINTWISE reranker for its gamma
# term (the bf16 table used unfiltered_0819_0955). Retrain it in fp32, then
# refit alpha/beta/gamma on the fp32 val shard -- mandatory, not optional:
# the existing decode_weights_unfiltered.json is fitted to bf16 feature
# scales, and logp_base has moved (the base model is now fp32 too), so
# reusing it would silently invalidate the fitted row.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;n=torch.cuda.get_device_name(0);print(f'{n} {t:.1f}GiB');assert t>=20,f'refusing {n}'"
python3 -m nbest.train_reranker --shards 'nbest_data/fp32_9_train_shard*.pt' \
    --run_dir run_nbest_reranker/pointwise_fp32 --steps 20000 --fp32
python3 -m nbest.fit_weights \
    --val-shards 'nbest_data/fp32feat_val_shard00.pt' \
    --reranker-ckpt run_nbest_reranker/pointwise_fp32/final.pt \
    --output nbest_data/decode_weights_fp32.json
echo FP32PTFIT_OK
cat nbest_data/decode_weights_fp32.json
