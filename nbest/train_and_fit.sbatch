#!/bin/bash
#SBATCH --job-name=nbest-rerank
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:2
#SBATCH --mem=48G
#SBATCH --time=24:00:00
#SBATCH --output=logs/nbest_rerank_%j.out
#SBATCH --error=logs/nbest_rerank_%j.out
# Reranker training (2-GPU DDP) then decode-weight fitting on the val shard.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 NCCL_P2P_DISABLE=1
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  -m nbest.train_reranker --shards 'nbest_data/train_shard*.pt' \
  --run_dir "$RUN_DIR" --steps 20000
python -m nbest.fit_weights --val-shards nbest_data/val_shard00.pt \
  --reranker-ckpt "$RUN_DIR/final.pt" --output nbest_data/decode_weights.json
