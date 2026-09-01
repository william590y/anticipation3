#!/bin/bash
#SBATCH --job-name=nbest-rerank-unf
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:4
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=logs/nbest_rerank_unf_%j.out
#SBATCH --error=logs/nbest_rerank_unf_%j.out
# Reranker training (4-GPU DDP) on the UNFILTERED N-best data, then
# decode-weight fitting on the unfiltered val shard.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 NCCL_P2P_DISABLE=1
RUN_DIR="${RUN_DIR:?set RUN_DIR}"
python -m torch.distributed.run --standalone --nproc_per_node=4 \
  -m nbest.train_reranker --shards 'nbest_data/unf_train_shard*.pt' \
  --run_dir "$RUN_DIR" --steps 20000
python -m nbest.fit_weights --val-shards nbest_data/unf_val_shard00.pt \
  --reranker-ckpt "$RUN_DIR/final.pt" \
  --output nbest_data/decode_weights_unfiltered.json
