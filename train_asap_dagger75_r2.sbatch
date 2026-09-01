#!/bin/bash
#SBATCH --job-name=dagger75r2
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --output=/home/wjl86/anticipation3/logs/dagger75r2_%j.out
#
# DAGGER ROUND 2 (2026-08-29). Round 1 (aggregate) improved every continued
# slide segment by ~+3 pts (16.5/12.5/8.6/7.4/6.5/8.0 vs baseline
# 12.8/9.2/5.3/7.4/4.1/4.9) holding val AR pitch at 94.84. Proper DAgger
# iterates: regenerate boundary windows under the CURRENT policy (v2), whose
# drift signature differs from pi_0's, aggregate D0 u D1 u D2, retrain.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
python -c "import torch;assert torch.cuda.device_count()==2;print('preflight OK')"
CKPT=run_paper_split_v2_dagger75_v2/checkpoint-5000
for S in 0 1; do
  CUDA_VISIBLE_DEVICES=$S python dagger_gen.py --checkpoint $CKPT \
    --out-dir dagger_windows/train_dagger75v2_r2 --max-perfs 300 \
    --shard-index $S --num-shards 2 > logs/dagger_gen_r2_${SLURM_JOB_ID}_$S.log 2>&1 &
done
wait
cat dagger_windows/train_dagger75v2_r2/*.txt > data/dagger_r2.txt
cat data/dagger_maskft75.txt data/dagger_r2.txt > data/dagger_r1r2.txt
NW=$(grep -c . data/dagger_r1r2.txt)
echo "DAgger aggregate windows (r1+r2): $NW"
[ "$NW" -ge 15000 ] || { echo "too few windows"; exit 1; }
awk 'NF != 1020 {print "BAD LINE", NR, NF; exit 1}' data/dagger_r2.txt
python -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
    --data_file data/dagger_clean_subset.txt \
    --dagger_file data/dagger_r1r2.txt \
    --val_file data/val_paper.txt \
    --model_name run_paper_split_v2_maskft/checkpoint-10000 \
    --output_dir ./run_paper_split_v2_dagger75_v3 \
    --wandb_run_name run_paper_split_v2_dagger75_v3 \
    --wandb_mode online \
    --loss_mask_first_score_slots 69 \
    --max_steps 5000 \
    --batch_size 8 \
    --val_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --save_steps 500 \
    --eval_steps 250
echo DAGGER_DONE
