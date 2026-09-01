#!/bin/bash
#SBATCH --job-name=dagger75
#SBATCH --partition=thickstun
#SBATCH --account=thickstun
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=/home/wjl86/anticipation3/logs/dagger75_%j.out
#
# DAGGER-STYLE SLIDING-WINDOW FINE-TUNE (2026-08-28). Diagnosis: whole-song
# decode fails because sliding re-packs prime each window with the model's own
# drifted predictions -- a state distribution never seen in training (windows
# are always GT-context). DAgger fix: roll the model forward with the
# PRODUCTION slide decode over TRAIN-split performances, harvest each re-pack
# boundary's actual self-context window, complete it with aligned GT targets
# for the 69 fresh slots, and fine-tune with loss masked to the GT
# continuation (--loss_mask_first_score_slots 69): model output as
# input/context, ground truth as target -- never self-distillation.
# Base checkpoint: the 0.75 mask fine-tune (best whole-song IOI scale-max).
# Phase 1: generation, 2 shards (one per GPU), 300-perf even subsample.
# Phase 2: train 5000 steps, batch 16 x accum 2 x 2 ranks = effective 64.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
python -c "import torch;assert torch.cuda.device_count()==2;print('preflight OK')"
CKPT=run_paper_split_v2_maskft/checkpoint-10000
for S in 0 1; do
  CUDA_VISIBLE_DEVICES=$S python dagger_gen.py --checkpoint $CKPT \
    --out-dir dagger_windows/train_maskft75 --max-perfs 300 \
    --shard-index $S --num-shards 2 > logs/dagger_gen_${SLURM_JOB_ID}_$S.log 2>&1 &
done
wait
cat dagger_windows/train_maskft75/*.txt > data/dagger_maskft75.txt
NW=$(grep -c . data/dagger_maskft75.txt)
echo "DAgger windows: $NW"
[ "$NW" -ge 500 ] || { echo "too few windows, aborting"; exit 1; }
awk 'NF != 1020 {print "BAD LINE LENGTH", NR, NF; exit 1}' data/dagger_maskft75.txt
python -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
    --data_file data/dagger_maskft75.txt \
    --val_file data/val_paper.txt \
    --model_name $CKPT \
    --output_dir ./run_paper_split_v2_dagger75 \
    --wandb_run_name run_paper_split_v2_dagger75 \
    --wandb_mode online \
    --loss_mask_first_score_slots 69 \
    --max_steps 5000 \
    --batch_size 16 \
    --val_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --save_steps 500 \
    --eval_steps 250
echo DAGGER_DONE
