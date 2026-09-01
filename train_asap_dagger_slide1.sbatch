#!/bin/bash
#SBATCH --job-name=dagger-s1
#SBATCH --partition=ellis
#SBATCH --account=thickstun
#SBATCH --gres=gpu:nvidia_geforce_rtx_3090:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=48:00:00
#SBATCH --output=/home/wjl86/anticipation3/logs/dagger_s1_%j.out
#
# DAGGER IN THE SLIDE-BY-ONE REGIME (2026-08-30). The r1-r3 arms harvested
# boundaries at overlap 69 (69 self-context notes + 69 fresh GT slots). This
# one matches the slide-by-1 DECODE regime instead: overlap 137, so each
# window is 137 notes of the model's own drifted context + exactly ONE fresh
# GT target -- the state the model is actually in when it re-primes every
# note. Boundaries then fire once per note, so --boundary-stride 2 keeps the
# file sane. Loss masks the 137 context slots (--loss_mask_first_score_slots
# 137), so the target is the single next note: exactly the DAgger question
# "given the state you got yourself into, what is the correct action now?".
#
# COST NOTE, stated because it bounds what this arm can show: each window
# supervises 1 note instead of 69, so per optimizer step the gradient sees
# ~69x fewer supervised notes than r1-r3 did. 8000 steps at effective batch
# 64 is ~512k windows seen; the clean subset is kept small (6k) so the dirty
# windows are not swamped by 138-slot clean supervision.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
python -c "import torch;assert torch.cuda.device_count()==2;print('preflight OK')"
CKPT=run_paper_split_v2_maskft/checkpoint-10000
for S in 0 1; do
  CUDA_VISIBLE_DEVICES=$S python dagger_gen.py --checkpoint $CKPT \
    --out-dir dagger_windows/train_slide1 --max-perfs 80 \
    --window-overlap 137 --boundary-stride 2 \
    --shard-index $S --num-shards 2 > logs/dagger_gen_s1_${SLURM_JOB_ID}_$S.log 2>&1 &
done
wait
cat dagger_windows/train_slide1/*.txt > data/dagger_slide1.txt
NW=$(grep -c . data/dagger_slide1.txt)
echo "DAgger slide-1 windows: $NW"
[ "$NW" -ge 10000 ] || { echo "too few windows"; exit 1; }
awk 'NF != 1020 {print "BAD LINE", NR, NF; exit 1}' data/dagger_slide1.txt
python -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
    --data_file data/dagger_clean_subset6k.txt \
    --dagger_file data/dagger_slide1.txt \
    --val_file data/val_paper.txt \
    --model_name $CKPT \
    --output_dir ./run_paper_split_v2_dagger_slide1 \
    --wandb_run_name run_paper_split_v2_dagger_slide1 \
    --wandb_mode online \
    --loss_mask_first_score_slots 137 \
    --max_steps 8000 \
    --batch_size 8 \
    --val_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --save_steps 1000 \
    --eval_steps 500
echo DAGGER_DONE
