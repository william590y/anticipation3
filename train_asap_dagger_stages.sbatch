#!/bin/bash
#SBATCH --job-name=dag-stages
#SBATCH --partition=thickstun
#SBATCH --account=thickstun
#SBATCH --gres=gpu:nvidia_rtx_6000_ada_generation:2
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=/home/wjl86/anticipation3/logs/dagger_stages_%j.out
#
# MULTI-STAGE DAgger (slide-69) for ONE mask-dropout model (env MODEL =
# mask25|mask50|mask75). Three proper DAgger rounds:
#   round r:  roll out under the CURRENT policy (r=1: the mask checkpoint
#             itself; r>1: round r-1's model) over the train split, harvest
#             every re-pack boundary's self-context window, complete it with
#             aligned GT targets, aggregate with all previous rounds, and
#             RETRAIN FROM THE MASK CHECKPOINT on clean u D_1..D_r.
# Retraining from the fixed init each round (rather than continuing) is what
# makes this DAgger and not sequential fine-tuning: the policy is refit on the
# whole aggregate, so early-round states keep their weight.
#
# The mask75 arm already has r1-r3 (whole-song slide 3.97 -> 5.49 -> 6.83 ->
# 6.05, converging); this runs the same ladder for 0.25 and 0.50 so the
# dropout rate x DAgger interaction is measurable rather than assumed.
#
# 150 perfs/round (half the original 300) keeps a 3-round ladder inside the
# 48h limit: ~4h generation + ~3h training per round.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 NCCL_P2P_DISABLE=1 TORCH_NCCL_ASYNC_ERROR_HANDLING=1
python -c "import torch;assert torch.cuda.device_count()==2;print('preflight OK')"

MODEL=${MODEL:?set MODEL=mask25|mask50|mask75}
case "$MODEL" in
  mask25) BASE=run_paper_split_v2_maskft25/checkpoint-10000 ;;
  mask50) BASE=run_paper_split_v2_maskft50/checkpoint-10000 ;;
  mask75) BASE=run_paper_split_v2_maskft/checkpoint-10000 ;;
  *) echo "unknown MODEL=$MODEL"; exit 1 ;;
esac
[ -d "$BASE" ] || { echo "missing base checkpoint $BASE"; exit 1; }
echo "=== multi-stage DAgger for $MODEL (base=$BASE) ==="

SRC=$BASE
AGG=""
for R in 1 2 3; do
  GENDIR=dagger_windows/${MODEL}_r${R}
  echo "=== round $R: generating under $SRC ==="
  for S in 0 1; do
    CUDA_VISIBLE_DEVICES=$S python dagger_gen.py --checkpoint "$SRC" \
      --out-dir "$GENDIR" --max-perfs 150 --window-overlap 69 \
      --shard-index $S --num-shards 2 \
      > logs/dagger_gen_${MODEL}_r${R}_${SLURM_JOB_ID}_$S.log 2>&1 &
  done
  wait
  cat "$GENDIR"/*.txt > data/dagger_${MODEL}_r${R}.txt
  AGG="$AGG data/dagger_${MODEL}_r${R}.txt"
  cat $AGG > data/dagger_${MODEL}_agg${R}.txt
  NW=$(grep -c . data/dagger_${MODEL}_agg${R}.txt)
  echo "=== round $R: aggregate D_1..D_${R} = $NW windows ==="
  [ "$NW" -ge 5000 ] || { echo "too few windows"; exit 1; }
  awk 'NF != 1020 {print "BAD LINE", NR, NF; exit 1}' data/dagger_${MODEL}_r${R}.txt
  OUT=run_${MODEL}_dagger_v${R}
  python -m torch.distributed.run --standalone --nproc_per_node=2 train.py \
      --data_file data/dagger_clean_subset36k.txt \
      --dagger_file data/dagger_${MODEL}_agg${R}.txt \
      --val_file data/val_paper.txt \
      --model_name "$BASE" \
      --output_dir ./$OUT \
      --wandb_run_name $OUT \
      --wandb_mode online \
      --loss_mask_first_score_slots 69 \
      --max_steps 5000 \
      --batch_size 16 \
      --val_batch_size 16 \
      --gradient_accumulation_steps 2 \
      --save_steps 1000 \
      --eval_steps 500
  echo "=== round $R done -> $OUT ==="
  SRC=$OUT/checkpoint-5000
done
echo DAGGER_STAGES_DONE
