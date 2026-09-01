#!/bin/bash
# FILE-GATED driver for the fp32 table. Replaces the afterok chain, which has
# now been killed TWICE by things that have nothing to do with the work:
# once by a dead GPU on `badfellow`, once by tier-10 preemption. A partially
# failed ARRAY leaves every afterok dependent in DependencyNeverSatisfied
# FOREVER -- so the chain must key off files on disk, exactly as
# nbest/fp32_sweeper.sh already does for the shard generation.
#
# Partition list puts the tier-20 partitions (thickstun, ellis, PriorityTier=20)
# ahead of default_partition (tier 10, where we were preempted); --requeue lets
# a preemption resume instead of failing.
set -uo pipefail
cd /home/wjl86/anticipation3
P="-p thickstun,ellis,default_partition -A thickstun --requeue"
X="--exclude=ma-compute-02,kuleshov-compute-01,badfellow"
log(){ echo "[$(date +%H:%M:%S)] $*"; }

missing_feat(){ local m=(); for t in 0 1 2 3 4 5; do
    [ -s "nbest_data/fp32feat_train_shard0${t}.pt" ] || m+=("$t"); done
  [ -s "nbest_data/fp32feat_val_shard00.pt" ] || m+=("6"); echo "${m[*]}"; }

# ---- stage 1: features, resubmit until every output file exists -------------
for pass in 1 2 3 4 5; do
  M=$(missing_feat); [ -z "$M" ] && { log "stage1 complete"; break; }
  A=$(echo "$M" | tr ' ' ','); log "stage1 pass $pass: submitting tasks $A"
  J=$(sbatch --parsable $P $X --array="$A%3" nbest/fp32_tokfeat.slurm)
  log "  job $J"; while squeue -j "$J" -h -o "%T" 2>/dev/null | grep -q .; do sleep 120; done
  log "  pass $pass done; still missing: [$(missing_feat)]"
done
[ -n "$(missing_feat)" ] && { log "STAGE 1 FAILED after 5 passes: [$(missing_feat)]"; exit 1; }

# ---- stage 2: the two feature trainers + pointwise/fit, in parallel ---------
J2=$(sbatch --parsable $P $X nbest/fp32_train_feat.slurm)
J3=$(sbatch --parsable $P $X nbest/fp32_pointwise_and_fit.slurm)
log "stage2: featTrainers=$J2 pointwise+fit=$J3"
while squeue -j "$J2,$J3" -h -o "%T" 2>/dev/null | grep -q .; do sleep 120; done

for f in run_nbest_reranker/pairwise32feat_fp32/final.pt \
         run_nbest_reranker/duel32_fp32/final.pt \
         run_nbest_reranker/pointwise_fp32/final.pt \
         nbest_data/decode_weights_fp32.json; do
  [ -s "$f" ] || { log "STAGE 2 INCOMPLETE, missing $f"; exit 1; }
done
log "stage2 complete"

# ---- stage 3: decode + merge + table ---------------------------------------
J4=$(sbatch --parsable $P $X visualizer/fp32_table_decode.slurm)
log "stage3: table job $J4"
while squeue -j "$J4" -h -o "%T" 2>/dev/null | grep -q .; do sleep 120; done
log "DRIVER DONE"; tail -40 logs/fp32table_${J4}.out 2>/dev/null
