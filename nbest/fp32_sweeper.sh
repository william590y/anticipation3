#!/bin/bash
# Keep resubmitting fp32 N-best shards until every output file exists.
#
# Three independent flakiness sources make a single array submission
# unreliable here, and this loop is immune to all of them:
#   * shared-partition PREEMPTION (we are guests on default_partition; two
#     jobs were preempted mid-run earlier tonight);
#   * bad GPU nodes -- `Failed to get device handle for GPU 0` kills a task
#     during the VRAM probe before any work happens (array 470308 lost 4);
#   * sacct on this cluster serves records for REUSED job IDs, so job state
#     cannot be trusted. Only squeue (while queued) and the output files on
#     disk are authoritative, and this script keys entirely off the files.
#
# generate_nbest.py checkpoints a resumable `.partial` every 25 batches and
# is invoked with --resume, so a resubmitted task continues rather than
# restarting -- an interrupted 8-hour shard loses minutes, not hours.
set -uo pipefail
cd /home/wjl86/anticipation3

MAX_PASSES=${MAX_PASSES:-12}
CONSTRAINT=${CONSTRAINT:-"a6000|6000ada|a100|a40"}
CONCURRENT=${CONCURRENT:-5}

outfile() {   # task index -> expected output path
  case $1 in
    0|1|2) echo "nbest_data/fp32_9_train_shard0$1.pt" ;;
    3)     echo "nbest_data/fp32_9_val_shard00.pt" ;;
    10)    echo "nbest_data/fp32_32_val_shard00.pt" ;;
    *)     echo "nbest_data/fp32_32_train_shard0$(( $1 - 4 )).pt" ;;
  esac
}

for pass in $(seq 1 "$MAX_PASSES"); do
  missing=()
  for t in $(seq 0 10); do
    [ -f "$(outfile "$t")" ] || missing+=("$t")
  done
  if [ ${#missing[@]} -eq 0 ]; then
    echo "[$(date -u +%H:%M:%S)] all 11 shards present; sweeper done"
    exit 0
  fi
  list=$(IFS=,; echo "${missing[*]}")
  echo "[$(date -u +%H:%M:%S)] pass ${pass}: ${#missing[@]} missing -> array ${list}"
  jid=$(sbatch --parsable --array="${list}%${CONCURRENT}" --gres=gpu:1 \
        --constraint="$CONSTRAINT" nbest/generate_fp32.slurm)
  echo "[$(date -u +%H:%M:%S)] submitted $jid"
  # squeue is the only trustworthy state source (see header)
  while squeue -j "$jid" -h 2>/dev/null | grep -q .; do sleep 300; done
  echo "[$(date -u +%H:%M:%S)] array $jid left the queue"
done

echo "[$(date -u +%H:%M:%S)] hit MAX_PASSES=${MAX_PASSES} with shards still missing:"
for t in $(seq 0 10); do [ -f "$(outfile "$t")" ] || echo "  task $t -> $(outfile "$t")"; done
exit 1
