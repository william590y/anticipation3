#!/bin/bash
# Preemption watchdog for the music-eval fleet. Two reflexes, every 2 min:
#   NUDGE      requeued jobs parked on Reason=BeginTime -> BeginTime=now
#   RESURRECT  a job FAMILY with incomplete outputs and no queued/running
#              member -> resubmit (all completeness checks are FILES ON DISK,
#              per this repo's rule that squeue+files are the only truth).
# Per-family resubmit cap of 4, tracked in $STATE, so a broken family cannot
# storm the scheduler. Emits one line per ACTION; silent otherwise.
set -u
cd /home/wjl86/anticipation3
STATE=/tmp/claude-1685572/-home-wjl86/972baae3-a147-431f-9d41-dd096a9f2831/scratchpad/watchdog_state
mkdir -p "$STATE"

cap_ok() { local n; n=$(cat "$STATE/$1" 2>/dev/null || echo 0); [ "$n" -lt 4 ]; }
bump()   { local n; n=$(cat "$STATE/$1" 2>/dev/null || echo 0); echo $((n+1)) > "$STATE/$1"; }
have_job() { squeue -h -u wjl86 -n "$1" -o "%T" 2>/dev/null | grep -qE "RUNNING|PENDING|CONFIGURING"; }

while true; do
  # ---- NUDGE ----------------------------------------------------------------
  while read -r jid reason; do
    [ "$reason" = "(BeginTime)" ] || continue
    scontrol update JobId="${jid%%_*}" BeginTime=now 2>/dev/null \
      && echo "NUDGED $jid out of BeginTime backoff"
  done < <(squeue -r -h -u wjl86 -o "%i %R" 2>/dev/null)

  # ---- RESURRECT ------------------------------------------------------------
  # papers-on-val-windows shards (target: 12 OK markers)
  ok=$(grep -l PAPERVALS_OK logs/papervalS_*.out 2>/dev/null | wc -l)
  if [ "$ok" -lt 12 ] && ! have_job papervalS && cap_ok papervalS; then
    J=$(sbatch --parsable scripts/paperval_shard.slurm) && bump papervalS \
      && echo "RESURRECTED papervalS as $J ($ok/12 done)"
  fi
  # plain val rollouts (86 files)
  n=$(ls fullsong_rollouts/val/slide 2>/dev/null | wc -l)
  if [ "$n" -lt 86 ] && ! have_job fs-ours && cap_ok fsval; then
    J=$(sbatch --parsable --export=ALL,SPLIT=val scripts/fullsong_ours.slurm) \
      && bump fsval && echo "RESURRECTED fs-ours val as $J ($n/86)"
  fi
  # pitch-forced rollouts (59 + 86 files)
  nt=$(ls fullsong_rollouts/test/slide_pforce 2>/dev/null | wc -l)
  nv=$(ls fullsong_rollouts/val/slide_pforce 2>/dev/null | wc -l)
  if { [ "$nt" -lt 59 ] || [ "$nv" -lt 86 ]; } && ! have_job fs-ours && cap_ok fspf; then
    J1=$(sbatch --parsable --export=ALL,SPLIT=test,PFORCE=1 scripts/fullsong_ours.slurm)
    J2=$(sbatch --parsable --export=ALL,SPLIT=val,PFORCE=1 scripts/fullsong_ours.slurm)
    bump fspf && echo "RESURRECTED pforce rollouts as $J1/$J2 (test $nt/59, val $nv/86)"
  fi
  # pforce token reruns + viz24 (single-file targets)
  if [ ! -s nbest_data/pitch_forced_tokens.pt ] && ! have_job pforce && cap_ok pft; then
    J=$(sbatch --parsable scripts/pitch_forcing.slurm) && bump pft \
      && echo "RESURRECTED pforce (test tokens) as $J"
  fi
  if [ ! -s nbest_data/pitch_forced_tokens_val.pt ] && ! have_job pforce-val && cap_ok pfv; then
    J=$(sbatch --parsable scripts/pitch_forcing_val.slurm) && bump pfv \
      && echo "RESURRECTED pforce-val as $J"
  fi
  if ! ls visualizer/rerank_feat_shards/pitch_forced.json >/dev/null 2>&1 \
      && ! have_job pf-viz24 && cap_ok viz24; then
    J=$(sbatch --parsable scripts/pforce_viz24.slurm) && bump viz24 \
      && echo "RESURRECTED pf-viz24 as $J"
  fi
  sleep 120
done
