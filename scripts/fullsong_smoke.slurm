#!/bin/bash
#SBATCH -J fs-smoke
#SBATCH -p ellis,default_partition
#SBATCH -A thickstun
#SBATCH --gres=gpu:1
#SBATCH --constraint="a6000|6000ada|a100|a40|3090"
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 1:00:00
#SBATCH -o /home/wjl86/anticipation3/logs/fs_smoke_%j.out
# SMOKE for the slide mode before committing 145 performances to it:
# one performance, capped at 400 notes (crosses >=2 window boundaries at
# overlap 69), run in BOTH modes, printing boundary counts and note counts.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
export PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1
python3 -c "import torch;t=torch.cuda.mem_get_info()[1]/2**30;n=torch.cuda.get_device_name(0);print(f'{n} {t:.1f}GiB');assert t>=10,f'refusing {n}'"
for MODE in slide reset; do
  python3 -u full_song_rollout.py --split test --window-mode $MODE \
      --max-notes 400 --num-shards 59 --shard-index 3 \
      --out-dir fullsong_smoke
done
echo "=== compare the two modes on the smoke piece ==="
python3 - <<'PY'
import json, glob
for mode in ("slide", "reset"):
    f = glob.glob(f"fullsong_smoke/test/{mode}/*.json")[0]
    r = json.load(open(f))
    pred, gt = r["pred"], r["gt"]
    on = [n["t"] for n in pred]
    mono_viol = sum(1 for i in range(1, len(on)) if on[i] < on[i-1] - 200)
    print(f"{mode:6} pred={len(pred)} boundaries={r['gen_stats']['num_window_resets']} "
          f"onset range=[{min(on)},{max(on)}] big-backjumps(>2s)={mono_viol}")
PY
echo SMOKE_OK
