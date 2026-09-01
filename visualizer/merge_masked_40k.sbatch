#!/bin/bash
#SBATCH --job-name=viz-m40k-merge
#SBATCH --partition=thickstun
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=logs/viz_masked_40k_merge_%j.out
#SBATCH --error=logs/viz_masked_40k_merge_%j.out

# CPU merge of 40k masked shards into data.js + F1 + cache-bust.
# Submit with --dependency=afterok:<3090>:<a6000>.
set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base

DATA="${DATA:-visualizer/data.js}"
OUT_DIR="${OUT_DIR:-visualizer/masked_40k_shards}"
WORLD_SIZE="${WORLD_SIZE:-5}"

export PYTHONUNBUFFERED=1
echo "Host: $(hostname)"
echo "Start: $(date)"

shards=()
missing=0
for rank in $(seq 0 $((WORLD_SIZE - 1))); do
  f="${OUT_DIR}/shard_$(printf '%02d' "$rank").json"
  if [[ ! -f "$f" ]]; then
    echo "missing shard $f" >&2
    missing=1
  else
    shards+=("$f")
  fi
done
if [[ "$missing" -ne 0 ]]; then
  exit 1
fi

echo "== merge =="
python visualizer/merge_masked_40k_rollouts.py --data "$DATA" --shards "${shards[@]}"

echo "== F1 =="
python visualizer/compute_f1.py --data "$DATA"

STAMP=$(date +%Y%m%d%H%M)
sed -i -E "s|data\\.js\\?v=[^\"]+|data.js?v=${STAMP}|" visualizer/visualizer.html || true
echo "cache-bust data.js?v=${STAMP}"

python - <<'PY'
import json
from pathlib import Path

text = Path("visualizer/data.js").read_text()
data = json.loads(text[text.index("{"): text.rindex("}") + 1])
order = data["example_order"]
missing = []
for group in ("rollouts_masked_40k", "rollouts_masked_40k_final"):
    for eid in order:
        block = data["examples"][eid].get(group) or {}
        raw = block.get("raw") or {}
        if "pred_score" not in raw:
            missing.append(f"{eid}.{group}.raw")
            continue
        if "f1" not in raw:
            missing.append(f"{eid}.{group}.raw.f1")
    # 20k key must still be present
for eid in order:
    if "rollouts_masked" not in data["examples"][eid]:
        missing.append(f"{eid}.rollouts_masked.MISSING")
print("missing", len(missing), missing[:12])
print("checkpoint_masked", data.get("checkpoint_masked"))
print("checkpoint_masked_40k", data.get("checkpoint_masked_40k"))
print("checkpoint_masked_40k_final", data.get("checkpoint_masked_40k_final"))
if missing:
    raise SystemExit(1)
print("OK")
PY

echo "Done: $(date)"
