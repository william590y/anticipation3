#!/bin/bash
#SBATCH --job-name=viz-bmfull-m
#SBATCH --partition=thickstun
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz_bmfull_merge_%j.out
#SBATCH --error=logs/viz_bmfull_merge_%j.out

set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

SHARDS=(visualizer/beam_shards/full_shard_{00..07}.json)
for s in "${SHARDS[@]}"; do
  [[ -f "$s" ]] || { echo "MISSING $s" >&2; exit 1; }
done

python visualizer/merge_beam_shards.py --data visualizer/data.js --shards "${SHARDS[@]}"
STAMP=$(date +%Y%m%d%H%M)
sed -i -E "s|data\\.js\\?v=[^\"]+|data.js?v=${STAMP}|" visualizer/visualizer.html || true

python - <<'PY'
import json
from pathlib import Path
text = Path('visualizer/data.js').read_text()
body = text[len('window.VISUALIZER_DATA = '):].rstrip().rstrip(';')
data = json.loads(body)
missing = []
for eid in data['example_order']:
    ex = data['examples'][eid]
    for group in ('rollouts', 'rollouts_lora'):
        block = ex.get(group) or {}
        for variant in ('filtered', 'filtered_seeded', 'raw', 'raw_seeded'):
            roll = block.get(variant)
            if not roll:
                continue
            if 'beams' not in roll or set(roll['beams']) != {str(i) for i in range(1, 11)}:
                missing.append(f'{eid}.{group}.{variant}')
roll = data['examples']['val-01']['rollouts']['filtered']
g, b = roll['pred_score'], roll['beams']['1']['pred_score']
mm = sum(1 for i in range(min(len(g), len(b))) if (g[i] or {}).get('p') != (b[i] or {}).get('p'))
print('missing', len(missing), missing[:10])
print('val-01 beam1 pitch mismatches', mm)
if missing:
    raise SystemExit(1)
print('OK')
PY
echo "FULL merge complete: $(date)"
