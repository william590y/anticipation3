#!/bin/bash
#SBATCH --job-name=viz-valloss-m
#SBATCH --partition=thickstun
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/viz_valloss_merge_%j.out
#SBATCH --error=logs/viz_valloss_merge_%j.out

set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

SHARDS=(visualizer/valloss_shards/shard_{00..06}.json)
for s in "${SHARDS[@]}"; do
  [[ -f "$s" ]] || { echo "MISSING $s" >&2; exit 1; }
done

echo "== merge rollouts =="
python visualizer/merge_valloss_rollouts.py --data visualizer/data.js --shards "${SHARDS[@]}"

echo "== F1 =="
python visualizer/compute_f1.py --data visualizer/data.js

STAMP=$(date +%Y%m%d%H%M)
sed -i -E "s|data\\.js\\?v=[^\"]+|data.js?v=${STAMP}|" visualizer/visualizer.html || true

python - <<'PY'
import json
from pathlib import Path
text = Path('visualizer/data.js').read_text()
data = json.loads(text[len('window.VISUALIZER_DATA = '):].rstrip().rstrip(';'))
missing = []
for eid in data['example_order']:
    ex = data['examples'][eid]
    for g in ('rollouts_valloss', 'rollouts_lora_valloss'):
        block = ex.get(g)
        if not block or 'raw' not in block or 'pred_score' not in block['raw']:
            missing.append(f'{eid}.{g}')
            continue
        if 'f1' not in block['raw']:
            missing.append(f'{eid}.{g}.f1')
print('missing', len(missing), missing[:8])
print('checkpoint_sets', data.get('checkpoint_sets'))
if missing:
    raise SystemExit(1)
print('OK')
PY
echo "Merge complete: $(date)"
echo "Next: sbatch visualizer/compute_sequence_ppl.sbatch  # scores entropy/seq PPL for new groups"
