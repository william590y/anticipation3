#!/bin/bash
#SBATCH --job-name=viz-vl-final
#SBATCH --partition=thickstun
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz_valloss_final_%j.out
#SBATCH --error=logs/viz_valloss_final_%j.out

set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

# Prefer fresh ppl shards if present; otherwise data.js was written in-place.
if ls visualizer/ppl_shards/shard_0*.json >/dev/null 2>&1; then
  python visualizer/merge_ppl_shards.py --data visualizer/data.js \
    --shards visualizer/ppl_shards/shard_{00..06}.json || true
fi

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
        block = ex.get(g) or {}
        raw = block.get('raw')
        if not raw or 'pred_score' not in raw:
            missing.append(f'{eid}.{g}.raw')
            continue
        for need in ('f1', 'sequence_perplexity', 'entropy', 'perplexity'):
            if not raw.get(need):
                missing.append(f'{eid}.{g}.raw.{need}')
        beams = raw.get('beams') or {}
        if '5' not in beams:
            missing.append(f'{eid}.{g}.raw.beams[5]')
        elif not beams['5'].get('perplexity'):
            missing.append(f'{eid}.{g}.raw.beams[5].perplexity')
print('missing', len(missing), missing[:12])
print('checkpoint_sets', data.get('checkpoint_sets'))
if missing:
    raise SystemExit(1)
print('OK')
PY
echo "Final verify done: $(date)"
