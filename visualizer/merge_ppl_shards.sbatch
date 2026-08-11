#!/bin/bash
#SBATCH --job-name=viz-ppl-mrg
#SBATCH --partition=thickstun
#SBATCH --account=thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/viz_ppl_merge_%j.out
#SBATCH --error=logs/viz_ppl_merge_%j.out

set -euo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
conda activate base
export PYTHONUNBUFFERED=1

SHARDS=(visualizer/ppl_shards/shard_{00..06}.json)
for s in "${SHARDS[@]}"; do
  [[ -f "$s" ]] || { echo "MISSING $s" >&2; exit 1; }
done

python visualizer/merge_ppl_shards.py --data visualizer/data.js --shards "${SHARDS[@]}"
STAMP=$(date +%Y%m%d%H%M)
sed -i -E "s|data\\.js\\?v=[^\"]+|data.js?v=${STAMP}|" visualizer/visualizer.html || true

python - <<'PY'
import json
from pathlib import Path
text = Path('visualizer/data.js').read_text()
body = text[len('window.VISUALIZER_DATA = '):].rstrip().rstrip(';')
data = json.loads(body)
n_beam_ppl = 0
missing = []
for eid in data['example_order']:
    for group in ('rollouts', 'rollouts_lora'):
        roll = (data['examples'][eid].get(group) or {}).get('raw')
        if not roll:
            continue
        beams = roll.get('beams') or {}
        for w, entry in beams.items():
            if entry.get('perplexity'):
                n_beam_ppl += 1
            else:
                missing.append(f'{eid}.{group}.raw.beams[{w}]')
print('beam entries with perplexity:', n_beam_ppl)
print('missing beam perplexity:', len(missing), missing[:8])
if missing:
    raise SystemExit(1)
print('OK')
PY
echo "PPL merge complete: $(date)"
