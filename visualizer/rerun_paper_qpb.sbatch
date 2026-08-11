#!/bin/bash
#SBATCH --job-name=rerun-qpb
#SBATCH --partition=ellis,thickstun
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=48G
#SBATCH --time=4:00:00
#SBATCH --output=logs/rerun_qpb_%j.out
#SBATCH --error=logs/rerun_qpb_%j.out

# Re-run both paper models after fixing quarters_per_annotated_beat to MEASURE the
# ASAP beat convention from midi_score_annotations.txt instead of inferring it from
# the meter. The old rule mis-read Chopin's 6/4 Ballade as 1.0 instead of 3.0, which
# both skipped windows spuriously AND silently mis-scaled ones it accepted.
#
# Two passes: the guarded one becomes the visualizer's data.js; the unguarded one
# gives the all-windows numbers that are actually comparable against our models.
set -uo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false
echo "Host: $(hostname)"; echo "Start: $(date)"

cp visualizer/data.js visualizer/data.js.prev_qpb
mkdir -p debug
python - <<'PY'
import json, pathlib
txt = pathlib.Path("visualizer/data.js").read_text()
p = json.loads(txt[txt.index("{"):txt.rindex("}")+1]); pre = txt[:txt.index("{")]
for ex in p["examples"].values():
    for g in ("rollouts_paper1", "rollouts_paper2"):
        ex.pop(g, None)
for out in ("visualizer/data.js", "debug/span_all.js"):
    with open(out, "w") as fh:
        fh.write(pre); json.dump(p, fh); fh.write(";\n")
print("cleared previous paper rollouts")
PY

conda activate paperpipe
echo "== [1/3] guarded pass -> visualizer/data.js =="
python -u visualizer/run_paper_models.py --data visualizer/data.js \
    --manifest visualizer/paper_windows.json --debug
echo "== [2/3] unguarded pass -> debug/span_all.js =="
python -u visualizer/run_paper_models.py --data debug/span_all.js \
    --manifest visualizer/paper_windows.json --no-span-guard
conda activate base

echo "== [3/3] F1 =="
echo "--- GUARDED (what the visualizer shows) ---"
python visualizer/compute_f1.py --data visualizer/data.js
echo "--- ALL WINDOWS (the comparable numbers) ---"
python visualizer/compute_f1.py --data debug/span_all.js --dry-run
echo "Done. End: $(date)"
