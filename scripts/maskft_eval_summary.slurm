#!/bin/bash
#SBATCH -J mfteval-sum
#SBATCH -p thickstun
#SBATCH -A thickstun
#SBATCH -c 16
#SBATCH --mem=32G
#SBATCH -t 2:00:00
#SBATCH -o /home/wjl86/anticipation3/logs/mfteval_sum_%j.out
# FILE-GATED summary (afterany-chained; never DependencyNeverSatisfied):
# segmented slide/reset comparison of a fine-tuned checkpoint's whole-song
# rollouts against the baseline arms. RUNDIR env selects the arm.
set -uo pipefail
cd /home/wjl86/anticipation3
source /home/wjl86/miniconda3/etc/profile.d/conda.sh; conda activate base
D="${OUTDIR:?}"
nt=$(ls "$D/test/slide" 2>/dev/null | wc -l); nr=$(ls "$D/test/reset" 2>/dev/null | wc -l)
echo "rollouts present: slide $nt/59, reset $nr/59"
[ "$nt" -lt 50 ] && { echo "INCOMPLETE -- rerun the decode arrays"; exit 1; }
python3 - <<PY
import json, sys, importlib.util
import numpy as np
from pathlib import Path
sys.path.insert(0, ".")
spec = importlib.util.spec_from_file_location("cf1", "visualizer/compute_f1.py")
cf1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(cf1)
def tol1(p, g): return cf1.score_notes(p, g)["onset_pitch_tol1"]["f1"]
def norm(ns):
    m = min(n["t"] for n in ns)
    return [{"t": n["t"]-m, "d": n["d"], "p": n["p"]} for n in ns]
SEG = 138
import os
D = os.environ["OUTDIR"]
print(f"=== {D} vs baseline (run_paper_split_v2/ckpt-2500) ===")
for arm, base_dir in (("slide", "fullsong_rollouts/test/slide"),
                      ("reset", "fullsong_rollouts/test/reset")):
    for tag, dd in (("baseline", base_dir), ("MASKFT", f"{D}/test/{arm}")):
        files = sorted(Path(dd).glob("*.json"))
        if not files: continue
        curve, whole = {}, []
        for f in files:
            r = json.load(open(f))
            pred, gt = norm(r["pred"]), norm(r["gt"])
            n = min(len(pred), len(gt))
            whole.append(tol1(pred[:n], gt[:n]))
            for si, s in enumerate(range(0, n - SEG + 1, SEG)):
                if si > 6: break
                curve.setdefault(si, []).append(
                    tol1(norm(pred[s:s+SEG]), norm(gt[s:s+SEG])))
        segs = "  ".join(f"{si}:{100*np.mean(v):.1f}" for si, v in sorted(curve.items()))
        print(f"{arm:6} {tag:9} whole={100*np.mean(whole):5.2f}%  segs {segs}")
print("reference oracle band (GT context): 42-68% per continued segment")
PY
echo SUMMARY_OK
