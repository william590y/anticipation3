"""Pitch-forcing diagnostic table (test; val appended when its eval lands)."""
import json, subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
SRC = [("test", "pitch_forcing_eval.json"), ("val", "pitch_forcing_eval_val.json")]
rows_out = []
for split, fn in SRC:
    p = ROOT / "nbest_data" / fn
    if not p.exists():
        continue
    recs = json.load(open(p))["records"]
    byw = defaultdict(list)
    for r in recs:
        if r.get("work"):
            byw[r["work"]].append(r)
    for crit, lab in (("onset_pitch", "exact"), ("onset_pitch_dur", "+dur"),
                      ("onset_pitch_tol1", r"$\pm1$")):
        g = 100*np.mean([r[f"greedy_{crit}"] for r in recs])
        f = 100*np.mean([r[f"forced_{crit}"] for r in recs])
        pg = 100*np.mean([np.mean([r[f"greedy_{crit}"] for r in v]) for v in byw.values()])
        pf = 100*np.mean([np.mean([r[f"forced_{crit}"] for r in v]) for v in byw.values()])
        rows_out.append(f"{split} & {lab} & {g:.2f} & {f:.2f} & {f-g:+.2f} & "
                        f"{pg:.2f} & {pf:.2f} & {pf-pg:+.2f} \\\\")
    fr = 100*np.mean([r["forced_frac"] for r in recs])
    rows_out.append(f"\\multicolumn{{8}}{{l}}{{\\emph{{{split}: "
                    f"{fr:.2f}\\% of pitch slots forced, "
                    f"{len(recs)} windows / {len(byw)} works}}}}\\\\ \\midrule")
body = ([r"\begin{tabular}{ll cccccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{window level} & \multicolumn{3}{c}{piece level} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"split & criterion & greedy & forced & $\Delta$ & greedy & forced & $\Delta$ \\",
         r"\midrule"] + rows_out)[:-1]
body += [rows_out[-1].replace(" \\midrule", ""), r"\bottomrule", r"\end{tabular}"]
DOC = ("\\documentclass[border=6pt]{standalone}\n\\usepackage{booktabs}"
       "\\usepackage{amsmath}\n\\begin{document}%s\\end{document}")
(HERE / "table_pitch_forcing.tex").write_text(DOC % "\n".join(body))
subprocess.run(["tectonic", "table_pitch_forcing.tex"], cwd=HERE, check=True,
               capture_output=True)
subprocess.run(["convert", "-density", "300", str(HERE/"table_pitch_forcing.pdf"),
                "-background", "white", "-alpha", "remove",
                str(HERE/"table_pitch_forcing.png")], check=True)
print("wrote table_pitch_forcing.{tex,pdf,png}")
