"""Task 3: LaTeX table for the whole-song (sliding-window rollout) comparison.

Reads nbest_data/fullsong_eval.json (test, or test+val when the val eval has
run) and renders table_fullsong.{tex,pdf,png} via tectonic -- same pipeline as
the windowed tables. Bold = best per row; sign-flip p at work level vs ours.
"""
import json, subprocess
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).parent
recs = json.load(open(ROOT / "nbest_data/fullsong_eval.json"))["records"]
METS = [("base", r"tol.\ $\pm1$"), ("smax", "scale-max"),
        ("ioi", "interarrival"), ("ioi_smax", "IOI scale-max")]
MODELS = [("ours", "Ours (slide)"), ("beyer", "Beyer \\& Dai"), ("zeng", "Zeng+")]

def sf(dv, n=200000, seed=0):
    rng = np.random.default_rng(seed)
    null = np.abs((rng.choice((-1., 1.), size=(n, dv.size)) * dv).mean(1))
    return float((np.count_nonzero(null >= abs(dv.mean()) - 1e-15) + 1) / (n + 1))

def block(rows, scope):
    byw = defaultdict(list)
    for r in rows:
        byw[r["work"]].append(r)
    out = [f"\\multicolumn{{9}}{{l}}{{\\emph{{{scope}: "
           f"{len(rows)} performances, {len(byw)} works}}}}\\\\"]
    for k, lab in METS:
        perf = [100*np.mean([r[f"{m}_{k}"] for r in rows]) for m, _ in MODELS]
        pw = {m: np.array([np.mean([r[f"{m}_{k}"] for r in g])
                           for g in byw.values()]) for m, _ in MODELS}
        pwv = [100*pw[m].mean() for m, _ in MODELS]
        def bold(vs):
            i = int(np.argmax(vs))
            return [f"\\textbf{{{v:.2f}}}" if j == i else f"{v:.2f}"
                    for j, v in enumerate(vs)]
        pb = sf(pw["beyer"] - pw["ours"]); pz = sf(pw["zeng"] - pw["ours"])
        out.append(f"{lab} & " + " & ".join(bold(perf)) + " & "
                   + " & ".join(bold(pwv)) + f" & {pb:.3f} & {pz:.3f} \\\\")
    return out

splits = sorted({r["split"] for r in recs})
body = [r"\begin{tabular}{l ccc ccc cc}", r"\toprule",
        r" & \multicolumn{3}{c}{per performance} & \multicolumn{3}{c}{per work} & & \\",
        r"\cmidrule(lr){2-4}\cmidrule(lr){5-7}",
        "metric & " + " & ".join(l for _, l in MODELS) + " & "
        + " & ".join(l for _, l in MODELS)
        + r" & $p_{\mathrm{B-O}}$ & $p_{\mathrm{Z-O}}$ \\", r"\midrule"]
for sp in splits:
    body += block([r for r in recs if r["split"] == sp], sp)
    body.append(r"\midrule")
if len(splits) > 1:
    body += block(recs, "test+val")
    body.append(r"\midrule")
body = body[:-1] + [r"\bottomrule", r"\end{tabular}"]
DOC = ("\\documentclass[border=6pt]{standalone}\n\\usepackage{booktabs}"
       "\\usepackage{amsmath}\n\\begin{document}%s\\end{document}")
tex = HERE / "table_fullsong.tex"
tex.write_text(DOC % "\n".join(body))
subprocess.run(["tectonic", str(tex)], cwd=HERE, check=True, capture_output=True)
subprocess.run(["convert", "-density", "300", str(HERE/"table_fullsong.pdf"),
                "-background", "white", "-alpha", "remove",
                str(HERE/"table_fullsong.png")], check=True)
print("wrote table_fullsong.{tex,pdf,png}  (splits: " + ", ".join(splits) + ")")
