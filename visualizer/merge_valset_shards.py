#!/usr/bin/env python
"""Fold the sharded paper rollouts back into data_valset.js."""
import json, glob
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
t = (ROOT / "visualizer/data_valset.js").read_text(encoding="utf-8")
d = json.loads(t[t.index("{"): t.rindex("}") + 1])
n = 0
for f in sorted(glob.glob(str(ROOT / "visualizer/data_valset_shard*.js"))):
    s = Path(f).read_text(encoding="utf-8")
    sd = json.loads(s[s.index("{"): s.rindex("}") + 1])
    for k, ex in sd["examples"].items():
        for g in ("rollouts_paper1", "rollouts_paper2"):
            if g in ex:
                d["examples"][k][g] = ex[g]
                n += 1
(ROOT / "visualizer/data_valset.js").write_text(
    "window.VISUALIZER_DATA = " + json.dumps(d) + ";", encoding="utf-8")
have = sum(1 for k in d["example_order"]
           if "rollouts_paper2" in d["examples"][k])
print(f"merged {n} group entries; {have}/{len(d['example_order'])} windows have beyer")
