#!/usr/bin/env python
"""Split data_valset.js by PERFORMANCE into K sub-payloads so run_paper_models
can shard across GPUs (its per-piece cache means windows of one performance
must stay in one shard), then merge_valset_shards.py folds the paper rollouts
back into the master payload."""
import json, sys
from collections import defaultdict
from pathlib import Path
K = int(sys.argv[1]) if len(sys.argv) > 1 else 12
ROOT = Path(__file__).resolve().parents[1]
t = (ROOT / "visualizer/data_valset.js").read_text(encoding="utf-8")
d = json.loads(t[t.index("{"): t.rindex("}") + 1])
byperf = defaultdict(list)
for k in d["example_order"]:
    byperf[d["examples"][k]["piece"]].append(k)
perfs = sorted(byperf)
for i in range(K):
    keys = [k for p in perfs[i::K] for k in byperf[p]]
    sub = dict(d)
    sub["example_order"] = keys
    sub["examples"] = {k: d["examples"][k] for k in keys}
    (ROOT / f"visualizer/data_valset_shard{i:02d}.js").write_text(
        "window.VISUALIZER_DATA = " + json.dumps(sub) + ";", encoding="utf-8")
    print(f"shard {i:02d}: {len(perfs[i::K])} perfs, {len(keys)} windows")
