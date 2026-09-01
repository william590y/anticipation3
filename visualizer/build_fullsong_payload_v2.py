"""Build data_fullsong.js for the rebuilt rollout visualizer.

Per performance: GT notes, every ours-variant lane (all regimes), Beyer/Zeng
lanes, per-song metrics (tol1/scale-max/IOI/IOI-scale-max + MUSTER
components), per-song DTW curves (warp factor, unwarped/warped |onset error|),
and — for the trace-decoded slide-69 lanes — per-note entropy, perplexity and
top-5 per-channel alternatives. Missing inputs are skipped so the payload can
be rebuilt incrementally as jobs land.
"""
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# variant -> (dir, has_trace) ; ours variants first, then papers
LANES = {
    "base_s69": ("fullsong_trace_base/test/slide", True),
    "base_s1":  ("fullsong_trace_base_s1/test/slide", True),
    "base_rst": ("fullsong_trace_base_rst/test/reset", True),
    "base_gtc": ("fullsong_trace_base_gtc/test/slide_gtctx", True),
    "base_pf":  ("fullsong_trace_base_pf/test/slide_pforce", True),
    "m25_s69":  ("fullsong_trace_mask25/test/slide", True),
    "m25_s1":   ("fullsong_trace_m25_s1/test/slide", True),
    "m25_rst":  ("fullsong_trace_m25_rst/test/reset", True),
    "m25_pf":   ("fullsong_trace_m25_pf/test/slide_pforce", True),
    "m50_s69":  ("fullsong_trace_mask50/test/slide", True),
    "m50_s1":   ("fullsong_trace_m50_s1/test/slide", True),
    "m50_rst":  ("fullsong_trace_m50_rst/test/reset", True),
    "m50_pf":   ("fullsong_trace_m50_pf/test/slide_pforce", True),
    "m75_s69":  ("fullsong_trace_mask75/test/slide", True),
    "m75_s1":   ("fullsong_trace_m75_s1/test/slide", True),
    "m75_rst":  ("fullsong_trace_m75_rst/test/reset", True),
    "m75_gtc":  ("fullsong_trace_m75_gtc/test/slide_gtctx", True),
    "m75_pf":   ("fullsong_trace_m75_pf/test/slide_pforce", True),
}
# fall back to the un-traced slide-69 dirs until trace decodes land
FALLBACK = {
    "base_s69": "fullsong_rollouts/test/slide",
    "base_s1": "fullsong_slide1_base/test/slide",
    "base_rst": "fullsong_rollouts/test/reset",
    "base_gtc": "fullsong_rollouts/test/slide_gtctx",
    "base_pf": "fullsong_rollouts/test/slide_pforce",
    "m25_s69": "fullsong_rollouts_maskft25/test/slide",
    "m25_s1": "fullsong_slide1_mask25/test/slide",
    "m25_rst": "fullsong_rollouts_maskft25/test/reset",
    "m25_pf": "fullsong_rollouts_maskft25/test/slide_pforce",
    "m50_s69": "fullsong_rollouts_maskft50/test/slide",
    "m50_s1": "fullsong_slide1_mask50/test/slide",
    "m50_rst": "fullsong_rollouts_maskft50/test/reset",
    "m50_pf": "fullsong_rollouts_maskft50/test/slide_pforce",
    "m75_s69": "fullsong_rollouts_maskft/test/slide",
    "m75_s1": "fullsong_slide1_mask75/test/slide",
    "m75_rst": "fullsong_rollouts_maskft/test/reset",
    "m75_gtc": "fullsong_rollouts_maskft/test/slide_gtctx",
    "m75_pf": "fullsong_rollouts_maskft/test/slide_pforce",
}
PAPERS = {"beyer": "fullsong_papers/test/paper2",
          "zeng": "fullsong_papers/test/paper1"}
# eval-table source names (nbest/eval_fullsong_regimes.SOURCES) per variant
EVAL_NAME = {
    "base_s69": "base s69", "base_s1": "base s1", "base_rst": "base rst",
    "base_gtc": "base gtc", "base_pf": "base pf",
    "m25_s69": "m25 s69", "m25_s1": "m25 s1", "m25_rst": "m25 rst",
    "m25_pf": "m25 pf",
    "m50_s69": "m50 s69", "m50_s1": "m50 s1", "m50_rst": "m50 rst",
    "m50_pf": "m50 pf",
    "m75_s69": "m75 s69", "m75_s1": "m75 s1", "m75_rst": "m75 rst",
    "m75_gtc": "m75 gtc", "m75_pf": "m75 pf",
    "beyer": "beyer", "zeng": "zeng",
}
DTW_FILE = {
    "base_s69": "dtw_drift.json", "base_s1": "dtw_slide1_base.json",
    "m25_s1": "dtw_slide1_mask25.json", "m75_s1": "dtw_slide1_mask75.json",
}
for v in list(EVAL_NAME):
    if v not in DTW_FILE:
        DTW_FILE[v] = f"dtw_lane_{v}.json"

METS = ("base", "smax", "ioi", "ioi_smax")
MKEYS = ("pitch_error_rate", "missing_note_rate", "extra_note_rate",
         "onset_time_error_rate", "offset_time_error_rate", "mean_error_rate")


def rnd(x, k=3):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) \
        else round(float(x), k)


def load_json(p):
    p = ROOT / p
    return json.load(open(p)) if p.exists() else None


def main():
    gt_dir = ROOT / "fullsong_rollouts/test/slide"
    stems = sorted(f.stem for f in gt_dir.glob("*.json"))
    persong = load_json("nbest_data/fullsong_regimes_persong.json") or {}
    muster = load_json("nbest_data/muster_fixed_fullsong_test.json") or {}
    mus = {}
    for r in muster.get("records", []):
        if r.get("metrics"):
            mus[(r["key"], r["name"])] = {k: rnd(r["metrics"][k], 2)
                                          for k in MKEYS}
    agg_raw = load_json("nbest_data/fullsong_regimes_eval.json") or {}

    dtw = {}
    for var, fn in DTW_FILE.items():
        d = load_json(f"nbest_data/{fn}")
        if not d:
            continue
        dtw[var] = {rec["file"][:-5]: {
            "warp": [rnd(x) for x in rec["warp"]],
            "eraw": [rnd(x, 1) for x in rec["err_raw"]],
            "ewrp": [rnd(x, 1) for x in rec["err_warp"]],
        } for rec in d}

    out = {"order": [], "meta": {}, "gt": {}, "lanes": {}, "trace": {},
           "f1": {}, "muster": {}, "dtw": {}, "agg": {},
           "variants": list(LANES) , "papers": list(PAPERS)}

    for stem in stems:
        base = json.load(open(gt_dir / f"{stem}.json"))
        key = "test/" + base["perf_path"].lstrip("./")
        out["order"].append(key)
        out["meta"][key] = {
            "stem": stem,
            "work": "/".join(base["perf_path"].lstrip("./").split("/")[:-1]),
            "overlap": base.get("window_overlap", 69),
            "boundaries": base["gen_stats"]["num_window_resets"],
        }
        out["gt"][key] = [[n["t"], n["d"], n["p"]] for n in base["gt"]]

    def add_lane(var, dd, kind):
        lane, trace = {}, {}
        for stem in stems:
            p = ROOT / dd / f"{stem}.json"
            if not p.exists():
                continue
            r = json.load(open(p))
            key = "test/" + (r["perf_path"].lstrip("./") if "perf_path" in r
                             else r.get("perf_path", stem))
            if kind == "units":
                lane[key] = [[n["t"], n["d"], n["p"]] for n in r["pred"]]
                tr = r.get("trace")
                if tr and tr.get("H_time"):
                    trace[key] = {
                        "ppl": [[rnd(a, 2), rnd(b, 2), rnd(c, 2)] for a, b, c
                                in zip(tr["time"], tr["dur"], tr["pitch"])],
                        "H": [[a, b, c] for a, b, c in
                              zip(tr["H_time"], tr["H_dur"], tr["H_pitch"])],
                        "alts": [[a, b, c] for a, b, c in
                                 zip(tr["alt_time"], tr["alt_dur"],
                                     tr["alt_pitch"])],
                    }
            else:
                f = 50.0 / float(r["qpb"])
                lane[key] = [[int(round(n["on_q"] * f)),
                              max(1, int(round(n["dur_q"] * f))), n["p"]]
                             for n in r["notes"] if n["p"] >= 1]
        return lane, trace

    for var, (dd, has_trace) in LANES.items():
        lane, trace = add_lane(var, dd, "units") if (ROOT / dd).exists() \
            else ({}, {})
        if var in FALLBACK and len(lane) < len(stems):
            fb, _ = add_lane(var, FALLBACK[var], "units")
            for k2, v2 in fb.items():
                lane.setdefault(k2, v2)
        if lane:
            out["lanes"][var] = lane
        if trace:
            out["trace"][var] = trace
    for var, dd in PAPERS.items():
        lane, _ = add_lane(var, dd, "quarters")
        if lane:
            out["lanes"][var] = lane

    stem2key = {out["meta"][k]["stem"]: k for k in out["order"]}
    for var, ename in EVAL_NAME.items():
        f1v, musv = {}, {}
        for stem, key in stem2key.items():
            rec = persong.get(stem, {})
            vals = {m: rnd(rec.get(f"{ename}_{m}"), 4) for m in METS}
            if any(v is not None for v in vals.values()):
                f1v[key] = vals
            if (stem, ename) in mus:
                musv[key] = mus[(stem, ename)]
        if f1v:
            out["f1"][var] = f1v
        if musv:
            out["muster"][var] = musv
        ag = {}
        for m in METS:
            v = agg_raw.get(f"{ename}|{m}")
            if v and v[0]:
                ag[m] = rnd(v[1], 2)
        mv = [mus[(s, ename)]["mean_error_rate"] for s in stem2key
              if (s, ename) in mus]
        if mv:
            ag["muster"] = rnd(sum(mv) / len(mv), 2)
        if ag:
            out["agg"][var] = ag
        out["dtw"][var] = {stem2key[s]: c for s, c in dtw.get(var, {}).items()
                           if s in stem2key}

    dst = ROOT / "visualizer/data_fullsong.js"
    with open(dst, "w") as fh:
        fh.write("window.FULLSONG_DATA = ")
        json.dump(out, fh, separators=(",", ":"))
        fh.write(";\n")
    lanes_n = {v: len(l) for v, l in out["lanes"].items()}
    print(f"wrote {dst} ({dst.stat().st_size/1e6:.1f} MB)")
    print("lanes:", lanes_n)
    print("trace lanes:", {v: len(t) for v, t in out['trace'].items()})
    print("dtw lanes:", {v: len(d) for v, d in out['dtw'].items() if d})


if __name__ == "__main__":
    main()
