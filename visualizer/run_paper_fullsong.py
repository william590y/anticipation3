#!/usr/bin/env python
"""Full-song transcriptions from BOTH papers for every test/val performance.

The window pipeline (run_paper_models.py) computed exactly these transcriptions
and kept only per-window slices; this saves the WHOLE piece, one JSON per
(model, performance):

  {perf_path, model, qpb, notes: [{on_q, dur_q, p}...]}

Must run in the `paperpipe` env. The model-loading block is copied from
run_paper_models.main (lines ~336-375) -- including the sys.modules eviction by
module ORIGIN, which is load-bearing when both repos are loaded in one process.
Shardable over performances and file-gated (skips existing outputs).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(REPO))

spec = importlib.util.spec_from_file_location("run_paper_models",
                                              HERE / "run_paper_models.py")
rpm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rpm)


def split_perfs(split_file: Path, want: str) -> list:
    tag = {"test": "=== TEST PERFORMANCES ===",
           "val": "=== VALIDATION PERFORMANCES ==="}[want]
    out, inside = [], False
    for line in open(split_file, encoding="utf-8"):
        line = line.strip()
        if line == tag:
            inside = True
            continue
        if line.startswith("==="):
            inside = False
            continue
        if inside and line and not line.startswith("#"):
            out.append(line.lstrip("./"))
    return sorted(out)


def load_paper_model(kind: str, device: str, p1_ckpt: str, p2_ckpt: str):
    """Copied from run_paper_models.main -- see its comments for the traps."""
    repo = REPO / ("external/paper2_midi2score" if kind == "paper2"
                   else "external/paper1_joint_apt_epr")
    sub = "midi2scoretransformer" if kind == "paper2" else None
    ckpt = p2_ckpt if kind == "paper2" else p1_ckpt
    rpm._add_repo_to_path(repo, sub)
    roots = tuple(str((REPO / "external" / d).resolve())
                  for d in ("paper1_joint_apt_epr", "paper2_midi2score"))
    for name, cached in list(sys.modules.items()):
        origin = getattr(cached, "__file__", None) or ""
        if origin.startswith(roots):
            del sys.modules[name]
    import tokenizer as tokenizer_mod          # noqa: F401
    import utils as utils_mod                  # noqa: F401
    import config as config_mod
    if hasattr(config_mod, "MyModelConfig"):
        torch.serialization.add_safe_globals([config_mod.MyModelConfig])
    with rpm.working_dir(repo):
        if kind == "paper2":
            from models.roformer import Roformer
            model = Roformer.load_from_checkpoint(ckpt, map_location=device)
        else:
            # Copied VERBATIM from run_paper_models.py:371-382 this time --
            # the first version reconstructed these kwargs from a truncated
            # read and passed a single bare `config=`, which left
            # pad_token_id None and crashed every paper1 shard.
            from config import MyModelConfig
            from train import JointModel
            common = dict(num_hidden_layers=6, hidden_size=512,
                          intermediate_size=3072, num_attention_heads=8)
            model = JointModel.load_from_checkpoint(
                ckpt,
                enc_config=MyModelConfig(**common),
                style_enc_config=MyModelConfig(**common, is_style_encoder=True),
                dec_config=MyModelConfig(**common, is_decoder=True,
                                         is_autoregressive=True,
                                         add_cross_attention=True),
                map_location=device)
    model = model.to(device).eval()
    return model, tokenizer_mod, utils_mod


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--splits", default="test,val")
    ap.add_argument("--split-file", default=str(REPO / "data/paper_split.txt"))
    ap.add_argument("--out-dir", default=str(REPO / "fullsong_papers"))
    ap.add_argument("--paper1-ckpt",
                    default=str(REPO / "external/weights/joint_apt_epr.ckpt"))
    ap.add_argument("--paper2-ckpt",
                    default=str(REPO / "external/weights/MIDI2ScoreTF.ckpt"))
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--zeng-first", action="store_true",
                    help="run paper1 before paper2 -- give this to half the "
                         "shards so both models make progress from minute one "
                         "instead of Zeng waiting behind every Beyer piece")
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    jobs = []
    for split in a.splits.split(","):
        for perf in split_perfs(Path(a.split_file), split):
            jobs.append((split, perf))
    mine = jobs[a.shard_index::a.num_shards]
    print(f"{len(jobs)} (split, perf) jobs; shard {a.shard_index}/"
          f"{a.num_shards} -> {len(mine)}", flush=True)

    order = ("paper1", "paper2") if a.zeng_first else ("paper2", "paper1")
    for kind in order:
        todo = []
        for split, perf in mine:
            out = (Path(a.out_dir) / split / kind
                   / (perf.replace("/", "__")[:-4] + ".json"))
            if not out.exists():
                todo.append((split, perf, out))
        if not todo:
            print(f"{kind}: nothing to do", flush=True)
            continue
        print(f"=== {kind}: {len(todo)} performances ===", flush=True)
        model, tok, utils_mod = load_paper_model(
            kind, device, a.paper1_ckpt, a.paper2_ckpt)
        for split, perf, out in todo:
            try:
                qpb = rpm.quarters_per_annotated_beat(perf)
                if qpb is None:
                    print(f"  {perf}: no meter; skipped", flush=True)
                    continue
                notes = rpm.transcribe_piece(kind, model, tok, utils_mod.infer,
                                             rpm.ASAP / perf, device)
                out.parent.mkdir(parents=True, exist_ok=True)
                tmp = out.with_suffix(".tmp")
                tmp.write_text(json.dumps(
                    {"perf_path": perf, "split": split, "model": kind,
                     "qpb": qpb,
                     "notes": [{"on_q": n["onset_q"], "dur_q": n["dur_q"],
                                "p": n["pitch"]} for n in notes]}))
                tmp.rename(out)
                print(f"  {split}/{perf}: {len(notes)} notes", flush=True)
            except Exception as exc:
                print(f"  {split}/{perf}: FAILED {type(exc).__name__}: {exc}",
                      flush=True)
        del model
        torch.cuda.empty_cache()
    print("SHARD_DONE", flush=True)


if __name__ == "__main__":
    main()
