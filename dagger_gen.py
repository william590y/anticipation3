"""DAgger window generation for sliding-window exposure bias.

Rolls the model over TRAIN-split performances with the production sliding
decode (window_mode="slide"). At every re-pack boundary the decoder hands us
the training-shaped context it just built -- header + the model's OWN
predicted overlap, re-anchored exactly as at decode time -- and we complete
it into a 1020-token training line by appending the 69 fresh (score, control)
pairs with ALIGNED GT score triplets, localized with the same offsets the
decoder would use for teacher forcing.

Train with train.py --loss_mask_first_score_slots 69 so the loss covers only
the GT continuation: model output as input/context, ground truth as target
(DAgger), never the model's own tokens as targets (no self-distillation).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import evaluate_muster_asap as ema                                  # noqa: E402
from anticipation.vocab import (CONTROL_OFFSET, DUR_OFFSET, NOTE_OFFSET,  # noqa: E402
                                TIME_OFFSET)

WIN = 138  # body score slots in a packed window


def split_perfs(split_file: str) -> set:
    out, inside = set(), False
    for line in open(split_file, encoding="utf-8"):
        line = line.strip()
        if line == "=== TRAIN PERFORMANCES ===":
            inside = True
            continue
        if line.startswith("==="):
            inside = False
            continue
        if inside and line and not line.startswith("#"):
            out.add(line.lstrip("./"))
    if not out:
        raise SystemExit("no train performances found")
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split-file", default="data/paper_split.txt")
    ap.add_argument("--out-dir", default="dagger_windows/train_maskft75")
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-perfs", type=int, default=None,
                    help="deterministic even subsample of the train split")
    ap.add_argument("--window-overlap", type=int, default=69,
                    help="decode overlap the windows are harvested at; the "
                         "window then holds `overlap` self-predicted notes and "
                         "138-overlap fresh GT-target slots")
    ap.add_argument("--boundary-stride", type=int, default=1,
                    help="keep every Nth harvested boundary; at overlap 137 a "
                         "boundary fires every note, so this caps the file")
    a = ap.parse_args()

    want = split_perfs(a.split_file)
    pieces = [p for p in ema.load_asap_metadata()
              if p["perf_path"].lstrip("./") in want]
    pieces.sort(key=lambda p: p["perf_path"])
    if a.max_perfs and a.max_perfs < len(pieces):
        step = len(pieces) / a.max_perfs
        pieces = [pieces[int(i * step)] for i in range(a.max_perfs)]
    mine = pieces[a.shard_index::a.num_shards]
    out_dir = Path(a.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"train split: {len(pieces)} perfs, shard {a.shard_index}/"
          f"{a.num_shards} -> {len(mine)}", flush=True)

    model, device = ema.load_model(a.checkpoint, config_source="auto")
    for pi in mine:
        name = pi["perf_path"].lstrip("./").replace("/", "__")[:-4] + ".txt"
        out = out_dir / name
        if out.exists():
            print(f"  skip (exists): {name}", flush=True)
            continue
        info = ema.preprocess_asap_piece(pi, gt_score_source="midi")
        ctl, gt = info["control_triplets"], info["gt_score_triplets"]
        if len(ctl) < 200 or len(gt) < 200:
            print(f"  skip (short): {name}", flush=True)
            continue
        lines = []
        fresh_n = max(1, WIN - a.window_overlap)
        seen = [0]

        def sink(context, note_idx, s_idx, anchor, ctl_off, future_idx):
            seen[0] += 1
            if (seen[0] - 1) % a.boundary_stride:
                return
            line = list(context)
            for j in range(fresh_n):
                k = note_idx + j
                ci = future_idx + j
                if k >= len(gt) or ci >= len(ctl):
                    return
                # GT score triplet localized exactly like decode-time teacher
                # forcing in this window (score_time_offset=anchor, min=0)
                time_tok = min(max(int(gt[k][0]) - anchor, TIME_OFFSET),
                               DUR_OFFSET - 1)
                dur_tok = min(max(int(gt[k][1]), DUR_OFFSET), NOTE_OFFSET - 1)
                pitch_tok = min(max(int(gt[k][2]), NOTE_OFFSET),
                                CONTROL_OFFSET - 1)
                line.extend([time_tok, dur_tok, pitch_tok])
                line.extend(ema.localize_control_triplet(ctl[ci], ctl_off))
            if len(line) == ema.PACKED_SEQUENCE_LENGTH:
                lines.append(" ".join(str(int(x)) for x in line))

        ema.autoregressive_generate_from_controls(
            model, ctl, gt, device, temperature=0.0,
            ground_truth_score_notes_to_feed=0,
            window_mode="slide", window_overlap=a.window_overlap,
            dagger_sink=sink)
        tmp = out.with_suffix(".tmp")
        tmp.write_text("\n".join(lines) + ("\n" if lines else ""))
        tmp.rename(out)
        print(f"  wrote {name}: {len(lines)} windows", flush=True)


if __name__ == "__main__":
    main()
