#!/usr/bin/env python
"""Add the two external papers' transcriptions to visualizer/data.js.

Runs, on each window's source performance MIDI:
  * paper 1 -- Zeng+, "Bridging Piano Transcription and Rendering via Disentangled
    Score Content and Style" (ICLR 2026), github.com/wei-zeng98/joint-apt-epr
  * paper 2 -- Beyer & Dai, "End-to-end Piano Performance-MIDI to Score Conversion
    with Transformers" (ISMIR 2024), github.com/TimFelixBeyer/MIDI2ScoreTransformer

and writes the per-window result into `rollouts_paper1` / `rollouts_paper2` in the
same {"t","d","p"} bin-grid shape our own rollouts use, so compute_f1.py can score
all three models identically.

MUST run in the `paperpipe` conda env (external/setup_env.sh): paper 2 requires
TimFelixBeyer's music21 fork, which would change our own engraving pipeline's
behaviour if installed into `base`.

Why a window can be sliced out of a full-piece transcription
------------------------------------------------------------
Both models are NOTE-ALIGNED: they consume the performance's note stream and emit
one score note per input note (their own eval slices `y[i, :length_i]` with the
input length). Our packed window is a contiguous run of *matched* performance
notes from the same piece, so we transcribe the whole piece once, then take the
outputs at the input-note indices our window covers.

A packed window holds PREFIX_CONTROLS (32) more performance notes than score slots
-- the prefix front-loads controls so the model can anticipate -- and score slot k
pairs with perf_notes[k], so it is the TRAILING 32 controls that have no score
note. Only the first len(gt_score) controls are scored.

Unit conversion
---------------
Both papers emit musical time in a MEASURE-RELATIVE form (paper 1 copied paper 2's
tokenizer, so the PARAMS are identical): an `offset` within the measure plus a
`downbeat` giving the length of the measure that just ended, both as bucket indices
on a 1/24-quarter step. `decode_score_streams` accumulates those into absolute
quarters. Our score grid is 50 bins per ANNOTATED beat -- one line of ASAP's
midi_score_annotations.txt -- which is not always a quarter: compound meters are
annotated at the dotted beat (6/8 -> 1.5 quarters, 6/4 -> 3), 4/2 at the half. So:

    bins = quarters * (BINS_PER_BEAT / quarters_per_annotated_beat)

`quarters_per_annotated_beat` is MEASURED from ASAP's annotations rather than
inferred from the meter (see that function). If it cannot be determined, or the
converted span disagrees badly with the ground truth's span, the window is left
unscored (the panel shows "not computed") rather than displaying a misaligned
transcription.
"""
import argparse
import contextlib
from collections import Counter
import json
import os
import sys
from pathlib import Path

import torch


@contextlib.contextmanager
def working_dir(path):
    """Temporarily chdir. Paper 1's config.py does `load("hparams.yaml")` relative to
    the CWD, so its model must be constructed from inside its own repo directory."""
    prev = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(prev)

REPO = Path(__file__).resolve().parent.parent
ASAP = REPO / "asap-dataset-master"
BINS_PER_BEAT = 50
UNITS_PER_QUARTER = 24
# Windows outside this span ratio get a UI warning. Informational only --
# nothing is withheld or rescaled on the strength of it.
SPAN_TOLERANCE = 1.35


def _add_repo_to_path(repo_dir, subdir=None):
    """Put this repo's import root first on sys.path, and take the OTHER repo's off.

    Both repos import their own modules by bare name (`from tokenizer import ...`),
    so the active one must come first. Removing the other is not optional, and
    ordering alone is not enough: paper 1's `dataset/` has no `__init__.py`, so it
    is a NAMESPACE package, and Python ranks those BELOW regular modules. With
    paper 2's `dataset.py` anywhere on sys.path, `import dataset` resolves to
    paper 2's file even with paper 1's directory at position 0 -- and clearing
    sys.modules does not help, because the re-import finds it again.
    """
    external = (REPO / "external").resolve()
    for stale in list(sys.path):
        try:
            Path(stale).resolve().relative_to(external)
        except (ValueError, OSError):
            continue                     # not one of the vendored repos; leave it
        sys.path.remove(stale)
    p = str(repo_dir / subdir) if subdir else str(repo_dir)
    sys.path.insert(0, p)


def quarters_per_annotated_beat(piece_rel):
    """Quarters spanned by one ASAP-annotated beat, MEASURED from ASAP's own files.

    Our score grid advances 50 bins per LINE of midi_score_annotations.txt (see
    `asap_aligned_stream.normalize_score_triplet_to_fixed_beat`, which places a
    note at `idx * 0.5s + progress * 0.5s` over the annotation times), so the
    papers' quarter-valued output needs quarters-per-annotated-beat to land on it.

    Take that straight from the annotations rather than inferring it from the
    meter. Each downbeat line carries the time signature in effect ("db,6/4,-2"),
    and the number of lines to the next downbeat is that measure's beat count, so

        qpb = (num * 4/den) / lines_per_measure

    per measure; the piece's modal value wins. This matches whatever convention
    ASAP actually used, and sidesteps two ways of getting it wrong by inference:
    reading the piece's FIRST time signature when the meter changes later, and
    mis-classifying compound meters (6/4 is compound duple -- 2 dotted-half beats,
    3 quarters each -- just as much as 6/8 is).
    """
    ann = ASAP / Path(piece_rel).parent / "midi_score_annotations.txt"
    if not ann.exists():
        return None
    try:
        kinds, ts_at = [], {}
        for i, line in enumerate(ann.read_text(encoding="utf-8").strip().split("\n")):
            parts = line.split("\t")
            field = parts[2] if len(parts) >= 3 else ""
            bits = field.split(",")
            kinds.append(bits[0])
            if len(bits) >= 2 and "/" in bits[1]:
                num, den = bits[1].split("/")
                ts_at[i] = (int(num), int(den))

        downbeats = [i for i, k in enumerate(kinds) if k == "db"]
        counts, current = Counter(), None
        for start, nxt in zip(downbeats, downbeats[1:]):
            current = ts_at.get(start, current)
            if current is None or nxt <= start:
                continue
            num, den = current
            counts[(num * 4.0 / den) / (nxt - start)] += 1
        if not counts:
            return None
        qpb = counts.most_common(1)[0][0]
        if len(counts) > 1:
            spread = ", ".join(f"{q:g}x{n}" for q, n in counts.most_common())
            print(f"    {piece_rel}: meter changes ({spread}); using modal qpb={qpb:g}")
        return qpb
    except Exception as exc:  # noqa: BLE001 - a bad annotation must not kill the run
        print(f"    meter lookup failed for {piece_rel}: {type(exc).__name__}: {exc}")
        return None


def to_bins(onset_q, dur_q, qpb):
    scale = BINS_PER_BEAT / qpb
    return int(round(onset_q * scale)), max(1, int(round(dur_q * scale)))


def _stream_indices(t, n_notes, kind):
    """A stream's bucket indices, plus the origin shift its repo's decoder applies.

    The two repos hand back different things from `infer`, and BOTH are 2-D, so
    the caller must say which -- shape alone cannot tell them apart:
      * paper 1 returns token INDICES with a batch dim, (1, T). Index 0 is <PAD>,
        so its `detokenize_mxl` subtracts 1 before unbucketing -> shift 1.
      * paper 2 returns per-class scores with no batch dim, (T, vocab) (its
        `infer` un-batches single examples itself). Its `one_hot_unbucketing`
        argmaxes and unbuckets straight off that, with no PAD offset -> shift 0.
    """
    if kind == "paper1":
        return t[0][:n_notes].cpu(), 1
    return t[:n_notes].argmax(dim=-1).cpu(), 0


def _unbucket(tok, minimum, step, shift):
    """Inverse of the papers' bucketing: value = min + (index - shift) * step."""
    return (tok.float() - shift) * step + minimum


def decode_score_streams(y, n_notes, kind="paper1"):
    """Per-note (onset, duration, pitch) in QUARTER lengths from a model's outputs.

    Both papers use the SAME measure-relative score representation -- paper 1
    copied paper 2's tokenizer -- with identical bucket PARAMS:
      `offset`   = the note's position within its measure     (min 0,     step 1/24)
      `downbeat` = how far the measure start advanced here    (min -1/24, step 1/24);
                   the sentinel -1/24 means "still in the same measure", otherwise
                   the value is the length of the measure that just ended.
      `duration` = note length                                (min 0,     step 1/24)
      `pitch`    = MIDI pitch                                 (min 0,     step 1)
    Absolute onset = running measure start + offset. They differ only in output
    tensor shape and bucket origin, which `_stream_indices` resolves.

    Verified by round-tripping a real `xml_score.musicxml` through the repo's own
    `parse_mxl` -> `bucket_mxl` -> this function: onsets match music21's absolute
    offsets to ~1e-5 quarters and pitches/durations exactly.
    """
    step = 1.0 / UNITS_PER_QUARTER
    off_tok, shift = _stream_indices(y["offset"], n_notes, kind)
    down_tok, _ = _stream_indices(y["downbeat"], n_notes, kind)
    dur_tok, _ = _stream_indices(y["duration"], n_notes, kind)
    pitch_tok, _ = _stream_indices(y["pitch"], n_notes, kind)

    offs = _unbucket(off_tok, 0.0, step, shift)
    down = _unbucket(down_tok, -step, step, shift)
    dur_q = _unbucket(dur_tok, 0.0, step, shift)
    pitch = _unbucket(pitch_tok, 0.0, 1.0, shift)

    onset_q, measure_start = [], 0.0
    for i in range(n_notes):
        adv = float(down[i])
        if adv > 0:                                   # a new measure begins here
            measure_start += adv
        onset_q.append(measure_start + float(offs[i]))

    return [{"onset_q": float(onset_q[i]), "dur_q": float(dur_q[i]),
             "pitch": int(round(float(pitch[i])))} for i in range(n_notes)]


def transcribe_piece(model_kind, model, tokenizer_mod, infer_fn, midi_path, device):
    """Full-piece transcription -> list of per-input-note dicts in QUARTER units."""
    x = tokenizer_mod.MultistreamTokenizer.tokenize_midi(str(midi_path))
    n_notes = x["pitch"].shape[0]

    # The two repos' inputs and infer() signatures differ; call each the way its
    # own driver does (paper 1's inference.py, paper 2's utils.quantize_path).
    with torch.no_grad():
        if model_kind == "paper1":
            # 1-D index streams. Index 0 is <PAD>, so ids shift +1 (not the pad
            # mask itself), and infer() only leaves the batch dim on if we add it.
            batch = {k: ((v + 1) if k != "pad" else v).unsqueeze(0).to(device)
                     for k, v in x.items()}
            y = infer_fn(batch, model, overlap=64, chunk=256, kv_cache=True,
                         device=device)
        else:
            # paper 2: 2-D ONE-HOT rows (n_notes, n_buckets) -- no +1 shift, and
            # its infer() takes no `device` kwarg and un-batches single examples
            # itself, so pass it exactly as tokenize_midi produced it.
            batch = {k: v.to(device) for k, v in x.items()}
            y = infer_fn(batch, model, overlap=64, chunk=512, kv_cache=True)
    # infer() pads the sequence up to a chunk boundary (e.g. 754 notes -> 832),
    # so trim back to the real note count to restore 1:1 input/output alignment.
    notes = decode_score_streams(y, n_notes, model_kind)

    # Attach each input note's real performance time so windows can be located.
    # parse_midi returns ABSOLUTE onset seconds in the same note order tokenize_midi
    # consumes (tokenize_midi == parse_midi + bucket_midi, no reordering).
    raw = tokenizer_mod.MultistreamTokenizer.parse_midi(str(midi_path))
    if len(raw["onset"]) != n_notes:
        raise ValueError(f"parse_midi/{len(raw['onset'])} vs tokenize_midi/{n_notes} "
                         "note-count mismatch; cannot align window")
    for i, n in enumerate(notes):
        n["perf_onset_s"] = float(raw["onset"][i])
        n["perf_pitch"] = int(raw["pitch"][i])
    return notes


def window_slice(piece_notes, perf_notes, tol_s=0.05):
    """Map our window's performance notes onto indices in the model's input notes.

    Two things make this more than a plain lookup:
      * our window's control times are RE-ANCHORED to the window's own start
        (perf_anchor), so they are relative, not absolute piece seconds; and
      * our stream is FILTERED (performance notes with no aligned score note were
        dropped at tokenization), so it is an ordered SUBSEQUENCE of the model's
        input, not a contiguous run.
    So: anchor on a piece note whose pitch matches our first control, then walk
    forward greedily matching each subsequent control on (relative time, pitch),
    allowing skips. The anchor with the most matches wins.

    Returns a list PARALLEL TO `perf_notes` holding the matched index into
    `piece_notes`, or None where that control could not be located. Keeping the
    alignment positional (rather than returning just the hits) is what lets the
    caller drop the packed prefix controls, which have no score note.
    """
    if not perf_notes or not piece_notes:
        return []
    first_p = perf_notes[0]["p"]
    base_t = perf_notes[0]["t"] / 100.0
    best, best_hits = [], -1
    for anchor in range(len(piece_notes)):
        if piece_notes[anchor]["perf_pitch"] != first_p:
            continue
        base_s = piece_notes[anchor]["perf_onset_s"]
        aligned, cur = [], anchor
        for pn in perf_notes:
            want_s = base_s + (pn["t"] / 100.0 - base_t)
            j, hit = cur, None
            while j < len(piece_notes) and piece_notes[j]["perf_onset_s"] <= want_s + tol_s:
                if (piece_notes[j]["perf_pitch"] == pn["p"]
                        and abs(piece_notes[j]["perf_onset_s"] - want_s) <= tol_s):
                    hit = j
                    break
                j += 1
            aligned.append(hit)
            if hit is not None:
                cur = hit + 1
        hits = sum(a is not None for a in aligned)
        if hits > best_hits:
            best, best_hits = aligned, hits
            if best_hits == len(perf_notes):
                break
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="visualizer/data.js")
    ap.add_argument("--manifest", default="visualizer/paper_windows.json")
    ap.add_argument("--paper1-ckpt", default="external/weights/joint_apt_epr.ckpt")
    ap.add_argument("--paper2-ckpt", default="external/weights/MIDI2ScoreTF.ckpt")
    ap.add_argument("--only", choices=["paper1", "paper2"], default=None,
                    help="Run just one of the two models.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Only process the first N windows (diagnostics).")
    ap.add_argument("--no-span-guard", action="store_true",
                    help="Deprecated no-op; windows are never withheld.")
    ap.add_argument("--debug", action="store_true",
                    help="Print each window's ground truth against the prediction, "
                         "and report span/scale diagnostics for skipped windows.")
    args = ap.parse_args()

    for a in ("data", "manifest", "paper1_ckpt", "paper2_ckpt"):
        setattr(args, a, str(Path(getattr(args, a)).resolve()))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    txt = Path(args.data).read_text(encoding="utf-8")
    payload = json.loads(txt[txt.index("{"): txt.rindex("}") + 1])
    prefix = txt[: txt.index("{")]
    examples = payload["examples"]

    kinds = [args.only] if args.only else ["paper2", "paper1"]
    for kind in kinds:
        repo = REPO / ("external/paper2_midi2score" if kind == "paper2"
                       else "external/paper1_joint_apt_epr")
        sub = "midi2scoretransformer" if kind == "paper2" else None
        ckpt = args.paper2_ckpt if kind == "paper2" else args.paper1_ckpt
        print(f"\n=== {kind}: loading {ckpt} ===")
        try:
            _add_repo_to_path(repo, sub)
            # Both repos expose identically-named top-level modules, so the second
            # model loaded in one process must not inherit the first's. Evict by
            # WHERE a module was loaded from, not by name: a hardcoded name list
            # kept missing entries (first `models.embedding`, then `dataset`), and
            # each miss surfaces only when both models run in one process -- which
            # testing them separately with --only never exercises.
            roots = tuple(str((REPO / "external" / d).resolve())
                          for d in ("paper1_joint_apt_epr", "paper2_midi2score"))
            for name, cached in list(sys.modules.items()):
                origin = getattr(cached, "__file__", None) or ""
                if origin.startswith(roots):
                    del sys.modules[name]
            import tokenizer as tokenizer_mod
            import utils as utils_mod
            # Both checkpoints pickle a `config.MyModelConfig` instance. torch >= 2.6
            # defaults to weights_only=True, which refuses unknown globals; allowlist
            # just that one class (these are the papers' own official releases)
            # instead of turning the safety check off wholesale.
            import config as config_mod
            if hasattr(config_mod, "MyModelConfig"):
                torch.serialization.add_safe_globals([config_mod.MyModelConfig])
            _load_cwd = working_dir(repo)
            with _load_cwd:
              if kind == "paper2":
                from models.roformer import Roformer
                model = Roformer.load_from_checkpoint(ckpt, map_location=device)
              else:
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
            model.to(device).eval()
        except Exception as exc:  # noqa: BLE001
            print(f"  SKIPPING {kind}: could not load model -- "
                  f"{type(exc).__name__}: {exc}")
            continue

        cache = {}
        n_ok = 0
        items = list(examples.items())
        if args.limit:
            items = items[: args.limit]
        for key, ex in items:
            piece = ex.get("piece")
            if not piece:
                continue
            qpb = quarters_per_annotated_beat(piece)
            if qpb is None:
                print(f"  {key}: no meter for {piece}; skipped")
                continue
            try:
                if piece not in cache:
                    cache[piece] = transcribe_piece(
                        kind, model, tokenizer_mod, utils_mod.infer,
                        ASAP / piece, device)
                notes = cache[piece]
            except Exception as exc:  # noqa: BLE001
                print(f"  {key}: transcription failed on {piece} -- "
                      f"{type(exc).__name__}: {exc}")
                continue

            gt = [n for n in (ex.get("gt_score") or []) if n]
            perf = ex.get("perf_notes") or []
            # A packed window carries PREFIX_CONTROLS (32) more controls than it
            # has score slots: the prefix front-loads 32 performance notes so the
            # model can anticipate, and the body then pairs score slot k with
            # control k. So score slot k lines up with perf_notes[k] and it is the
            # TRAILING 32 controls that have no score note (verified: the gt pitch
            # sequence equals the first len(gt) control pitches, never the last).
            # Scoring all 170 adds 32 phantom predictions past the window's end.
            if len(perf) < len(gt):
                print(f"  {key}: {len(perf)} controls < {len(gt)} score notes; skipped")
                continue
            aligned = window_slice(notes, perf[: len(gt)])
            idx = [a for a in aligned if a is not None]
            if len(idx) < 0.5 * len(gt):
                print(f"  {key}: only matched {len(idx)}/{len(gt)} notes; skipped")
                continue
            # A model may emit <PAD> for an input note (declining to place a score
            # note there); bucket 0 unbuckets to pitch -1. Their own decoders drop
            # those via the pad mask, so do the same -- keeping them would invent a
            # note at an impossible pitch that can never match, penalising
            # precision for what is really a missed note (a recall failure, which
            # dropping it already reflects).
            sel = [notes[i] for i in idx if 0 <= notes[i]["pitch"] <= 127]
            n_pad = len(idx) - len(sel)
            if len(sel) < 0.5 * len(gt):
                print(f"  {key}: only {len(sel)}/{len(gt)} notes survive "
                      f"({n_pad} padded); skipped")
                continue
            base_q = min(n["onset_q"] for n in sel) if sel else 0.0
            pred, quarters = [], []
            for n in sel:
                t, d = to_bins(n["onset_q"] - base_q, n["dur_q"], qpb)
                pred.append({"t": t, "d": d, "p": n["pitch"]})
                # Keep the model's own QUARTER-valued output. Everything above this
                # point costs a GPU; everything below is arithmetic. Having already
                # changed the quarters->bins conversion twice, storing the input to
                # it means the next correction is a CPU rebin, not a rerun.
                quarters.append({"on": round(n["onset_q"] - base_q, 6),
                                 "dur": round(n["dur_q"], 6), "p": n["pitch"]})

            gt_span = max((n["t"] for n in gt), default=0) or 1
            pred_span = max((n["t"] for n in pred), default=0) or 1
            ratio = pred_span / gt_span
            if args.debug:
                print(f"\n  --- {key}  piece={piece}  qpb={qpb}  "
                      f"matched={len(sel)}/{len(gt)} (+{n_pad} pad)  ratio={ratio:.3f}")
                print(f"      gt   t: {[n['t'] for n in gt[:14]]}")
                print(f"      pred t: {[n['t'] for n in pred[:14]]}")
                print(f"      gt   p: {[n['p'] for n in gt[:14]]}")
                print(f"      pred p: {[n['p'] for n in pred[:14]]}")
                print(f"      gt   d: {[n['d'] for n in gt[:14]]}")
                print(f"      pred d: {[n['d'] for n in pred[:14]]}")
                print(f"      spans: gt={gt_span} pred={pred_span}")
            # No span guard. The quarters->bins conversion is the only scaling
            # applied and it is fixed by the time signature (verified against ASAP's
            # own annotations on every window), so a span that disagrees with the
            # ground truth is the model's actual output. Withholding those would
            # score each model on a different, self-selected subset. `span_ratio` is
            # kept as a descriptive statistic and drives a label in the UI only.
            span_ok = 1 / SPAN_TOLERANCE <= ratio <= SPAN_TOLERANCE

            # Record the window either way. `pred_score` -- what compute_f1 scores
            # and the UI draws -- appears only when the span checks out, but the
            # quarter-valued output and the ratio are always kept, so a guard-
            # tripping window can be diagnosed (and rebinned) without a rerun.
            block = {
                "source": f"{kind} released weights",
                "quarters_per_beat": qpb,
                "matched_notes": len(sel),
                "padded_notes": n_pad,
                "pred_quarters": quarters,
                "span_ratio": round(ratio, 4),
                "span_ok": bool(span_ok),
            }
            block["pred_score"] = pred
            n_ok += 1
            if not span_ok:
                print(f"  {key}: span ratio {ratio:.2f} vs ground truth "
                      f"(qpb={qpb}); scored anyway, flagged in the UI")
            ex.setdefault(f"rollouts_{kind}", {})["filtered"] = block
        print(f"  {kind}: wrote transcriptions for {n_ok}/{len(items)} windows")

    with Path(args.data).open("w", encoding="utf-8") as fh:
        fh.write(prefix)
        json.dump(payload, fh)
        fh.write(";\n")
    print(f"\nUpdated {args.data}")


if __name__ == "__main__":
    main()
