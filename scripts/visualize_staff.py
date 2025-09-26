import argparse
import os
from typing import List, Tuple, Dict, Any

import pandas as pd
from tqdm import tqdm

from alignment import align_tokens2
from anticipation.config import *
from anticipation.vocab import *


def decode_event_triplet(triplet: List[int], is_control: bool) -> Tuple[float, float, int]:
    """Convert a token triplet into (start_seconds, duration_seconds, midi_pitch).

    If is_control is True, interpret using ATIME/ADUR/ANOTE offsets; otherwise TIME/DUR/NOTE.
    """
    t, d, n = triplet
    if is_control:
        start_ticks = t - ATIME_OFFSET
        dur_ticks = d - ADUR_OFFSET
        note_val = n - ANOTE_OFFSET
    else:
        start_ticks = t - TIME_OFFSET
        dur_ticks = d - DUR_OFFSET
        note_val = n - NOTE_OFFSET

    pitch = note_val % 128
    start_seconds = start_ticks / float(TIME_RESOLUTION)
    dur_seconds = max(0.01, dur_ticks / float(TIME_RESOLUTION))  # avoid zero-duration notes
    return start_seconds, dur_seconds, pitch


def build_interleave_order(matched: List[List[Any]], method: str = "t4", prefix_controls: int = 33) -> List[Tuple[str, int]]:
    """Return a list of (type, matched_index) in the order they are emitted in the interleaved stream.

    type is one of: 'score', 'control', 'pad' (pad appears only in t4 prefix).
    matched_index refers to the index into the matched_tuples list for the source note.
    """
    order: List[Tuple[str, int]] = []
    if method == "t3":
        # time-based control prefix
        prefix_len = 0
        for m in matched:
            if m[0][0] - CONTROL_OFFSET <= DELTA * TIME_RESOLUTION:
                prefix_len += 1
        for i, m in enumerate(matched):
            sc = m[2]
            if sc[0] is not None:
                order.append(("score", i))
            ii = i + prefix_len
            if ii < len(matched):
                order.append(("control", ii))
    else:
        k = min(prefix_controls, len(matched))
        # fixed count control+pad prefix
        for i in range(k):
            order.append(("control", i))
            order.append(("pad", i))
        for i, m in enumerate(matched):
            sc = m[2]
            if sc[0] is not None:
                order.append(("score", i))
            ii = i + k
            if ii < len(matched):
                order.append(("control", ii))
    return order


def main():
    parser = argparse.ArgumentParser(description="Render score and performance on staves with interleaving order")
    parser.add_argument("--asap-root", default="./asap-dataset-master")
    parser.add_argument("--row", type=int, default=0, help="Row index in metadata.csv to visualize")
    parser.add_argument("--skip-nones", action="store_true")
    parser.add_argument("--method", choices=["t3", "t4"], default="t4")
    parser.add_argument("--prefix-controls", type=int, default=33)
    parser.add_argument("--qpm", type=float, default=120.0, help="Tempo (quarters per minute) for display timing")
    parser.add_argument("--out", default="./data/staff_interleave.musicxml", help="Output file path (.musicxml or .png)")
    parser.add_argument("--format", choices=["musicxml", "png"], default=None, help="Output format override; inferred from --out if not set")
    parser.add_argument("--musescore", default=None, help="Path to MuseScore executable for PNG export (e.g., C:/Program Files/MuseScore 4/bin/MuseScore4.exe)")
    args = parser.parse_args()

    import music21 as m21  # import inside to avoid requiring it globally

    meta = os.path.join(args.asap_root, "metadata.csv")
    df = pd.read_csv(meta)
    row = df.iloc[args.row]
    perf = os.path.join(args.asap_root, row["midi_performance"])
    score = os.path.join(args.asap_root, row["midi_score"])
    perf_ann = os.path.join(args.asap_root, row["performance_annotations"])
    score_ann = os.path.join(args.asap_root, row["midi_score_annotations"])

    matched = align_tokens2(perf, score, perf_ann, score_ann, skip_Nones=args.skip_nones)

    # Decode all usable notes
    score_notes: List[Tuple[float, float, int, int]] = []  # (start_s, dur_s, pitch, idx)
    perf_notes: List[Tuple[float, float, int, int]] = []   # (start_s, dur_s, pitch, idx)
    for i, m in enumerate(matched):
        # performance control
        start_s, dur_s, pitch = decode_event_triplet(m[0], is_control=True)
        perf_notes.append((start_s, dur_s, pitch, i))
        # score event
        if m[2][0] is not None:
            start_s2, dur_s2, pitch2 = decode_event_triplet(m[2], is_control=False)
            score_notes.append((start_s2, dur_s2, pitch2, i))

    # Build interleaving order to annotate
    order = build_interleave_order(matched, method=args.method, prefix_controls=args.prefix_controls)
    order_map: Dict[Tuple[str, int], int] = {}
    for seq_idx, (ty, idx) in enumerate(order):
        order_map[(ty, idx)] = seq_idx + 1  # 1-based for readability

    # Build the score: two Parts
    s = m21.stream.Score()
    part_score = m21.stream.Part(id="Score")
    part_perf = m21.stream.Part(id="Performance")

    # Set tempo
    mm = m21.tempo.MetronomeMark(number=args.qpm)
    ts = m21.meter.TimeSignature('4/4')
    part_score.insert(0, mm)
    part_perf.insert(0, mm)
    part_score.insert(0, ts)
    part_perf.insert(0, ts)

    # Helper to add notes with lyric numbers indicating interleaving order
    def add_notes(part: m21.stream.Part, notes: List[Tuple[float, float, int, int]], ty: str):
        # Quantize helper to avoid inexpressible durations in MusicXML
        def q(x: float, denom: int = 32) -> float:
            return max(1.0/denom, round(x * denom) / denom)
        for start_s, dur_s, pitch, idx in notes:
            ql = q(dur_s * (args.qpm / 60.0))
            start_ql = q(start_s * (args.qpm / 60.0))
            n = m21.note.Note(pitch)
            n.quarterLength = ql
            # annotate order if exists
            if (ty, idx) in order_map:
                n.lyric = str(order_map[(ty, idx)])
            part.insert(start_ql, n)

    add_notes(part_score, score_notes, "score")
    add_notes(part_perf, perf_notes, "control")

    # Order parts consistently: performance below score for readability
    s.insert(0, part_score)
    s.insert(0, part_perf)

    # Write output
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    # Let music21 create measures explicitly to stabilize export
    part_score.makeMeasures(inPlace=True)
    part_perf.makeMeasures(inPlace=True)

    # Determine format
    out_fmt = args.format
    if out_fmt is None:
        out_fmt = 'png' if args.out.lower().endswith('.png') else 'musicxml'

    if out_fmt == 'musicxml':
        s.write('musicxml', fp=args.out)
        print(f"Wrote {args.out}")
    else:
        # PNG export: write MusicXML then invoke MuseScore CLI to render PNG
        import subprocess
        import shutil
        base, _ = os.path.splitext(args.out)
        xml_out = base + '.musicxml'
        s.write('musicxml', fp=xml_out)
        if not args.musescore:
            # Try verovio -> SVG -> PNG as a pure-Python fallback
            try:
                import verovio  # type: ignore
                from cairosvg import svg2png  # type: ignore
                tk = verovio.toolkit()
                tk.loadFile(xml_out)
                page_count = tk.getPageCount()
                # Render first page (extend later if multi-page support is needed)
                svg = tk.renderToSVG(1, {})
                svg2png(bytestring=svg.encode('utf-8'), write_to=args.out)
                print(f"Wrote {args.out} (via verovio)")
                return
            except Exception as e:
                print("PNG export via verovio failed. Error:\n", e)
                print(f"MusicXML written to {xml_out}.")
                print("To render PNG, either:\n"
                      "  - Install MuseScore and run: \"<MuseScorePath>\" \"{xml_out}\" -o \"{args.out}\"\n"
                      "  - Or install verovio and cairosvg: pip install verovio cairosvg")
                return
        if not os.path.isfile(args.musescore):
            print(f"MuseScore not found at: {args.musescore}")
            print(f"MusicXML written to {xml_out}. Render manually with your MuseScore path.")
            return
        try:
            completed = subprocess.run([args.musescore, xml_out, '-o', args.out], capture_output=True, text=True, check=True)
            print(f"Wrote {args.out}")
        except subprocess.CalledProcessError as e:
            print("Failed to export PNG via MuseScore. STDERR:\n", e.stderr)
            print("STDOUT:\n", e.stdout)
            print(f"MusicXML is available at {xml_out}. Try running the command manually in a terminal.")


if __name__ == "__main__":
    main()
