"""
Sidecar precompute: for every window in visualizer/data.js, find the window's
span in its piece's REAL MusicXML score (asap-dataset-master/**/xml_score.musicxml)
and export just those measures for display in the visualizer's engraved sheet view.

CPU-only post-process (like compute_muster.py) — run after precompute_visualizer.py.
Writes visualizer/scores_xml.js (window.VISUALIZER_SCORES = {...}), a sidecar the
HTML loads next to data.js; data.js itself is not touched.

How the mapping works: the aligned-stream cache stores each matched performance
note's score triplet in piece-level grid units (50 bins per annotated score beat,
i.e. the k-th line of the piece's midi_score_annotations.txt is grid time 50*k).
Each window is located in its piece exactly like precompute_visualizer's raw-stream
recovery (pitch-bytes probe + exact control verification), giving the piece-level
onset span of the window's score slots -> a beat range -> a measure range via the
annotation file's downbeat (db) markers -> music21 slices those measures out of
xml_score.musicxml. Pieces whose downbeat count disagrees with the XML measure
list (e.g. repeats unfolded in annotations) are skipped with a warning rather
than displayed misaligned.

Run:
  python visualizer/precompute_scores_xml.py --data visualizer/data.js
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
import time
from bisect import bisect_right
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from anticipation.asap_aligned_stream import STREAM_CACHE_DIR  # noqa: E402
from anticipation.vocab import ATIME_OFFSET, ADUR_OFFSET, ANOTE_OFFSET  # noqa: E402
from anticipation.vocab import TIME_OFFSET, DUR_OFFSET  # noqa: E402
from precompute_visualizer import locate_window  # noqa: E402
from compute_muster import load_payload  # noqa: E402

BINS_PER_BEAT = 50  # 0.5 s beat grid at 10 ms resolution


def load_cache_pieces_with_scores(cache_dir=STREAM_CACHE_DIR):
    """Like precompute_visualizer._load_cache_pieces, but additionally keeps each
    matched note's piece-level score (onset, dur) and the piece's source paths.

    ``cache_dir`` must match the cache ``--data`` was tokenized against (e.g. the
    0.48s run uses data/asap_aligned_stream_cache_b048).
    """
    pieces = []
    for path in sorted(glob.glob(str(Path(cache_dir) / "*.json"))):
        try:
            payload = json.load(open(path))
        except Exception:
            continue
        fp = payload.get("fingerprint", {})
        piece_id = fp.get("perf_midi", {}).get("path", path)
        raw, matched, score_spans = [], [], []
        for item in payload.get("items", []):
            c = item["control"]
            raw.append((c[0] - ATIME_OFFSET, c[1] - ADUR_OFFSET, c[2] - ANOTE_OFFSET))
            sc = item.get("score")
            matched.append(sc is not None)
            score_spans.append(
                None if sc is None else (sc[0] - TIME_OFFSET, sc[1] - DUR_OFFSET)
            )
        filtered_to_raw = [i for i, m in enumerate(matched) if m]
        pitch_bytes = bytes((raw[i][2] % 256) for i in filtered_to_raw)
        pieces.append({
            "piece_id": piece_id,
            "raw": raw,
            "matched": matched,
            "filtered_to_raw": filtered_to_raw,
            "pitch_bytes": pitch_bytes,
            "score_spans": score_spans,
            "score_beats_path": fp.get("score_beats", {}).get("path"),
            "score_midi_path": fp.get("score_midi", {}).get("path"),
        })
    return pieces


def load_beat_kinds(annotations_path):
    """Beat kinds ('b'/'db'/...) per line of a midi_score_annotations.txt; the
    k-th line is grid time 50*k."""
    kinds = []
    for line in Path(annotations_path).read_text(encoding="utf-8").strip().split("\n"):
        parts = line.split("\t")
        kinds.append(parts[2].split(",")[0] if len(parts) >= 3 else "?")
    return kinds


class PieceXml:
    """Lazy per-piece music21 parse + annotated-downbeat -> XML-measure mapping.

    The r-th downbeat (1-based) normally corresponds to the r-th *real* measure.
    Three benign structures make the raw XML measure-element count differ from
    the downbeat count, all observed in ASAP and all handled here:
      - a pickup measure (first element shorter than the modal measure length,
        often numbered 0): unannotated, sits before the first downbeat;
      - a trailing final measure whose downbeat the annotations omit;
      - a real measure split across two XML measure elements (page/line-break
        splits): detected as ADJACENT sub-modal-length elements whose durations
        sum to the modal length, and merged.
    Anything that still disagrees after accounting for those is rejected
    (mapping would be misaligned mid-piece) rather than displayed wrong.
    """

    def __init__(self, xml_path, kinds):
        self.xml_path = xml_path
        self.kinds = kinds
        self.db_indices = [i for i, k in enumerate(kinds) if k == "db"]
        self.first_is_db = bool(kinds) and kinds[0] == "db"
        self._score = None
        self._real_measures = None  # list of (xml_num_from, xml_num_to) per real measure
        self.valid = None
        self.invalid_reason = None

    def _ensure_parsed(self):
        if self._score is not None or self.valid is False:
            return
        from collections import Counter
        from fractions import Fraction
        from music21 import converter

        t0 = time.perf_counter()
        self._score = converter.parse(str(self.xml_path))
        measures = list(self._score.parts[0].getElementsByClass("Measure"))
        qls = [Fraction(m.duration.quarterLength).limit_denominator(96) for m in measures]

        # bar length of the time signature ACTIVE at each measure (mixed-meter
        # aware — using a piece-wide modal length misreads e.g. a 4/4 intro in
        # a mostly-6/4 piece as a run of split measures); also track the active
        # notated signature and key per measure number for meter_for_window
        bar_qls = []
        current = None
        cur_ts = None
        cur_ks = None
        self._ts_by_num = {}
        self._ks_by_num = {}
        for m in measures:
            if m.timeSignature is not None:
                current = Fraction(m.timeSignature.barDuration.quarterLength).limit_denominator(96)
                cur_ts = (m.timeSignature.numerator, m.timeSignature.denominator)
            if m.keySignature is not None:
                cur_ks = m.keySignature.sharps
            if current is None:
                current = Counter(qls).most_common(1)[0][0]
            bar_qls.append(current)
            self._ts_by_num[m.number] = cur_ts
            self._ks_by_num[m.number] = cur_ks

        # pickup: first element strictly shorter than its own bar length, or
        # explicitly numbered 0
        has_pickup = bool(measures) and (
            measures[0].number == 0 or 0 < qls[0] < bar_qls[0]
        )
        body = list(zip(measures, qls, bar_qls))
        if has_pickup:
            body = body[1:]

        # group split-measure elements: sub-bar-length neighbors merge until
        # the group's bar length is reached; zero-length elements attach to the
        # current group (music21 parse noise); over-length elements count as
        # one measure
        real = []
        acc_ql = Fraction(0)
        acc_from = None
        acc_bar = None
        for m, ql, bar in body:
            if acc_from is None:
                acc_from, acc_ql, acc_bar = m.number, Fraction(0), bar
            acc_ql += ql
            if acc_ql >= acc_bar or ql > acc_bar:
                real.append((acc_from, m.number))
                acc_from = None
        if acc_from is not None:
            real.append((acc_from, body[-1][0].number))

        n_db = len(self.db_indices)
        nums = [m.number for m in measures]
        tier = None
        # tier A: duration-based reconstruction (handles split measures);
        # allow one trailing unannotated final measure
        if len(real) == n_db or len(real) == n_db + 1:
            self._real_measures = real
            self.valid = True
            tier = "A:durations"
        # tier B: trust the written measure numbers when they are strictly
        # sequential and the element count is downbeats + pickup + optional
        # trailing final (duration reconstruction can fail on parse noise like
        # zero-length measures even though the numbering itself is sound)
        elif (
            all(b == a + 1 for a, b in zip(nums, nums[1:]))
            and 0 <= len(nums) - n_db - (1 if has_pickup else 0) <= 1
        ):
            offset = 1 if has_pickup else 0
            self._real_measures = [
                (nums[i + offset], nums[i + offset]) for i in range(n_db)
            ]
            self.valid = True
            tier = "B:numbers"
        else:
            self.valid = False
            self.invalid_reason = (
                f"reconstructed {len(real)} real measures from {len(measures)} XML "
                f"elements but annotations have {n_db} downbeats "
                f"(pickup={has_pickup}, "
                f"sequential_numbers={all(b == a + 1 for a, b in zip(nums, nums[1:]))})"
            )
        print(f"    parsed {self.xml_path} ({time.perf_counter() - t0:.1f}s, "
              f"valid={self.valid}{', tier=' + tier if tier else ''})", flush=True)

    def _real_index_for_beat(self, beat_idx):
        """Index into self._real_measures of the annotated beat (clamped)."""
        n_db_at_or_before = bisect_right(self.db_indices, beat_idx)
        idx = n_db_at_or_before - 1 if n_db_at_or_before > 0 else 0
        return max(0, min(idx, len(self._real_measures) - 1))

    def meter_for_window(self, beat_lo, beat_hi, onset_lo):
        """Real-meter info for engraving the predicted rollout in the piece's
        actual meter/key with barlines on the real downbeat phase, or None when
        the window's span isn't uniform enough to declare a single meter (the
        JS then falls back to the exact-grid 4/4 export).

        The rollout grid is 50 bins per ANNOTATED beat, so the notated meter is
        recovered as (time signature active over the window's measures, annotated
        beats per measure = downbeat spacing). offset_bins is the window's t=0
        phase within its first measure (windows anchor to their first matched
        score note, not to a barline)."""
        self._ensure_parsed()
        if not self.valid or not self.db_indices:
            return None
        last_beat = len(self.kinds) - 1
        ri_lo = self._real_index_for_beat(min(beat_lo, last_beat))
        ri_hi = self._real_index_for_beat(min(beat_hi, last_beat))
        if ri_lo >= len(self.db_indices):
            return None  # window entirely inside a trailing unannotated measure
        gaps = [
            self.db_indices[i + 1] - self.db_indices[i]
            for i in range(ri_lo, min(ri_hi + 1, len(self.db_indices) - 1))
        ]
        if not gaps or any(g != gaps[0] for g in gaps):
            return None
        beats_per_measure = gaps[0]
        sigs = {
            self._ts_by_num.get(self._real_measures[ri][0])
            for ri in range(ri_lo, ri_hi + 1)
        }
        if len(sigs) != 1 or None in sigs:
            return None
        beats, beat_type = sigs.pop()
        # 1/12-beat grid units per quarter note must come out integral
        if beats <= 0 or (3 * beats_per_measure * beat_type) % beats != 0:
            return None
        offset_bins = onset_lo - BINS_PER_BEAT * self.db_indices[ri_lo]
        if not (0 <= offset_bins < BINS_PER_BEAT * beats_per_measure):
            return None
        fifths = self._ks_by_num.get(self._real_measures[ri_lo][0])
        return {
            "beats": beats,
            "beat_type": beat_type,
            "beats_per_measure": beats_per_measure,
            "offset_bins": offset_bins,
            "fifths": fifths if fifths is not None else 0,
            "measure_number_from": self._real_measures[ri_lo][0],
        }

    def excerpt_xml(self, beat_lo, beat_hi):
        self._ensure_parsed()
        if not self.valid:
            return None, None, None
        from music21.musicxml.m21ToXml import GeneralObjectExporter
        last_beat = len(self.kinds) - 1
        m_from = self._real_measures[self._real_index_for_beat(min(beat_lo, last_beat))][0]
        m_to = self._real_measures[self._real_index_for_beat(min(beat_hi, last_beat))][1]
        ex = self._score.measures(m_from, m_to)
        try:
            xml = GeneralObjectExporter().parse(ex).decode("utf-8")
        except Exception as exc:  # noqa: BLE001
            # Some ASAP scores hold durations music21 cannot re-express as
            # MusicXML note types ("inexpressible durations"). Refuse this one
            # window the same way an unreconcilable downbeat mapping is refused,
            # rather than killing the run at the very last step.
            print(f"    excerpt export failed (measures {m_from}-{m_to}): "
                  f"{type(exc).__name__}: {exc}")
            return None, None, None
        return _sanitize_for_osmd(xml), m_from, m_to


def _sanitize_for_osmd(xml):
    """Remove <direction> elements with no visual content (their direction-types
    hold only an empty <words/>) — a music21 export artifact for bare tempo
    <sound> carriers that crashes OSMD's createMetronomeMark."""
    import xml.etree.ElementTree as ET

    body_start = xml.find("<score-partwise")
    if body_start < 0:
        return xml
    root = ET.fromstring(xml[body_start:])
    for parent in root.iter():
        for child in [c for c in parent if c.tag == "direction"]:
            visual = False
            for dt in child.findall("direction-type"):
                for el in dt:
                    if el.tag != "words" or len(el) or (el.text and el.text.strip()):
                        visual = True
            if not visual:
                parent.remove(child)
    return xml[:body_start] + ET.tostring(root, encoding="unicode")


def main():
    global BINS_PER_BEAT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="visualizer/data.js")
    parser.add_argument("--output", default="visualizer/scores_xml.js")
    parser.add_argument(
        "--cache-dir", default=None,
        help="aligned-stream cache dir (default data/asap_aligned_stream_cache). Must "
             "match the cache --data was tokenized against, e.g. "
             "data/asap_aligned_stream_cache_b048 for the 0.48s run.",
    )
    parser.add_argument(
        "--bins-per-beat", type=int, default=BINS_PER_BEAT,
        help="grid bins per annotated score beat = beat_seconds * 100 (50 for the "
             "default 0.5s grid, 48 for the 0.48s grid). Score onsets in --data are in "
             "these units, so this must match how --data was tokenized.",
    )
    parser.add_argument(
        "--var-name", default="VISUALIZER_SCORES",
        help="global the sidecar assigns to (window.<var-name> = {...}). Use a distinct "
             "name (e.g. VISUALIZER_SCORES_B048) when a page loads more than one sidecar.",
    )
    args = parser.parse_args()
    BINS_PER_BEAT = args.bins_per_beat

    t_start = time.perf_counter()
    payload = load_payload(args.data)
    examples = payload["examples"]

    cache_dir = Path(args.cache_dir) if args.cache_dir else STREAM_CACHE_DIR
    print(f"Loading aligned-stream cache (with score spans) from {cache_dir} "
          f"[{BINS_PER_BEAT} bins/beat]...", flush=True)
    pieces = load_cache_pieces_with_scores(cache_dir)
    print(f"  {len(pieces)} cached pieces", flush=True)

    piece_xml_cache = {}
    windows_out = {}
    n_ok = n_skip = 0

    for wid, ex in examples.items():
        controls = [(n["t"], n["d"], n["p"]) for n in ex["perf_notes"]]
        piece, start = locate_window(pieces, controls)
        if piece is None:
            print(f"  {wid}: window not located in any cached piece — skipped", flush=True)
            windows_out[wid] = None
            n_skip += 1
            continue

        n_slots = len(ex["gt_score"])
        ftr = piece["filtered_to_raw"]
        spans = [piece["score_spans"][ftr[start + s]] for s in range(n_slots)
                 if start + s < len(ftr)]
        spans = [s for s in spans if s is not None]
        if not spans:
            windows_out[wid] = None
            n_skip += 1
            continue
        onset_lo = min(t for t, _ in spans)
        onset_hi = max(t + max(d, 1) for t, d in spans)
        beat_lo = onset_lo // BINS_PER_BEAT
        beat_hi = (onset_hi + BINS_PER_BEAT - 1) // BINS_PER_BEAT

        xml_path = Path(piece["score_midi_path"]).parent / "xml_score.musicxml"
        if not xml_path.exists() or not piece["score_beats_path"]:
            print(f"  {wid}: no xml_score.musicxml for {piece['piece_id']} — skipped", flush=True)
            windows_out[wid] = None
            n_skip += 1
            continue

        key = str(xml_path)
        if key not in piece_xml_cache:
            piece_xml_cache[key] = PieceXml(xml_path, load_beat_kinds(piece["score_beats_path"]))
        pxml = piece_xml_cache[key]
        xml, m_from, m_to = pxml.excerpt_xml(beat_lo, beat_hi)
        if xml is None:
            print(f"  {wid}: {pxml.invalid_reason} — skipped", flush=True)
            windows_out[wid] = None
            n_skip += 1
            continue

        meter = pxml.meter_for_window(beat_lo, beat_hi, onset_lo)
        windows_out[wid] = {
            "xml": xml,
            "measure_from": m_from,
            "measure_to": m_to,
            "beat_from": beat_lo,
            "beat_to": beat_hi,
            "piece": ex.get("piece"),
            "meter": meter,
        }
        n_ok += 1
        meter_desc = (
            f"{meter['beats']}/{meter['beat_type']}, {meter['beats_per_measure']} beats/bar, "
            f"offset {meter['offset_bins']} bins, fifths {meter['fifths']}"
            if meter else "none (rollout engraving falls back to 4/4 exact grid)"
        )
        print(f"  {wid}: measures {m_from}-{m_to} (beats {beat_lo}-{beat_hi}, "
              f"{len(xml) / 1024:.0f}KB, meter: {meter_desc})", flush=True)

    out = {"format": 2, "bins_per_beat": BINS_PER_BEAT, "windows": windows_out}
    with open(args.output, "w", encoding="utf-8") as handle:
        handle.write(f"window.{args.var_name} = ")
        json.dump(out, handle)
        handle.write(";\n")
    print(f"Wrote {args.output}: {n_ok} windows with real XML, {n_skip} skipped "
          f"({time.perf_counter() - t_start:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
