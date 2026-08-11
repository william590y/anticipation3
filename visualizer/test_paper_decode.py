#!/usr/bin/env python
"""Round-trip test for `run_paper_models.decode_score_streams`.

Runs a real `xml_score.musicxml` through paper 1's OWN encoder
(`parse_mxl` -> `bucket_mxl`), feeds the resulting bucket indices to our decoder
as if they were model output, and checks we recover music21's absolute note
offsets, durations and MIDI pitches. This takes model quality out of the picture
entirely: any disagreement is a bug in our decoding of their representation.

It caught exactly that -- an unbucketed `pitch` stream, which left every
predicted note a semitone sharp and drove note-level F1 to ~0.

Must run in the `paperpipe` env (needs their music21 fork):

    conda run -n paperpipe python visualizer/test_paper_decode.py
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAPER1 = REPO / "external/paper1_joint_apt_epr"
DEFAULT_XML = REPO / "asap-dataset-master/Bach/Fugue/bwv_846/xml_score.musicxml"

# Their `config.py` resolves hparams.yaml relative to the CWD.
sys.path.insert(0, str(PAPER1))
os.chdir(PAPER1)

from tokenizer import MultistreamTokenizer as MT  # noqa: E402

sys.path.insert(0, str(REPO / "visualizer"))
from run_paper_models import decode_score_streams  # noqa: E402

# Their bucketing clamps durations to PARAMS["duration"]["max"], so a longer note
# cannot round-trip; compare against the clamped truth rather than the raw value.
DURATION_MAX = 4.0
ONSET_TOL = 1e-3          # quarters; float error in repeated 1/24 steps


def main():
    xml = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_XML
    streams = MT.parse_mxl(str(xml))
    toks = MT.bucket_mxl(streams)
    n = toks["pitch"].shape[0]

    # Mimic a paper-1 model output: batch dim, and the +1 shift that reserves
    # index 0 for <PAD> (bucket_mxl emits raw 0-based indices).
    y = {k: (v + 1).unsqueeze(0) for k, v in toks.items() if k != "pad"}
    dec = decode_score_streams(y, n)

    mxl_list, _ = MT.mxl_to_list(str(xml))
    true_onset = [float(x.offset) for x in mxl_list]
    true_dur = [min(DURATION_MAX, float(v)) for v in streams["duration"][:n]]
    true_pitch = [int(v) for v in streams["pitch"][:n]]

    onset_err = max(abs(dec[i]["onset_q"] - true_onset[i]) for i in range(n))
    dur_err = max(abs(dec[i]["dur_q"] - true_dur[i]) for i in range(n))
    bad_pitch = [i for i in range(n) if dec[i]["pitch"] != true_pitch[i]]

    print(f"{xml.relative_to(REPO)}: {n} notes")
    print(f"  max onset error   : {onset_err:.2e} quarters")
    print(f"  max duration error: {dur_err:.2e} quarters")
    print(f"  pitch mismatches  : {len(bad_pitch)}")
    if bad_pitch:
        i = bad_pitch[0]
        print(f"    e.g. note {i}: decoded {dec[i]['pitch']} vs true {true_pitch[i]}")

    ok = onset_err < ONSET_TOL and dur_err < ONSET_TOL and not bad_pitch
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
