#!/usr/bin/env python
"""Reproduce the ASAP train/validation/test split used by BOTH

  * Beyer & Dai, "End-to-end Piano Performance-MIDI to Score Conversion with
    Transformers" (ISMIR 2024)  -- github.com/TimFelixBeyer/MIDI2ScoreTransformer
  * Zeng, Zhao & Wang, "Bridging Piano Transcription and Rendering via
    Disentangled Score Content and Style" (ICLR 2026)
    -- github.com/wei-zeng98/joint-apt-epr

The two papers use a BYTE-IDENTICAL split: joint-apt-epr copied Beyer's
`constants.py` verbatim (its docstring says "From Beyer's paper") and both run
the same filtering pipeline in `dataset.py::_load_metadata`. This script is a
faithful re-implementation of that pipeline, emitting a manifest that
`tokenize-asap-sliding.py --split-input` consumes.

The split is defined over the ACPAS dataset metadata (github.com/cheriell/
ACPAS-dataset), restricted to ASAP performances, and keyed by ACPAS `piece_id`
(so every performance of a given score lands in the same split -- no leakage):

    test        : piece_id in TEST_PIECE_IDS  (first piece per composer)
    validation  : piece_id % 10 == 0  and not a test piece
    train       : piece_id % 10 != 0  and not a test piece

Provenance of the hardcoded index/id lists is preserved from the papers'
`constants.py` (they are positional indices into the filtered+reset frame and a
frozen set of held-out piece ids, so they must be applied in exactly this
order).
"""
import argparse
import json
import os

import pandas as pd

# --- Verbatim from the papers' constants.py -------------------------------
# Not parsable by music21.
SKIP = set(
    ["{ASAP}/Glinka/The_Lark/Denisova10M.mid", "{ASAP}/Glinka/The_Lark/Kleisen07M.mid"]
)
# Not aligned correctly or other issues (positional indices AFTER reset_index).
TO_IGNORE_INDICES = [152, 153, 154, 165, 166, 179, 180, 181, 332, 333, 334, 335, 349, 350,
                     351, 418, 419, 420, 426, 428, 429, 430, 472, 473, 474, 489, 490, 491,
                     516, 517, 518, 519, 520, 521, 522, 540, 541, 560, 609, 774, 798, 799,
                     800, 801, 802, 803, 819, 920, 921, 935, 936, 937, 938, 939, 940, 941,
                     979, 980, 981, 997, 998, 999, 1012, 1013, 1014, 1017, 1018]
# Held-out test pieces (first piece per composer). "To keep eval consistent, we
# hardcode test piece ids here."
TEST_PIECE_IDS = [15, 78, 159, 172, 254, 288, 322, 374, 395, 399, 411, 418, 452, 478]


def build_split_frame(acpas_dir, asap_annotations_path):
    """Run the papers' exact _load_metadata filtering pipeline. Returns the
    filtered ACPAS frame with a 'split' column added."""
    data_real = pd.read_csv(os.path.join(acpas_dir, "metadata_R.csv"))
    data_synthetic = pd.read_csv(os.path.join(acpas_dir, "metadata_S.csv"))
    asap_annotations = json.load(open(asap_annotations_path))
    unaligned = set(
        "{ASAP}/" + k
        for k, v in asap_annotations.items()
        if not v["score_and_performance_aligned"]
    )

    data = pd.concat([data_real, data_synthetic])
    data = data[(data["source"] == "ASAP") & data["aligned"]]
    data = data[~data["performance_MIDI_external"].isin(SKIP)]
    data = data[~data["performance_MIDI_external"].isin(unaligned)]
    data = data.drop_duplicates(subset=["performance_MIDI_external"])
    data.reset_index(inplace=True)
    data.drop(TO_IGNORE_INDICES, inplace=True)

    is_test = data["piece_id"].isin(TEST_PIECE_IDS)
    is_val = (data["piece_id"] % 10 == 0) & (~is_test)
    split = pd.Series("train", index=data.index)
    split[is_val] = "validation"
    split[is_test] = "test"
    data = data.assign(split=split)

    # No leakage guard: every score (piece_id) must live in exactly one split,
    # and each performance path maps to a single score.
    per_piece_splits = data.groupby("piece_id")["split"].nunique()
    assert (per_piece_splits == 1).all(), "a piece_id spans multiple splits!"
    assert not data["performance_MIDI_external"].duplicated().any()
    return data


def rel_path(perf_external):
    """'{ASAP}/Bach/Fugue/bwv_846/Shi05M.mid' -> 'Bach/Fugue/bwv_846/Shi05M.mid'
    (matches asap-dataset-master/metadata.csv's midi_performance column)."""
    return perf_external.replace("{ASAP}/", "")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--acpas-dir", default="data/acpas",
                    help="Directory holding ACPAS metadata_R.csv + metadata_S.csv.")
    ap.add_argument("--asap-annotations",
                    default="asap-dataset-master/asap_annotations.json",
                    help="Path to ASAP's asap_annotations.json.")
    ap.add_argument("--output", default="data/paper_split.txt",
                    help="Split manifest to write (consumed by "
                         "tokenize-asap-sliding.py --split-input).")
    args = ap.parse_args()

    data = build_split_frame(args.acpas_dir, args.asap_annotations)

    groups = {name: sorted(rel_path(p) for p in sub["performance_MIDI_external"])
              for name, sub in data.groupby("split")}
    order = ["train", "validation", "test"]
    counts = {k: len(groups.get(k, [])) for k in order}
    pieces = {name: sub["piece_id"].nunique() for name, sub in data.groupby("split")}

    with open(args.output, "w") as f:
        f.write("# ASAP train/validation/test split -- shared by Beyer&Dai (ISMIR 2024,\n")
        f.write("# MIDI2ScoreTransformer) and Zeng+ (ICLR 2026, joint-apt-epr); the two\n")
        f.write("# papers' split code is byte-identical. Reproduced by make_paper_split.py\n")
        f.write("# from ACPAS metadata (cheriell/ACPAS-dataset) + ASAP asap_annotations.json.\n")
        f.write("# Split by ACPAS piece_id: test=first-piece-per-composer (TEST_PIECE_IDS),\n")
        f.write("# validation=piece_id%%10==0, train=remainder. Paths are relative to\n")
        f.write("# asap-dataset-master/ (== metadata.csv midi_performance).\n")
        f.write(
            f"# Performances: train={counts['train']} val={counts['validation']} "
            f"test={counts['test']}. "
            f"Pieces: train={pieces.get('train',0)} val={pieces.get('validation',0)} "
            f"test={pieces.get('test',0)}.\n\n"
        )
        for name, header in [("train", "TRAIN"), ("validation", "VALIDATION"),
                             ("test", "TEST")]:
            f.write(f"=== {header} PERFORMANCES ===\n")
            for rel in groups.get(name, []):
                f.write(f"./{rel}\n")
            f.write("\n")

    print(f"Wrote {args.output}")
    print(f"  performances: train={counts['train']} "
          f"validation={counts['validation']} test={counts['test']} "
          f"(total {sum(counts.values())})")
    print(f"  pieces:       train={pieces.get('train',0)} "
          f"validation={pieces.get('validation',0)} test={pieces.get('test',0)}")
    print(f"  test piece_ids: {sorted(data[data['split']=='test']['piece_id'].unique().tolist())}")


if __name__ == "__main__":
    main()
