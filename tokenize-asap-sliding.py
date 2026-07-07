"""
Tokenize ASAP dataset using sliding window to extract all possible packed 1020-token sequences.

This creates multiple training examples from each piece by starting the interleaving
process at every valid position that has enough remaining notes to form a complete
packed sequence.

Performance notes that the aligner could not match to a score note are DROPPED from
the stream entirely (they appear in neither the prefix nor the body), so every control
triplet in a packed sequence has a corresponding real score triplet. The only REST
placeholders left are the structural ones paired with the prefix controls.

Score normalization ENFORCES 0.5 second beat spacing regardless of original tempo.
Performance/control times preserve original tempo but are shifted to start at 0.

Run directly: ``python tokenize-asap-sliding.py``
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool

from anticipation.config import *
from anticipation.vocab import *
from anticipation.asap_aligned_stream import (
    STREAM_CACHE_DIR,
    build_performance_anchored_stream,
)
from anticipation.packed_sequence import (
    ALTERNATING_START,
    PREFIX_CONTROLS,
    dummy_rest_triplet,
    is_dummy_score_triplet,
    is_prefix_placeholder_triplet,
    is_real_score_triplet,
)

# Fraction of unique scores held out for the test/validation set (90/10 split).
TEST_FRACTION = 0.1
SPLIT_SEED = 42

# Default number of parallel workers (override with --workers).
NUM_WORKERS = 128

# ASAP dataset path
ASAP_PATH = 'asap-dataset-master'
META_CSV = os.path.join(ASAP_PATH, 'metadata.csv')

# Output paths
TRAIN_OUTPUT = 'data/train_normalized.txt'
TEST_OUTPUT = 'data/test_normalized.txt'
SPLIT_FILE = 'data/normalized_split.txt'

# Minimum performance notes required to attempt tokenizing a piece.
MIN_PERFORMANCE_NOTES = 20

# Serialized packed-sequence length (no trailing header/mode tokens).
PACKED_BODY_LENGTH = CONTEXT_SIZE - 4


def _strip_control_offsets(control_triplet):
    return [
        control_triplet[0] - ATIME_OFFSET,
        control_triplet[1] - ADUR_OFFSET,
        control_triplet[2] - ANOTE_OFFSET,
    ]


def _build_score_suffix_min(score_triplets):
    """suffix_min[i] = min score onset (in time units) over score_triplets[i:].

    Used to re-anchor each sliding window's score timeline so its earliest score
    onset lands at 0 (and pickup notes with negative onsets are shifted up).
    """
    suffix_min = [0] * len(score_triplets)
    current = None
    for idx in range(len(score_triplets) - 1, -1, -1):
        score_time_units = score_triplets[idx][0] - TIME_OFFSET
        current = score_time_units if current is None else min(current, score_time_units)
        suffix_min[idx] = current
    return suffix_min


def _clamp_units(value, max_value, overflow):
    """Clamp a time/duration value to [0, max_value); record any clamping.

    Onset/duration units must stay below their range size (MAX_TIME / MAX_DUR) or
    the token would bleed into the next token-type block (e.g. an onset past
    MAX_TIME would alias to a duration token). Clamping keeps the token type valid;
    ``overflow`` accumulates how often and how badly this happened so the caller can
    warn. In practice ASAP windows never exceed the range, so this is an identity
    op on real data and just a safety net.
    """
    value = int(value)
    clamped = max(0, min(max_value - 1, value))
    if clamped != value:
        overflow["count"] += 1
        overflow["max_units"] = max(overflow["max_units"], value)
    return clamped


def _encode_control_triplet(local_time_units, dur_units, pitch_units, overflow):
    return [
        ATIME_OFFSET + _clamp_units(local_time_units, MAX_TIME, overflow),
        ADUR_OFFSET + _clamp_units(dur_units, MAX_DUR, overflow),
        ANOTE_OFFSET + int(pitch_units),
    ]


def _encode_score_triplet(time_units, dur_units, pitch_token, overflow):
    return [
        TIME_OFFSET + _clamp_units(time_units, MAX_TIME, overflow),
        DUR_OFFSET + _clamp_units(dur_units, MAX_DUR, overflow),
        int(pitch_token),
    ]


def tokenize_sliding_windows(filegroup, prefix_controls=PREFIX_CONTROLS):
    """
    Tokenize a single performance-score pair, extracting all possible packed sequences
    using a sliding window approach.

    Matches the exact interleaving logic from tokenize-asap-openings.py but applied at
    multiple starting positions.

    Score times are ENFORCED to have exactly 0.5 seconds between beats (TARGET_BEAT_INTERVAL=0.5).
    This means beat[0]->0.0s, beat[1]->0.5s, beat[2]->1.0s, etc., regardless of original tempo.

    Performance/control times are normalized to start at 0 but keep original tempo.
    Score times are first beat-normalized globally, then shifted to a window-local
    timeline for each sliding window.

    Args:
        filegroup: Tuple of (perf_midi, score_midi, perf_beats, score_beats)
        prefix_controls: Number of control notes in the prefix (default PREFIX_CONTROLS)

    Returns:
        Dict with:
          - piece_id: piece identifier for warnings
          - sequences: list of "token1 token2 ... tokenN | " strings
          - count: number of sequences
          - pickup_warning: aggregated pickup warning metadata or None
          - failure_reason: human-readable reason when count == 0
    """
    file1, file2, file3, file4 = filegroup
    piece_id = file1

    try:
        aligned_items = build_performance_anchored_stream(file1, file2, file3, file4)

        # Drop performance notes with no aligned score note: they are simply absent
        # from the interleaving (no dummy placeholders), so every control triplet in
        # the output has a corresponding real score triplet.
        num_unmatched = sum(1 for item in aligned_items if item["score"] is None)
        aligned_items = [item for item in aligned_items if item["score"] is not None]

        if len(aligned_items) < MIN_PERFORMANCE_NOTES:
            return {
                "piece_id": piece_id,
                "sequences": [],
                "count": 0,
                "pickup_warning": None,
                "overflow_warning": None,
                "num_unmatched_dropped": num_unmatched,
                "failure_reason": (
                    f"only {len(aligned_items)} matched performance notes after dropping "
                    f"{num_unmatched} unmatched (need at least {MIN_PERFORMANCE_NOTES})"
                ),
            }

        sequences = []
        num_items = len(aligned_items)
        k = min(prefix_controls, num_items)
        pickup_start_indices = []
        most_negative_score_time_units = None
        overflow = {"count": 0, "max_units": 0}
        full_length_candidates = 0
        raw_perf_triplets = [_strip_control_offsets(item["control"]) for item in aligned_items]
        score_triplets = [item["score"] for item in aligned_items]
        score_suffix_min_times = _build_score_suffix_min(score_triplets)

        # Try different starting positions
        for start_idx in range(num_items):
            remaining = num_items - start_idx

            # The body emits strict (score, control) pairs: score i is paired with
            # control i + k (its own control appeared k slots earlier, the first k
            # of them in the prefix). Emitting lone score triplets after controls
            # run out would break the every-6-tokens slot structure, so the window
            # ends with the last full pair.
            num_body_pairs = remaining - k
            window_tokens = k * 6 + num_body_pairs * 6
            if window_tokens < PACKED_BODY_LENGTH:
                break  # Later start positions are even shorter.

            perf_anchor = raw_perf_triplets[start_idx][0]
            min_score_time_units = score_suffix_min_times[start_idx]
            if min_score_time_units < 0:
                pickup_start_indices.append(start_idx)
                if (
                    most_negative_score_time_units is None
                    or min_score_time_units < most_negative_score_time_units
                ):
                    most_negative_score_time_units = min_score_time_units

            interleaved_tokens = []

            def _append_control(item_idx):
                perf_triplet = raw_perf_triplets[item_idx]
                interleaved_tokens.extend(
                    _encode_control_triplet(
                        perf_triplet[0] - perf_anchor,
                        perf_triplet[1],
                        perf_triplet[2],
                        overflow,
                    )
                )

            def _append_score(item_idx):
                score_triplet = score_triplets[item_idx]
                # Re-anchor onset to the window-local timeline, then clamp.
                interleaved_tokens.extend(
                    _encode_score_triplet(
                        (score_triplet[0] - min_score_time_units) - TIME_OFFSET,
                        score_triplet[1] - DUR_OFFSET,
                        score_triplet[2],
                        overflow,
                    )
                )

            # Prefix: (control, REST placeholder) pairs for the first k notes.
            for i in range(k):
                _append_control(start_idx + i)
                interleaved_tokens.extend(dummy_rest_triplet(0))

            # Body: strictly alternating (score i, control i + k) pairs, stopping
            # once the sequence is full.
            for i in range(num_body_pairs):
                _append_score(start_idx + i)
                _append_control(start_idx + i + k)
                if len(interleaved_tokens) >= PACKED_BODY_LENGTH:
                    break

            full_length_candidates += 1
            sequence = interleaved_tokens[:PACKED_BODY_LENGTH]
            assert len(sequence) == PACKED_BODY_LENGTH, (
                f"Expected {PACKED_BODY_LENGTH} tokens, got {len(sequence)}"
            )

            token_str = ' '.join(str(tok) for tok in sequence)
            sequences.append(f"{token_str} | ")

        pickup_warning = None
        if pickup_start_indices:
            pickup_warning = {
                "piece_id": piece_id,
                "affected_windows": len(pickup_start_indices),
                "sample_start_indices": pickup_start_indices[:5],
                "most_negative_time_units": most_negative_score_time_units,
            }

        overflow_warning = None
        if overflow["count"] > 0:
            overflow_warning = {
                "piece_id": piece_id,
                "clamped_tokens": overflow["count"],
                "max_units": overflow["max_units"],
            }

        failure_reason = None
        if not sequences:
            if full_length_candidates == 0:
                failure_reason = (
                    f"piece is too short to form a full packed sequence of {PACKED_BODY_LENGTH} tokens"
                )
            else:
                failure_reason = "no valid packed windows generated"

        return {
            "piece_id": piece_id,
            "sequences": sequences,
            "count": len(sequences),
            "pickup_warning": pickup_warning,
            "overflow_warning": overflow_warning,
            "num_unmatched_dropped": num_unmatched,
            "failure_reason": failure_reason,
        }

    except Exception as e:
        return {
            "piece_id": piece_id,
            "sequences": [],
            "count": 0,
            "pickup_warning": None,
            "overflow_warning": None,
            "num_unmatched_dropped": 0,
            "failure_reason": f"{type(e).__name__}: {e}",
        }


def format_pickup_warning(warning_info):
    preview_indices = ", ".join(str(idx) for idx in warning_info["sample_start_indices"])
    if warning_info["affected_windows"] > len(warning_info["sample_start_indices"]):
        preview_indices = f"{preview_indices}, ..."
    most_negative_seconds = warning_info["most_negative_time_units"] / TIME_RESOLUTION
    return (
        f"WARNING: {warning_info['piece_id']}: shifted score times for pickup preservation in "
        f"{warning_info['affected_windows']} window(s) (start_idx examples: {preview_indices}); "
        f"most negative pre-shift score onset was {warning_info['most_negative_time_units']} bins "
        f"({most_negative_seconds:.3f}s)."
    )


def format_overflow_warning(warning_info):
    max_seconds = warning_info["max_units"] / TIME_RESOLUTION
    return (
        f"WARNING: {warning_info['piece_id']}: clamped {warning_info['clamped_tokens']} "
        f"onset/duration token(s) that exceeded the representable range "
        f"(largest was {warning_info['max_units']} bins / {max_seconds:.3f}s; "
        f"MAX_TIME={MAX_TIME} bins). Window(s) spanned longer than the vocab allows."
    )


def format_failure_warning(piece_id, failure_reason):
    return f"WARNING: {piece_id}: tokenization produced no sequences - {failure_reason}"


def process_single_piece(filegroup):
    """Worker function for multiprocessing. Returns a dict from tokenize_sliding_windows()."""
    return tokenize_sliding_windows(filegroup)


def load_valid_filegroups():
    """Load ASAP metadata and return (filegroups, score_keys, piece_names) for pieces
    whose performance/score MIDI and both annotation files all exist."""
    df = pd.read_csv(META_CSV)
    print(f"Found {len(df)} pieces in metadata")

    datafiles = []
    score_keys = []   # use midi_score path as the split key
    piece_names = []  # track piece names for the split file

    for _, row in df.iterrows():
        file1 = os.path.join(ASAP_PATH, row['midi_performance'])
        file2 = os.path.join(ASAP_PATH, row['midi_score'])
        file3 = os.path.join(ASAP_PATH, row['performance_annotations'])
        file4 = os.path.join(ASAP_PATH, row['midi_score_annotations'])

        if all(os.path.exists(f) for f in [file1, file2, file3, file4]):
            datafiles.append((file1, file2, file3, file4))
            score_keys.append(file2)
            piece_names.append(row['midi_performance'])

    print(f"Found {len(datafiles)} valid pieces with all required files")
    return datafiles, score_keys, piece_names


def split_by_score(datafiles, score_keys, piece_names, test_frac=TEST_FRACTION, seed=SPLIT_SEED):
    """Split pieces into train/test by unique score to avoid data leakage."""
    rng = np.random.default_rng(seed)
    unique_scores = list(sorted(set(score_keys)))
    rng.shuffle(unique_scores)
    n_test = int(np.ceil(test_frac * len(unique_scores)))
    test_scores = set(unique_scores[:n_test])

    train_pairs, test_pairs = [], []
    train_piece_names, test_piece_names = [], []
    for filegroup, score, piece_name in zip(datafiles, score_keys, piece_names):
        if score in test_scores:
            test_pairs.append(filegroup)
            test_piece_names.append(piece_name)
        else:
            train_pairs.append(filegroup)
            train_piece_names.append(piece_name)

    return train_pairs, test_pairs, train_piece_names, test_piece_names


def write_split_file(datafiles, train_pairs, test_pairs, train_piece_names, test_piece_names):
    print(f"Writing split information to {SPLIT_FILE}...")
    with open(SPLIT_FILE, 'w') as f:
        f.write(f"# Train/Test Split (seed={SPLIT_SEED}, test_frac={TEST_FRACTION})\n")
        f.write(
            f"# Total pieces: {len(datafiles)} "
            f"(train: {len(train_pairs)}, test: {len(test_pairs)})\n"
        )
        f.write("# Split by unique scores to prevent data leakage\n")
        f.write("# Strategy: Sliding window - all possible packed 1020-token sequences from each piece\n")
        f.write("# Unmatched performance notes are dropped (no dummy score placeholders in the body)\n\n")

        f.write("=== TRAINING PIECES ===\n")
        for piece_name in sorted(train_piece_names):
            f.write(f"./{piece_name}\n")

        f.write("\n=== TEST PIECES ===\n")
        for piece_name in sorted(test_piece_names):
            f.write(f"./{piece_name}\n")
    print(f"Split file written: {SPLIT_FILE}\n")


def tokenize_split(pairs, output_path, desc, workers=NUM_WORKERS):
    """Tokenize a list of pieces in parallel, streaming results to output_path.

    Returns (num_sequences, num_pieces_success, num_pieces_failed).
    """
    sequences_total = 0
    pieces_success = 0
    pieces_failed = 0
    unmatched_dropped_total = 0

    with open(output_path, 'w') as out_file:
        with Pool(processes=workers) as pool:
            with tqdm(total=len(pairs), desc=desc, unit='piece') as pbar:
                for result in pool.imap_unordered(process_single_piece, pairs):
                    if result["pickup_warning"] is not None:
                        tqdm.write(format_pickup_warning(result["pickup_warning"]))
                    if result.get("overflow_warning") is not None:
                        tqdm.write(format_overflow_warning(result["overflow_warning"]))
                    unmatched_dropped_total += result.get("num_unmatched_dropped", 0)
                    if result["count"] > 0:
                        for seq in result["sequences"]:
                            out_file.write(seq + '\n')
                        sequences_total += result["count"]
                        pieces_success += 1
                    else:
                        pieces_failed += 1
                        tqdm.write(format_failure_warning(result["piece_id"], result["failure_reason"]))
                    pbar.update(1)

    print(f"{desc}: dropped {unmatched_dropped_total} unmatched performance notes across all pieces")
    return sequences_total, pieces_success, pieces_failed


def verify_first_sequence(output_path):
    """Print a breakdown of the first generated training sequence for sanity checking."""
    with open(output_path, 'r') as f:
        first_line = f.readline().strip()
    tokens_part = first_line.split('|')[0].strip()
    first_seq = [int(x) for x in tokens_part.split()]

    print(f"First training sequence length: {len(first_seq)} tokens")
    print("No mode/bootstrap tokens are serialized")

    control_count = score_count = dummy_score_count = prefix_rest_count = 0
    for i in range(min(100, len(first_seq) // 3)):
        pos = i * 3
        if pos + 2 >= len(first_seq):
            break
        if first_seq[pos] >= CONTROL_OFFSET:
            control_count += 1
        elif is_prefix_placeholder_triplet(first_seq, pos, ALTERNATING_START):
            prefix_rest_count += 1
        elif is_dummy_score_triplet(first_seq, pos, ALTERNATING_START):
            dummy_score_count += 1
        elif is_real_score_triplet(first_seq, pos, ALTERNATING_START):
            score_count += 1

    print("\nFirst 100 triplets breakdown:")
    print(f"  Control triplets: {control_count}")
    print(f"  Score triplets (notes): {score_count}")
    print(f"  Score triplets (dummy REST): {dummy_score_count}")
    print(f"  Prefix placeholders (REST): {prefix_rest_count}")
    if dummy_score_count > 0:
        print(
            "WARNING: found dummy REST score triplets in the body; unmatched "
            "performance notes should have been dropped."
        )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Tokenize the ASAP dataset into packed sliding-window sequences."
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=NUM_WORKERS,
        help=f"Number of parallel worker processes (default: {NUM_WORKERS}).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    print("Tokenization configuration:")
    print(f"  Workers: {args.workers}")
    print(f"  Context size: {CONTEXT_SIZE}")
    print(f"  Serialized length: {PACKED_BODY_LENGTH}")
    print(f"  Prefix controls: {PREFIX_CONTROLS} (fixed)")
    print("  Strategy: Sliding window over all piece positions")
    print("  Output format: space-separated tokens (one sequence per line)")
    print(f"  Aligned-stream cache: {STREAM_CACHE_DIR}")
    print()

    datafiles, score_keys, piece_names = load_valid_filegroups()
    train_pairs, test_pairs, train_piece_names, test_piece_names = split_by_score(
        datafiles, score_keys, piece_names
    )
    print(f"Train: {len(train_pairs)} pieces")
    print(f"Test: {len(test_pairs)} pieces\n")

    write_split_file(datafiles, train_pairs, test_pairs, train_piece_names, test_piece_names)

    os.makedirs('data', exist_ok=True)

    print("Processing training set...")
    train_total, train_success, train_failed = tokenize_split(
        train_pairs, TRAIN_OUTPUT, 'Train', workers=args.workers
    )
    print(f"Train: {train_total} sequences from {train_success} pieces, {train_failed} pieces failed")

    print("\nProcessing test set...")
    test_total, test_success, test_failed = tokenize_split(
        test_pairs, TEST_OUTPUT, 'Test', workers=args.workers
    )
    print(f"Test: {test_total} sequences from {test_success} pieces, {test_failed} pieces failed")

    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)
    if train_total > 0:
        verify_first_sequence(TRAIN_OUTPUT)
    else:
        print("No sequences generated!")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Training sequences: {train_total} from {train_success}/{len(train_pairs)} pieces")
    if train_success > 0:
        print(f"  Average sequences per piece: {train_total / train_success:.1f}")
    print(f"Test sequences: {test_total} from {test_success}/{len(test_pairs)} pieces")
    if test_success > 0:
        print(f"  Average sequences per piece: {test_total / test_success:.1f}")
    print(f"Total sequences: {train_total + test_total}")
    print("\nOutput files:")
    print(f"  {TRAIN_OUTPUT}")
    print(f"  {TEST_OUTPUT}")
    print(f"  {SPLIT_FILE}")
    print("\nTokenization complete!")


if __name__ == '__main__':
    main()
