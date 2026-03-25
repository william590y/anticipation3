"""
Tokenize the ASAP dataset using sliding windows to extract interleaved
performance/score sequences without any ATEPP dependency.

This mirrors the ASAP branch of tokenize-combined.py:
- ASAP beat-annotation alignment via align_tokens2
- score beat normalization to a fixed 0.5s beat interval
- composition-key train/test split to avoid cross-performance leakage
- fixed-length packed sequences written one per line
"""

import os
import re
import sys
import unicodedata
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd
from tqdm import tqdm

from anticipation.config import *
from anticipation.vocab import *
from anticipation import ops
from alignment import align_tokens2, load_annotation_file

# Suppress music21-style warnings that can surface via imported utilities.
warnings.filterwarnings('ignore', category=UserWarning)

DEFAULT_MAX_WORKERS = 32
DEFAULT_MAX_TASKS_PER_CHILD = 32
WORKER_ENV_VAR = 'ASAP_TOKENIZE_WORKERS'
MAX_TASKS_ENV_VAR = 'ASAP_TOKENIZE_MAX_TASKS_PER_CHILD'


def _read_positive_int_env(var_name):
    raw_value = os.environ.get(var_name)
    if raw_value is None:
        return None

    try:
        value = int(raw_value)
    except ValueError:
        warnings.warn(f"Ignoring invalid {var_name}={raw_value!r}; expected a positive integer.")
        return None

    if value <= 0:
        warnings.warn(f"Ignoring invalid {var_name}={raw_value!r}; expected a positive integer.")
        return None

    return value


def _resolve_num_workers():
    cpu_count = os.cpu_count() or 1
    env_override = _read_positive_int_env(WORKER_ENV_VAR)
    if env_override is not None:
        return env_override

    workers = min(cpu_count, DEFAULT_MAX_WORKERS)

    # Multiprocessing pools use several pipes/queues per worker. On shared
    # clusters with a modest RLIMIT_NOFILE, a CPU-count-sized pool can fail
    # during teardown/recreation with "Too many open files".
    try:
        import resource

        soft_limit, _ = resource.getrlimit(resource.RLIMIT_NOFILE)
        if soft_limit > 0:
            fd_headroom = 128
            approx_fds_per_worker = 8
            fd_capped_workers = max(1, (soft_limit - fd_headroom) // approx_fds_per_worker)
            workers = min(workers, fd_capped_workers)
    except (ImportError, OSError, ValueError):
        pass

    return max(1, workers)


def _resolve_max_tasks_per_child():
    env_override = _read_positive_int_env(MAX_TASKS_ENV_VAR)
    if env_override is not None:
        return env_override
    return DEFAULT_MAX_TASKS_PER_CHILD


# Number of parallel workers
NUM_WORKERS = _resolve_num_workers()
POOL_MAX_TASKS_PER_CHILD = _resolve_max_tasks_per_child()

# Dataset paths
ASAP_PATH = 'asap-dataset-master'
ASAP_META_CSV = os.path.join(ASAP_PATH, 'metadata.csv')

# Output paths
TRAIN_OUTPUT = 'data/train_asap_only.txt'
TEST_OUTPUT = 'data/test_asap_only.txt'
SPLIT_FILE = 'data/asap_only_split.txt'

PACKED_SEQUENCE_LENGTH = CONTEXT_SIZE - 4
TARGET_BEAT_INTERVAL = 0.5
TEST_SPLIT_RATIO = 0.1

print("ASAP-Only Tokenization")
print("=" * 60)
print("Configuration:")
print(f"  Workers: {NUM_WORKERS}")
print(f"  Max tasks/child: {POOL_MAX_TASKS_PER_CHILD}")
print(f"  Context size: {CONTEXT_SIZE}")
print(f"  Serialized length: {PACKED_SEQUENCE_LENGTH}")
print("  Prefix controls: 33 (fixed)")
print("  Output format: space-separated tokens (one sequence per line)")
print()


def _normalize_composer(name):
    """Map full or short composer name to a consistent lowercase last name."""
    name_map = {
        'Bach': 'bach', 'Johann Sebastian Bach': 'bach',
        'Beethoven': 'beethoven', 'Ludwig van Beethoven': 'beethoven',
        'Brahms': 'brahms', 'Johannes Brahms': 'brahms',
        'Chopin': 'chopin', 'Frédéric Chopin': 'chopin', 'Frederic Chopin': 'chopin',
        'Debussy': 'debussy', 'Claude Debussy': 'debussy',
        'Haydn': 'haydn', 'Franz Joseph Haydn': 'haydn',
        'Liszt': 'liszt', 'Franz Liszt': 'liszt',
        'Mozart': 'mozart', 'Wolfgang Amadeus Mozart': 'mozart',
        'Rachmaninoff': 'rachmaninoff', 'Sergei Rachmaninoff': 'rachmaninoff',
        'Ravel': 'ravel', 'Maurice Ravel': 'ravel',
        'Scarlatti': 'scarlatti', 'Domenico Scarlatti': 'scarlatti',
        'Schubert': 'schubert', 'Franz Schubert': 'schubert',
        'Schumann': 'schumann', 'Robert Schumann': 'schumann',
        'Scriabin': 'scriabin', 'Alexander Scriabin': 'scriabin',
        'Balakirev': 'balakirev', 'Glinka': 'glinka', 'Prokofiev': 'prokofiev',
    }
    return name_map.get(name, name.split()[-1].lower())


_GENERIC_WORK_WORDS = {
    'etude', 'etudes', 'prelude', 'preludes', 'nocturne', 'nocturnes',
    'fugue', 'fugues', 'toccata', 'toccatas',
    'impromptu', 'impromptus', 'moment', 'moments', 'morceaux',
    'waltz', 'waltzes', 'mazurka', 'mazurkas', 'polonaise', 'polonaises',
    'intermezzo', 'intermezzi', 'caprice', 'caprices',
    'variation', 'variations', 'theme',
    'sonata', 'sonatina', 'concerto', 'symphony', 'quartet', 'quintet',
    'suite', 'piece', 'pieces', 'collection', 'album',
    'book', 'volume', 'heft',
    'piano', 'keyboard', 'clavier', 'klavier', 'wohltemperierte',
    'tempered', 'welltemp',
    'grand', 'grande', 'petit', 'petite', 'little', 'great', 'kleine',
    'first', 'second', 'third', 'fourth', 'fifth',
    'execution', 'exécution', 'pour', 'dans', 'avec', 'sans', 'sous',
    'vers', 'nach', 'über',
    'major', 'minor', 'sharp', 'flat', 'natural',
}


def _extract_catalog(text):
    """Extract a work-level catalog key from ASAP metadata text."""
    text = text.lower().replace('_', ' ').replace('-', ' ')

    bwv_matches = re.findall(r'\bbwv\s*(\d+)', text)
    if bwv_matches:
        return f"bwv{bwv_matches[-1]}"

    match = re.search(r'\b(?:piano|keyboard)\s*sonatas?\s*(\d+)', text)
    if match:
        return f"son{match.group(1)}"

    match = re.search(r'\bsonata\s*no[\s.]*(\d+)', text)
    if match:
        return f"son{match.group(1)}"

    match = re.search(r'\bsonata[\s_]*(\d+)', text)
    if match:
        return f"son{match.group(1)}"

    match = re.search(r'\bhob[.\s]*xvi[.\s:]*(\d+)', text)
    if match:
        return f"son{match.group(1)}"

    match = re.search(r'\bop[\s.]*(\d+)\b', text)
    if match:
        return f"op{match.group(1)}"

    match = re.search(r'\bk[v]?[\s.]*(\d+)\b', text)
    if match:
        return f"k{match.group(1)}"

    match = re.search(r'\bd[\s.]+(\d{3,})\b', text)
    if match:
        return f"d{match.group(1)}"

    match = re.search(r'\bs[\s.]*(\d{3,})\b', text)
    if match:
        return f"s{match.group(1)}"

    return None


_ASAP_WORK_OVERRIDES = {
    ('Liszt', 'Sonata'): 'liszt__s178',
}


def composition_key_asap(composer, title, midi_score):
    """Canonical composition key for ASAP entries."""
    override = _ASAP_WORK_OVERRIDES.get((composer, title))
    if override:
        return override

    composer_key = _normalize_composer(composer)

    catalog = _extract_catalog(title)
    if catalog:
        return f"{composer_key}__{catalog}"

    parts = midi_score.replace('\\', '/').split('/')
    if len(parts) >= 2:
        work_folder = parts[-2]
        catalog = _extract_catalog(work_folder)
        if catalog:
            return f"{composer_key}__{catalog}"

    return f"asap__{midi_score}"


def load_asap_data():
    """Load valid ASAP piece tuples plus metadata needed for splitting."""
    if not os.path.exists(ASAP_META_CSV):
        print(f"ERROR: ASAP metadata not found at {ASAP_META_CSV}")
        sys.exit(1)

    df_asap = pd.read_csv(ASAP_META_CSV)
    print(f"[ASAP] Found {len(df_asap)} pieces in metadata")

    datafiles = []
    composition_keys = []
    piece_names = []

    for _, row in df_asap.iterrows():
        perf_midi = os.path.join(ASAP_PATH, row['midi_performance'])
        score_midi = os.path.join(ASAP_PATH, row['midi_score'])
        perf_beats = os.path.join(ASAP_PATH, row['performance_annotations'])
        score_beats = os.path.join(ASAP_PATH, row['midi_score_annotations'])

        if all(os.path.exists(path) for path in [perf_midi, score_midi, perf_beats, score_beats]):
            datafiles.append(('asap', perf_midi, score_midi, perf_beats, score_beats))
            composition_keys.append(
                composition_key_asap(row['composer'], row['title'], row['midi_score'])
            )
            piece_names.append(row['midi_performance'])

    print(f"[ASAP] Found {len(datafiles)} valid pieces with all required files")
    return datafiles, composition_keys, piece_names


def split_by_composition(datafiles, composition_keys, piece_names):
    """Create a reproducible 90/10 train/test split by composition key."""
    rng = np.random.default_rng(42)
    unique_comp_keys = list(sorted(set(composition_keys)))
    rng.shuffle(unique_comp_keys)
    n_test = int(np.ceil(TEST_SPLIT_RATIO * len(unique_comp_keys)))
    test_comp_keys = set(unique_comp_keys[:n_test])

    train_pairs = []
    test_pairs = []
    train_piece_names = []
    test_piece_names = []

    for datafile, comp_key, piece_name in zip(datafiles, composition_keys, piece_names):
        if comp_key in test_comp_keys:
            test_pairs.append(datafile)
            test_piece_names.append(piece_name)
        else:
            train_pairs.append(datafile)
            train_piece_names.append(piece_name)

    return train_pairs, test_pairs, train_piece_names, test_piece_names


def write_split_file(total_pieces, train_piece_names, test_piece_names):
    """Write the ASAP-only train/test split summary to disk."""
    print(f"Writing split information to {SPLIT_FILE}...")
    with open(SPLIT_FILE, 'w') as split_file:
        split_file.write(
            f"# Total pieces: {total_pieces} (train: {len(train_piece_names)}, "
            f"test: {len(test_piece_names)})\n"
        )
        split_file.write("# ASAP-only split\n\n")

        split_file.write("=== TRAINING PIECES ===\n")
        for piece_name in sorted(train_piece_names):
            split_file.write(f"./{piece_name}\n")

        split_file.write("\n=== TEST PIECES ===\n")
        for piece_name in sorted(test_piece_names):
            split_file.write(f"./{piece_name}\n")

    print(f"Split file written: {SPLIT_FILE}\n")


def _normalize_score_time_sec(original_time_sec, score_beat_times):
    """Map a raw score time into the fixed 0.5s-per-beat timeline."""
    if score_beat_times and len(score_beat_times) >= 2:
        if original_time_sec < score_beat_times[0]:
            beat_duration = score_beat_times[1] - score_beat_times[0]
            if beat_duration > 0:
                progress = (original_time_sec - score_beat_times[0]) / beat_duration
            else:
                progress = 0.0
            return progress * TARGET_BEAT_INTERVAL

        for i in range(len(score_beat_times) - 1):
            if score_beat_times[i] <= original_time_sec <= score_beat_times[i + 1]:
                beat_duration = score_beat_times[i + 1] - score_beat_times[i]
                if beat_duration > 0:
                    progress = (original_time_sec - score_beat_times[i]) / beat_duration
                else:
                    progress = 0.0
                return i * TARGET_BEAT_INTERVAL + progress * TARGET_BEAT_INTERVAL

        last_beat_idx = len(score_beat_times) - 1
        last_beat_duration = score_beat_times[-1] - score_beat_times[-2]
        if last_beat_duration > 0:
            progress = (original_time_sec - score_beat_times[-1]) / last_beat_duration
        else:
            progress = 0.0
        return last_beat_idx * TARGET_BEAT_INTERVAL + progress * TARGET_BEAT_INTERVAL

    return original_time_sec - (score_beat_times[0] if score_beat_times else 0.0)


def tokenize_sliding_windows_asap(filegroup, prefix_controls=33):
    """
    Tokenize an ASAP performance-score pair using beat annotations.
    """
    _, perf_midi, score_midi, perf_beats, score_beats = filegroup

    try:
        matched_tuples = align_tokens2(perf_midi, score_midi, perf_beats, score_beats, skip_Nones=True)

        if len(matched_tuples) < 20:
            return []

        score_annotations = load_annotation_file(score_beats)
        score_beat_times = [annotation[0] for annotation in score_annotations]

        normalized_matched_tuples = []
        for match in matched_tuples:
            perf_triplet = match[0]
            score_triplet = match[2]

            if score_triplet[0] is not None:
                original_time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
                original_duration_sec = (score_triplet[1] - DUR_OFFSET) / TIME_RESOLUTION
                original_end_time_sec = original_time_sec + original_duration_sec
                pitch = score_triplet[2]

                normalized_time_sec = _normalize_score_time_sec(original_time_sec, score_beat_times)
                normalized_end_time_sec = _normalize_score_time_sec(
                    original_end_time_sec,
                    score_beat_times,
                )
                normalized_duration_sec = max(0.0, normalized_end_time_sec - normalized_time_sec)

                normalized_time_units = round(normalized_time_sec * TIME_RESOLUTION)
                normalized_duration_units = round(normalized_duration_sec * TIME_RESOLUTION)
                normalized_time_units = max(0, normalized_time_units)
                normalized_duration_units = max(0, normalized_duration_units)
                normalized_score = [
                    normalized_time_units + TIME_OFFSET,
                    normalized_duration_units + DUR_OFFSET,
                    pitch,
                ]
            else:
                normalized_score = score_triplet

            normalized_matched_tuples.append([perf_triplet, match[1], normalized_score, match[3]])

        return _build_sequences(normalized_matched_tuples, prefix_controls)

    except Exception:
        return []


def _build_sequences(normalized_matched_tuples, prefix_controls=33):
    """
    Build fixed-length interleaved sequences from normalized matched tuples.
    """
    sequences = []
    prefix_count = min(prefix_controls, len(normalized_matched_tuples))

    for start_idx in range(len(normalized_matched_tuples)):
        interleaved_tokens = []

        subset = normalized_matched_tuples[start_idx:]

        if len(subset) < prefix_count:
            break

        perf_triplets = [
            [match[0][0] - ATIME_OFFSET, match[0][1] - ADUR_OFFSET, match[0][2] - ANOTE_OFFSET]
            for match in subset
        ]

        if perf_triplets:
            perf_min_time = min(triplet[0] for triplet in perf_triplets)
            perf_triplets = [
                [triplet[0] - perf_min_time, triplet[1], triplet[2]]
                for triplet in perf_triplets
            ]

        score_triplets = [match[2] for match in subset]
        score_times = [triplet[0] - TIME_OFFSET for triplet in score_triplets if triplet[0] is not None]
        score_min_time = min(score_times) if score_times else 0
        score_triplets = [
            [triplet[0] - score_min_time, triplet[1], triplet[2]] if triplet[0] is not None else triplet
            for triplet in score_triplets
        ]

        for i in range(prefix_count):
            perf_triplet = perf_triplets[i]

            interleaved_tokens.extend([
                perf_triplet[0] + ATIME_OFFSET,
                perf_triplet[1] + ADUR_OFFSET,
                perf_triplet[2] + ANOTE_OFFSET,
            ])

            cc_time = max(0, perf_triplet[0])
            interleaved_tokens.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])

        for i in range(len(subset)):
            score_triplet = score_triplets[i]

            if score_triplet[0] is not None:
                interleaved_tokens.extend(score_triplet)

            ii = i + prefix_count
            if ii < len(subset):
                perf_triplet = perf_triplets[ii]
                interleaved_tokens.extend([
                    perf_triplet[0] + ATIME_OFFSET,
                    perf_triplet[1] + ADUR_OFFSET,
                    perf_triplet[2] + ANOTE_OFFSET,
                ])

        max_body = PACKED_SEQUENCE_LENGTH
        if len(interleaved_tokens) < max_body:
            break

        interleaved_tokens = interleaved_tokens[:max_body]

        if ops.max_time(interleaved_tokens, seconds=False) >= MAX_TIME:
            continue

        sequence = interleaved_tokens

        assert len(sequence) == PACKED_SEQUENCE_LENGTH, (
            f"Expected {PACKED_SEQUENCE_LENGTH} tokens, got {len(sequence)}"
        )

        token_str = ' '.join(str(token) for token in sequence)
        sequences.append(f"{token_str} | ")

    return sequences


def process_single_piece(filegroup):
    """
    Worker function for multiprocessing.
    Returns: (list_of_sequences, num_sequences)
    """
    sequences = tokenize_sliding_windows_asap(filegroup)
    return sequences, len(sequences)


def _process_split(pool, pairs, output_path, desc):
    """
    Process one split using an existing worker pool.

    Returns: (total_sequences, pieces_success, pieces_failed)
    """
    sequences_total = 0
    pieces_success = 0
    pieces_failed = 0

    with open(output_path, 'w') as output_file:
        with tqdm(total=len(pairs), desc=desc, unit='piece') as progress:
            for sequences, count in pool.imap_unordered(process_single_piece, pairs):
                if count > 0:
                    for sequence in sequences:
                        output_file.write(sequence + '\n')
                    sequences_total += count
                    pieces_success += 1
                else:
                    pieces_failed += 1
                progress.update(1)

    return sequences_total, pieces_success, pieces_failed


def main():
    print("Processing ASAP metadata...")
    os.makedirs('data', exist_ok=True)

    asap_datafiles, asap_composition_keys, asap_piece_names = load_asap_data()
    if len(asap_datafiles) == 0:
        print("ERROR: No valid ASAP pieces found. Please check dataset paths.")
        sys.exit(1)

    print(f"\nTotal: {len(asap_datafiles)} ASAP pieces")

    train_pairs, test_pairs, train_piece_names, test_piece_names = split_by_composition(
        asap_datafiles,
        asap_composition_keys,
        asap_piece_names,
    )

    print(f"Train: {len(train_pairs)} pieces")
    print(f"Test: {len(test_pairs)} pieces")
    print()

    write_split_file(len(asap_datafiles), train_piece_names, test_piece_names)

    with Pool(processes=NUM_WORKERS, maxtasksperchild=POOL_MAX_TASKS_PER_CHILD) as pool:
        print("Processing training set...")
        train_sequences_total, train_pieces_success, train_pieces_failed = _process_split(
            pool,
            train_pairs,
            TRAIN_OUTPUT,
            'Train',
        )

        print(
            f"Train: {train_sequences_total} sequences from "
            f"{train_pieces_success} ASAP pieces, {train_pieces_failed} failed"
        )

        print("\nProcessing test set...")
        test_sequences_total, test_pieces_success, test_pieces_failed = _process_split(
            pool,
            test_pairs,
            TEST_OUTPUT,
            'Test',
        )

    print(
        f"Test: {test_sequences_total} sequences from "
        f"{test_pieces_success} ASAP pieces, {test_pieces_failed} failed"
    )

    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    if train_sequences_total > 0:
        with open(TRAIN_OUTPUT, 'r') as train_file:
            first_line = train_file.readline().strip()
            tokens_part = first_line.split('|')[0].strip()
            first_seq = [int(x) for x in tokens_part.split()]

        print(f"First training sequence length: {len(first_seq)} tokens")
        print("No mode/bootstrap tokens are serialized")

    print()
    print("Final summary:")
    print(f"  Training sequences: {train_sequences_total} from {train_pieces_success}/{len(train_pairs)} pieces")
    print(f"  Test sequences: {test_sequences_total} from {test_pieces_success}/{len(test_pairs)} pieces")
    print(f"  Output train file: {TRAIN_OUTPUT}")
    print(f"  Output test file: {TEST_OUTPUT}")
    print(f"  Split file: {SPLIT_FILE}")

    if train_sequences_total == 0 or test_sequences_total == 0:
        print()
        print("WARNING: ASAP-only tokenization produced an empty output split.")
        print("Training will fail if either the train or test token file has zero sequences.")
        print("Check the piece success/failure counts above to see whether pieces were too short or failed during processing.")


if __name__ == '__main__':
    main()
