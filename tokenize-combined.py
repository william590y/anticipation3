"""
Tokenize ASAP and ATEPP datasets using sliding window to extract all possible
interleaved sequences without ANTICIPATE or SEP header tokens.

This combines both datasets:
- ASAP: Uses beat annotations for precise alignment
- ATEPP: Uses direct note matching for alignment (for pieces with scores)

Score normalization ENFORCES 1.0 second beat spacing regardless of original tempo.
Performance/control times preserve original tempo but are shifted to start at 0.

Uses parallel processing with the available CPU workers.
"""

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import mido
from music21 import converter as m21_converter
import tempfile
import re
import unicodedata
import warnings
import json
from pathlib import Path
from functools import lru_cache

from anticipation.config import *
from anticipation.vocab import *
from anticipation import ops
from anticipation.convert import midi_to_events
from alignment import align_tokens2, load_annotation_file

# Suppress music21 warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Number of parallel workers
NUM_WORKERS = os.cpu_count()

# Dataset paths
ASAP_PATH = 'asap-dataset-master'
ASAP_META_CSV = os.path.join(ASAP_PATH, 'metadata.csv')

ATEPP_PATH = 'ATEPP'  # Base folder containing the dataset
ATEPP_DATA_PATH = os.path.join(ATEPP_PATH, 'ATEPP-1.2')  # Subfolder with actual MIDI/score files
ATEPP_META_CSV = os.path.join(ATEPP_PATH, 'ATEPP-metadata-1.2.csv')
# Output paths
TRAIN_OUTPUT = 'data/train_combined.txt'
TEST_OUTPUT = 'data/test_combined.txt'
SPLIT_FILE = 'data/combined_split.txt'
# Curriculum learning: separate files by source quality
TRAIN_ASAP_OUTPUT = 'data/train_asap.txt'
TRAIN_ATEPP_OUTPUT = 'data/train_atepp.txt'
TEST_ASAP_OUTPUT = 'data/test_asap.txt'
TEST_ASAP_MUSTER_CACHE = 'data/test_asap_muster_cache.jsonl'

PACKED_SEQUENCE_LENGTH = CONTEXT_SIZE - 4
TARGET_BEAT_INTERVAL = 1.0

print(f"Combined ASAP + ATEPP Tokenization")
print(f"=" * 60)
print(f"Configuration:")
print(f"  Workers: {NUM_WORKERS}")
print(f"  Context size: {CONTEXT_SIZE}")
print(f"  Serialized length: {PACKED_SEQUENCE_LENGTH}")
print(f"  Prefix controls: 33 (fixed)")
print(f"  Output format: space-separated tokens (one sequence per line)")
print()


def _make_temp_path(final_path):
    final_path = Path(final_path)
    final_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=final_path.parent,
        prefix=f"{final_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tmp:
        return Path(tmp.name)


def _replace_atomic(temp_path, final_path):
    os.replace(str(temp_path), str(final_path))


def _events_to_triplets(events):
    assert len(events) % 3 == 0
    triplets = []
    for i in range(0, len(events), 3):
        triplets.append([
            int(round(events[i])),
            int(round(events[i + 1])),
            int(round(events[i + 2])),
        ])
    return triplets


def _normalize_asap_score_time_and_duration(original_time_sec, original_duration_sec, score_beat_times):
    normalized_time_sec = 0.0
    time_scale_factor = 1.0

    if score_beat_times and len(score_beat_times) >= 2:
        if original_time_sec < score_beat_times[0]:
            beat_duration = score_beat_times[1] - score_beat_times[0]
            if beat_duration > 0:
                progress = (original_time_sec - score_beat_times[0]) / beat_duration
                time_scale_factor = TARGET_BEAT_INTERVAL / beat_duration
            else:
                progress = 0.0
                time_scale_factor = 1.0
            normalized_time_sec = progress * TARGET_BEAT_INTERVAL
        else:
            found = False
            for i in range(len(score_beat_times) - 1):
                if score_beat_times[i] <= original_time_sec <= score_beat_times[i + 1]:
                    beat_duration = score_beat_times[i + 1] - score_beat_times[i]
                    if beat_duration > 0:
                        progress = (original_time_sec - score_beat_times[i]) / beat_duration
                        time_scale_factor = TARGET_BEAT_INTERVAL / beat_duration
                    else:
                        progress = 0.0
                        time_scale_factor = 1.0
                    normalized_time_sec = (
                        i * TARGET_BEAT_INTERVAL + progress * TARGET_BEAT_INTERVAL
                    )
                    found = True
                    break

            if not found:
                last_beat_idx = len(score_beat_times) - 1
                if len(score_beat_times) >= 2:
                    last_beat_duration = score_beat_times[-1] - score_beat_times[-2]
                else:
                    last_beat_duration = 1.0

                if last_beat_duration > 0:
                    progress = (original_time_sec - score_beat_times[-1]) / last_beat_duration
                    time_scale_factor = TARGET_BEAT_INTERVAL / last_beat_duration
                else:
                    progress = 0.0
                    time_scale_factor = 1.0
                normalized_time_sec = (
                    last_beat_idx * TARGET_BEAT_INTERVAL + progress * TARGET_BEAT_INTERVAL
                )
    else:
        normalized_time_sec = original_time_sec - (score_beat_times[0] if score_beat_times else 0.0)

    normalized_duration_sec = original_duration_sec * time_scale_factor
    normalized_time_units = max(0, round(normalized_time_sec * TIME_RESOLUTION))
    normalized_duration_units = max(0, round(normalized_duration_sec * TIME_RESOLUTION))
    return normalized_time_units, normalized_duration_units


def _normalize_asap_score_triplet(score_triplet, score_beat_times):
    normalized_time_units, normalized_duration_units = _normalize_asap_score_time_and_duration(
        (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION,
        (score_triplet[1] - DUR_OFFSET) / TIME_RESOLUTION,
        score_beat_times,
    )
    return [
        normalized_time_units + TIME_OFFSET,
        normalized_duration_units + DUR_OFFSET,
        score_triplet[2],
    ]


def _score_triplet_from_seconds(time_sec, duration_sec, note_token):
    time_units = max(0, round(time_sec * TIME_RESOLUTION))
    dur_units = min(max(0, round(duration_sec * TIME_RESOLUTION)), MAX_DUR - 1)
    return [TIME_OFFSET + time_units, DUR_OFFSET + dur_units, note_token]


@lru_cache(maxsize=256)
def _load_musicxml_score_tuples_60bpm(score_musicxml):
    score = m21_converter.parse(score_musicxml)
    score_tuples = []

    for element in score.flatten().notes:
        onset_sec = float(element.offset)
        dur_sec = float(element.duration.quarterLength)

        if hasattr(element, 'pitches'):
            for pitch in element.pitches:
                score_tuples.append([onset_sec, dur_sec, int(pitch.midi)])
        else:
            score_tuples.append([onset_sec, dur_sec, int(element.pitch.midi)])

    score_tuples.sort(key=lambda tup: (tup[0], tup[2], tup[1]))
    return score_tuples


def build_asap_eval_cache_record(filegroup):
    _, perf_midi, score_midi, _perf_beats, score_beats = filegroup

    try:
        perf_triplets = _events_to_triplets(midi_to_events(perf_midi, quantize=False))
        raw_score_triplets = _events_to_triplets(midi_to_events(score_midi, quantize=False))
        score_annotations = load_annotation_file(score_beats)
        score_beat_times = [anno[0] for anno in score_annotations]
    except Exception:
        return None

    if len(perf_triplets) < 33 or len(raw_score_triplets) < 5:
        return None

    normalized_score_triplets = [
        _normalize_asap_score_triplet(score_triplet, score_beat_times)
        for score_triplet in raw_score_triplets
    ]
    perf_controls = [
        [
            ATIME_OFFSET + (triplet[0] - TIME_OFFSET),
            ADUR_OFFSET + (triplet[1] - DUR_OFFSET),
            ANOTE_OFFSET + (triplet[2] - NOTE_OFFSET),
        ]
        for triplet in perf_triplets
    ]

    return {
        "perf_path": os.path.relpath(perf_midi, ASAP_PATH).replace("\\", "/"),
        "perf_triplets": perf_controls,
        "raw_score_triplets": raw_score_triplets,
        "normalized_score_triplets": normalized_score_triplets,
        "score_beat_times": score_beat_times,
    }

# ============================================================================
# Composition Key Extraction
# Ensures that performances of the same work from ASAP and ATEPP always land
# on the same side of the train/test split, preventing cross-dataset leakage.
# ============================================================================

def _normalize_composer(name):
    """Map full or short composer name to a consistent lowercase last name."""
    _map = {
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
    return _map.get(name, name.split()[-1].lower())


# Genre/function words that are too generic to identify a specific composition.
# Works whose names start with one of these NEED a catalog number (op/BWV/K/…)
# to be unambiguous; works whose names start with a distinctive word (arabeske,
# mephisto, gaspard, …) are identified by that word instead.
_GENERIC_WORK_WORDS = {
    # genre terms
    'etude', 'etudes', 'prelude', 'preludes', 'nocturne', 'nocturnes',
    'fugue', 'fugues', 'toccata', 'toccatas',
    'impromptu', 'impromptus', 'moment', 'moments', 'morceaux',
    'waltz', 'waltzes', 'mazurka', 'mazurkas', 'polonaise', 'polonaises',
    'intermezzo', 'intermezzi', 'caprice', 'caprices',
    'variation', 'variations', 'theme',
    'sonata', 'sonatina', 'concerto', 'symphony', 'quartet', 'quintet',
    'suite', 'piece', 'pieces', 'collection', 'album',
    'book', 'volume', 'heft',
    # instrument / language qualifiers
    'piano', 'keyboard', 'clavier', 'klavier', 'wohltemperierte',
    # WTC-specific: prevents "tempered" being extracted as a distinctive word
    # (ATEPP's "The_Well-Tempered_Clavier,_Book_I" should fall through to BWV number)
    'tempered', 'welltemp',
    # size adjectives
    'grand', 'grande', 'petit', 'petite', 'little', 'great', 'kleine',
    # ordinals (written out)
    'first', 'second', 'third', 'fourth', 'fifth',
    # French/German preposition fragments that survive after stripping contractions
    'execution', 'exécution', 'pour', 'dans', 'avec', 'sans', 'sous',
    'vers', 'nach', 'über',
    # key names
    'major', 'minor', 'sharp', 'flat', 'natural',
}


def _first_significant_word(text):
    """Return the first 'distinctive' word (≥5 chars, not a generic genre/function
    word) from text, truncated to 9 characters so that French/English cognates
    match (transcendante → transcend == transcendental → transcend).

    Returns None if no such word is found (caller should fall back to catalog).
    """
    # strip French contractions d'/l'/s' so "d'exécution" → "execution"
    t = text.replace("d'", " ").replace("l'", " ").replace("s'", " ")
    t = text.replace("d\u2019", " ").replace("l\u2019", " ").replace("s\u2019", " ")
    # remove diacritics (NFD decomposition + strip combining chars)
    t = ''.join(c for c in unicodedata.normalize('NFD', t)
                if unicodedata.category(c) != 'Mn')
    t = t.lower()
    t = re.sub(r'[^a-z0-9 ]', ' ', t)
    for word in t.split():
        if word.isdigit():
            continue
        if len(word) < 5:
            continue
        if word in _GENERIC_WORK_WORDS:
            continue
        if re.match(r'^book\d', word):   # volume labels: 'book2', 'book1', etc.
            continue
        return word[:9]   # truncate for cross-language matching
    return None


def _extract_catalog(text):
    """Extract the most significant catalog number from text.
    Groups all pieces within the same opus/BWV together (no sub-piece number)
    so that ASAP 'op_32_10' and ATEPP 'Op. 32 No. 10' resolve to the same key.
    """
    t = text.lower().replace('_', ' ').replace('-', ' ')
    # BWV (Bach) — use LAST match to skip range prefixes like "BWV_870-893" in
    # ATEPP WTC folder names; the individual piece BWV appears at the end.
    bwv_matches = re.findall(r'\bbwv\s*(\d+)', t)
    if bwv_matches:
        return f"bwv{bwv_matches[-1]}"
    # Sonata number — checked BEFORE opus so that ATEPP "Piano Sonata No. 12, Op. 26"
    # extracts son12, matching ASAP's "Piano_Sonatas_12" numbering scheme
    m = re.search(r'\b(?:piano|keyboard)\s*sonatas?\s*(\d+)', t)
    if m:
        return f"son{m.group(1)}"
    m = re.search(r'\bsonata\s*no[\s.]*(\d+)', t)
    if m:
        return f"son{m.group(1)}"
    # ASAP-style: "Sonata_2_1st" or "Sonata_3_4th" — number immediately after 'sonata'
    m = re.search(r'\bsonata[\s_]*(\d+)', t)
    if m:
        return f"son{m.group(1)}"
    # Hob. XVI (Haydn) — map to son{N} so ATEPP "Hob.XVI:48" == ASAP "Keyboard_Sonatas_48"
    m = re.search(r'\bhob[.\s]*xvi[.\s:]*(\d+)', t)
    if m:
        return f"son{m.group(1)}"
    # Opus number — intentionally omit piece-within-opus so ASAP and ATEPP unify
    m = re.search(r'\bop[\s.]*(\d+)\b', t)
    if m:
        return f"op{m.group(1)}"
    # K. / Kv. (Mozart, Scarlatti)
    m = re.search(r'\bk[v]?[\s.]*(\d+)\b', t)
    if m:
        return f"k{m.group(1)}"
    # D. NNN (Schubert) — require 3+ digits to avoid false positives
    m = re.search(r'\bd[\s.]+(\d{3,})\b', t)
    if m:
        return f"d{m.group(1)}"
    # S. NNN (Liszt Searle catalogue) — require 3+ digits, must be standalone 's'
    m = re.search(r'\bs[\s.]*(\d{3,})\b', t)
    if m:
        return f"s{m.group(1)}"
    return None


# Explicit key overrides for ASAP works whose title/path contain insufficient
# catalog information to match the corresponding ATEPP entry automatically.
# Format: (composer_as_in_csv, title_as_in_csv) -> canonical_key
_ASAP_WORK_OVERRIDES = {
    ('Liszt', 'Sonata'): 'liszt__s178',  # Piano Sonata in B minor, S.178
}


def composition_key_asap(composer, title, midi_score):
    """Canonical composition key for an ASAP entry."""
    # 0. Explicit override for works with insufficient metadata
    override = _ASAP_WORK_OVERRIDES.get((composer, title))
    if override:
        return override
    comp = _normalize_composer(composer)
    # 1. Try catalog patterns on title (handles op/BWV/K/D/sonata-number)
    cat = _extract_catalog(title)
    if cat:
        return f"{comp}__{cat}"
    # 2. Try catalog on the work folder (parts[1] = work folder, not movement subfolder)
    parts = midi_score.replace('\\', '/').split('/')
    work_folder = parts[1] if len(parts) > 1 else parts[-1]
    cat = _extract_catalog(work_folder)
    if cat:
        return f"{comp}__{cat}"
    # 3. Fall back to first distinctive word of the title
    word = _first_significant_word(title)
    if word:
        return f"{comp}__{word}"
    # 4. Final fallback: composer/work_folder so all movements of the same work
    #    stay together even when no catalog number or distinctive word is found
    composer_folder = parts[0] if parts else 'unknown'
    return f"asap__{composer_folder}/{work_folder}"


def composition_key_atepp(composer, score_path):
    """Canonical composition key for an ATEPP entry.

    Priority: distinctive work-name word > catalog number > raw path.
    Trying the distinctive word FIRST ensures that titled pieces like
    "Arabeske, Op. 18" or "Mephisto Waltz, S. 514" produce the same key as
    the ASAP entries which only carry the work title without the catalog suffix.
    Generic pieces (Preludes Op.23, Etudes S.139, …) have no distinctive word
    so they fall through to the catalog number, which ASAP also extracts.
    """
    comp = _normalize_composer(composer)
    parts = score_path.replace('\\', '/').split('/')
    work_folder = parts[1] if len(parts) > 1 else score_path

    # 1. First significant (non-generic) word from the work-level folder
    word = _first_significant_word(work_folder)
    if word:
        return f"{comp}__{word}"

    # 2. Catalog number from full path (handles generic collections: Preludes Op.23, etc.)
    cat = _extract_catalog(score_path)
    if cat:
        return f"{comp}__{cat}"

    # 3. Catalog from folder alone
    cat = _extract_catalog(work_folder)
    if cat:
        return f"{comp}__{cat}"

    return f"atepp__{score_path}"


# ============================================================================
# ASAP Dataset Loading
# ============================================================================

asap_datafiles = []
asap_score_paths = []
asap_composition_keys = []
asap_piece_names = []

if os.path.exists(ASAP_META_CSV):
    df_asap = pd.read_csv(ASAP_META_CSV)
    print(f"[ASAP] Found {len(df_asap)} pieces in metadata")
    
    for _, row in df_asap.iterrows():
        perf_midi = os.path.join(ASAP_PATH, row['midi_performance'])
        score_midi = os.path.join(ASAP_PATH, row['midi_score'])
        perf_beats = os.path.join(ASAP_PATH, row['performance_annotations'])
        score_beats = os.path.join(ASAP_PATH, row['midi_score_annotations'])
        
        if all(os.path.exists(f) for f in [perf_midi, score_midi, perf_beats, score_beats]):
            asap_datafiles.append(('asap', perf_midi, score_midi, perf_beats, score_beats))
            asap_score_paths.append(score_midi)
            asap_composition_keys.append(composition_key_asap(row['composer'], row['title'], row['midi_score']))
            asap_piece_names.append(row['midi_performance'])
    
    print(f"[ASAP] Found {len(asap_datafiles)} valid pieces with all required files")
else:
    print(f"[ASAP] Metadata not found at {ASAP_META_CSV}, skipping ASAP dataset")

# ============================================================================
# ATEPP Dataset Loading
# ============================================================================

atepp_datafiles = []
atepp_score_paths = []
atepp_composition_keys = []
atepp_piece_names = []

def musicxml_to_midi_path(musicxml_path):
    """Convert MusicXML to a temporary MIDI file and return the path."""
    try:
        score = m21_converter.parse(musicxml_path)
        # Create temp file
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            temp_path = f.name
        score.write('midi', fp=temp_path)
        return temp_path
    except Exception as e:
        return None


if os.path.exists(ATEPP_META_CSV):
    df_atepp = pd.read_csv(ATEPP_META_CSV)
    print(f"[ATEPP] Found {len(df_atepp)} pieces in metadata")
    
    # Filter for pieces with scores and good quality
    valid_atepp = 0
    for _, row in df_atepp.iterrows():
        # Skip low quality and corrupted pieces
        quality = str(row.get('quality', '')).lower()
        if 'low quality' in quality or 'corrupted' in quality:
            continue
        
        # Check if score_path exists and is not empty
        score_path = row.get('score_path', '')
        if pd.isna(score_path) or score_path == '':
            continue
        
        midi_path = row.get('midi_path', '')
        if pd.isna(midi_path) or midi_path == '':
            continue
        
        perf_midi = os.path.join(ATEPP_DATA_PATH, midi_path)
        score_file = os.path.join(ATEPP_DATA_PATH, score_path)
        
        if os.path.exists(perf_midi) and os.path.exists(score_file):
            # ATEPP entries: ('atepp', perf_midi, score_musicxml_path, composition_id)
            atepp_datafiles.append(('atepp', perf_midi, score_file, row.get('composition_id', 0)))
            atepp_score_paths.append(score_file)
            atepp_composition_keys.append(composition_key_atepp(str(row.get('composer', '')), score_path))
            atepp_piece_names.append(midi_path)
            valid_atepp += 1
    
    print(f"[ATEPP] Found {valid_atepp} valid pieces with scores (filtered for quality)")
else:
    print(f"[ATEPP] Metadata not found at {ATEPP_META_CSV}, skipping ATEPP dataset")

# ============================================================================
# Combine and Split
# ============================================================================

all_datafiles = asap_datafiles + atepp_datafiles
all_score_paths = asap_score_paths + atepp_score_paths
all_composition_keys = asap_composition_keys + atepp_composition_keys
all_piece_names = asap_piece_names + atepp_piece_names

print(f"\nTotal: {len(all_datafiles)} pieces ({len(asap_datafiles)} ASAP + {len(atepp_datafiles)} ATEPP)")

if len(all_datafiles) == 0:
    print("ERROR: No valid pieces found. Please check dataset paths.")
    exit(1)

# Split by canonical composition key (not raw score file path) so that all
# ASAP and ATEPP performances of the same musical work always land on the
# same side of the split, preventing cross-dataset leakage.
rng = np.random.default_rng(42)
unique_comp_keys = list(sorted(set(all_composition_keys)))
rng.shuffle(unique_comp_keys)
n_test = int(np.ceil(0.2 * len(unique_comp_keys)))
test_comp_keys = set(unique_comp_keys[:n_test])

train_pairs = []
test_pairs = []
train_piece_names = []
test_piece_names = []

for df_entry, comp_key, piece_name in zip(all_datafiles, all_composition_keys, all_piece_names):
    if comp_key in test_comp_keys:
        test_pairs.append(df_entry)
        test_piece_names.append(piece_name)
    else:
        train_pairs.append(df_entry)
        train_piece_names.append(piece_name)

print(f"Train: {len(train_pairs)} pieces")
print(f"Test: {len(test_pairs)} pieces")
print()

# Write split information
print(f"Writing split information to {SPLIT_FILE}...")
split_tmp = _make_temp_path(SPLIT_FILE)
with open(split_tmp, 'w') as f:
    f.write(f"# Total pieces: {len(all_datafiles)} (train: {len(train_pairs)}, test: {len(test_pairs)})\n")
    f.write(f"# ASAP pieces: {len(asap_datafiles)}\n")
    f.write(f"# ATEPP pieces: {len(atepp_datafiles)}\n\n")
    
    f.write(f"=== TRAINING PIECES ===\n")
    for piece_name in sorted(train_piece_names):
        f.write(f"./{piece_name}\n")
    
    f.write(f"\n=== TEST PIECES ===\n")
    for piece_name in sorted(test_piece_names):
        f.write(f"./{piece_name}\n")

print(f"Split file staged at temporary path for {SPLIT_FILE}\n")


# ============================================================================
# Alignment Functions
# ============================================================================

def align_atepp_tokens(perf_midi, score_musicxml, thres=0.1):
    """
    Align ATEPP performance MIDI with MusicXML score.
    
    Since ATEPP doesn't have beat annotations, we use direct note matching
    based on pitch and temporal proximity.
    
    Returns: List of [perf_triplet, perf_idx, score_triplet, score_idx]
    """
    try:
        perf = midi_to_events(perf_midi, quantize=False)
        score_tuples = _load_musicxml_score_tuples_60bpm(score_musicxml)
        
        # Create tuples: [time_sec, duration, pitch]
        p_tuples = [[perf[3*i]/TIME_RESOLUTION, perf[3*i+1] - DUR_OFFSET, perf[3*i+2] - NOTE_OFFSET] 
                    for i in range(int(len(perf)/3))]
        s_tuples = score_tuples
        
        if len(p_tuples) == 0 or len(s_tuples) == 0:
            return []
        
        # Estimate tempo ratio based on total durations
        p_duration = max(t[0] for t in p_tuples) if p_tuples else 1
        s_duration = max(t[0] for t in s_tuples) if s_tuples else 1
        tempo_ratio = p_duration / s_duration if s_duration > 0 else 1.0
        
        matched_tuples = []
        s_tuples_copy = s_tuples.copy()
        
        for i, p_tuple in enumerate(p_tuples):
            best_dist = np.inf
            best_match = [None, None, None]
            best_index = None
            
            p_time, p_note = p_tuple[0], p_tuple[2]
            
            for j, s_tuple in enumerate(s_tuples_copy):
                s_time, s_note = s_tuple[0], s_tuple[2]
                
                # Skip if pitch doesn't match
                if p_note != s_note:
                    continue
                
                # Compute distance using tempo-scaled score time
                scaled_s_time = s_time * tempo_ratio
                dist = abs(p_time - scaled_s_time)
                
                # Allow wider window based on tempo ratio
                adaptive_thres = thres * max(1.0, tempo_ratio)
                
                if dist <= adaptive_thres and dist < best_dist:
                    best_dist = dist
                    best_match = s_tuple
                    best_index = s_tuples.index(s_tuple)
            
            if best_index is not None:
                # Convert to token format
                perf_triplet = [
                    round(p_tuple[0] * TIME_RESOLUTION) + ATIME_OFFSET,
                    p_tuple[1] + ADUR_OFFSET,
                    p_tuple[2] + ANOTE_OFFSET
                ]
                score_triplet = _score_triplet_from_seconds(
                    best_match[0],
                    best_match[1],
                    best_match[2] + NOTE_OFFSET,
                )
                matched_tuples.append([perf_triplet, i, score_triplet, best_index])
                s_tuples_copy.remove(best_match)
            else:
                perf_triplet = [
                    round(p_tuple[0] * TIME_RESOLUTION) + ATIME_OFFSET,
                    p_tuple[1] + ADUR_OFFSET,
                    p_tuple[2] + ANOTE_OFFSET
                ]
                projected_score_time = p_time / tempo_ratio if tempo_ratio > 0 else p_time
                score_triplet = _score_triplet_from_seconds(
                    projected_score_time,
                    0.0,
                    REST,
                )
                matched_tuples.append([perf_triplet, i, score_triplet, None])
        
        return matched_tuples
        
    except Exception as e:
        return []


def tokenize_sliding_windows_asap(filegroup, prefix_controls=33):
    """
    Tokenize an ASAP performance-score pair using beat annotations.
    """
    _, perf_midi, score_midi, perf_beats, score_beats = filegroup
    
    try:
        # Align using beat annotations
        matched_tuples = align_tokens2(
            perf_midi,
            score_midi,
            perf_beats,
            score_beats,
            skip_Nones=False,
            preserve_unmatched_perf=True,
        )
        
        if len(matched_tuples) < 20:
            return []

        # Pre-normalize score triplets into quarter-based 60 BPM space
        normalized_matched_tuples = []
        for match in matched_tuples:
            perf_triplet = match[0]
            score_triplet = match[2]
            
            if score_triplet[0] is not None:
                normalized_score = _normalize_asap_score_triplet(score_triplet, score_beat_times)
            else:
                normalized_score = score_triplet
            
            normalized_matched_tuples.append([perf_triplet, match[1], normalized_score, match[3]])
        
        return _build_sequences(normalized_matched_tuples, prefix_controls)
        
    except Exception as e:
        return []


def tokenize_sliding_windows_atepp(filegroup, prefix_controls=33):
    """
    Tokenize an ATEPP performance-score pair using direct note matching.
    """
    _, perf_midi, score_musicxml, composition_id = filegroup
    
    try:
        # Align using direct note matching
        matched_tuples = align_atepp_tokens(perf_midi, score_musicxml, thres=0.5)
        
        if len(matched_tuples) < 20:
            return []
        
        return _build_sequences(matched_tuples, prefix_controls)
        
    except Exception as e:
        return []


def _build_sequences(normalized_matched_tuples, prefix_controls=33):
    """
    Build fixed-length interleaved sequences from normalized matched tuples.
    Shared by both ASAP and ATEPP tokenizers.
    """
    sequences = []
    k = min(prefix_controls, len(normalized_matched_tuples))
    
    for start_idx in range(len(normalized_matched_tuples)):
        interleaved_tokens = []
        
        subset = normalized_matched_tuples[start_idx:]
        
        if len(subset) < k:
            break
        
        # Extract performance triplets (remove offsets first)
        perf_triplets = [[match[0][0] - ATIME_OFFSET, match[0][1] - ADUR_OFFSET, match[0][2] - ANOTE_OFFSET] for match in subset]
        
        # Normalize performance to start at time 0
        if perf_triplets:
            perf_min_time = min(triplet[0] for triplet in perf_triplets)
            perf_triplets = [
                [triplet[0] - perf_min_time, triplet[1], triplet[2]]
                for triplet in perf_triplets
            ]
        
        # Extract score triplets and normalize to start at time 0
        score_triplets = [match[2] for match in subset]
        score_times = [triplet[0] - TIME_OFFSET for triplet in score_triplets if triplet[0] is not None]
        score_min_time = min(score_times) if score_times else 0
        score_triplets = [
            [triplet[0] - score_min_time, triplet[1], triplet[2]] if triplet[0] is not None else triplet
            for triplet in score_triplets
        ]
        
        # Prefix: control + rest pairs
        for i in range(k):
            perf_triplet = perf_triplets[i]
            
            interleaved_tokens.extend([
                perf_triplet[0] + ATIME_OFFSET,
                perf_triplet[1] + ADUR_OFFSET,
                perf_triplet[2] + ANOTE_OFFSET
            ])
            
            cc_time = max(0, perf_triplet[0])
            interleaved_tokens.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])
        
        # Main body: alternate score/control
        for i in range(len(subset)):
            score_triplet = score_triplets[i]
            
            if score_triplet[0] is not None:
                interleaved_tokens.extend(score_triplet)
            
            ii = i + k
            if ii < len(subset):
                perf_triplet = perf_triplets[ii]
                interleaved_tokens.extend([
                    perf_triplet[0] + ATIME_OFFSET,
                    perf_triplet[1] + ADUR_OFFSET,
                    perf_triplet[2] + ANOTE_OFFSET
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
        
        token_str = ' '.join(str(tok) for tok in sequence)
        sequences.append(f"{token_str} | ")
    
    return sequences


def tokenize_sliding_windows(filegroup, prefix_controls=33):
    """
    Dispatch to appropriate tokenizer based on dataset type.
    """
    dataset_type = filegroup[0]
    
    if dataset_type == 'asap':
        return tokenize_sliding_windows_asap(filegroup, prefix_controls)
    elif dataset_type == 'atepp':
        return tokenize_sliding_windows_atepp(filegroup, prefix_controls)
    else:
        return []


def process_single_piece(filegroup):
    """
    Worker function for multiprocessing.
    Returns: (list_of_sequences, num_sequences, dataset_type)
    """
    sequences = tokenize_sliding_windows(filegroup)
    return (sequences, len(sequences), filegroup[0])


# ============================================================================
# Main Processing
# ============================================================================

if __name__ == '__main__':
    print("Processing training set...")
    os.makedirs('data', exist_ok=True)

    train_output_tmp = _make_temp_path(TRAIN_OUTPUT)
    train_asap_tmp = _make_temp_path(TRAIN_ASAP_OUTPUT)
    train_atepp_tmp = _make_temp_path(TRAIN_ATEPP_OUTPUT)
    test_output_tmp = _make_temp_path(TEST_OUTPUT)
    test_asap_tmp = _make_temp_path(TEST_ASAP_OUTPUT)
    test_asap_cache_tmp = _make_temp_path(TEST_ASAP_MUSTER_CACHE)
    
    train_sequences_total = 0
    train_pieces_success = 0
    train_pieces_failed = 0
    train_asap_success = 0
    train_atepp_success = 0
    
    with open(train_output_tmp, 'w') as f_train, \
         open(train_asap_tmp, 'w') as f_asap, \
         open(train_atepp_tmp, 'w') as f_atepp:
        with Pool(processes=NUM_WORKERS) as pool:
            with tqdm(total=len(train_pairs), desc='Train', unit='piece') as pbar:
                for sequences, count, dataset_type in pool.imap_unordered(process_single_piece, train_pairs):
                    if count > 0:
                        for seq in sequences:
                            f_train.write(seq + '\n')
                            if dataset_type == 'asap':
                                f_asap.write(seq + '\n')
                            else:
                                f_atepp.write(seq + '\n')
                        train_sequences_total += count
                        train_pieces_success += 1
                        if dataset_type == 'asap':
                            train_asap_success += 1
                        else:
                            train_atepp_success += 1
                    else:
                        train_pieces_failed += 1
                    pbar.update(1)

    print(f"Train: {train_sequences_total} sequences from {train_pieces_success} pieces ({train_asap_success} ASAP, {train_atepp_success} ATEPP), {train_pieces_failed} failed")
    
    # Process test set
    print("\nProcessing test set...")
    
    test_sequences_total = 0
    test_pieces_success = 0
    test_pieces_failed = 0
    test_asap_success = 0
    test_atepp_success = 0
    
    with open(test_output_tmp, 'w') as f_test, open(test_asap_tmp, 'w') as f_test_asap:
        with Pool(processes=NUM_WORKERS) as pool:
            with tqdm(total=len(test_pairs), desc='Test', unit='piece') as pbar:
                for sequences, count, dataset_type in pool.imap_unordered(process_single_piece, test_pairs):
                    if count > 0:
                        for seq in sequences:
                            f_test.write(seq + '\n')
                            if dataset_type == 'asap':
                                f_test_asap.write(seq + '\n')
                        test_sequences_total += count
                        test_pieces_success += 1
                        if dataset_type == 'asap':
                            test_asap_success += 1
                        else:
                            test_atepp_success += 1
                    else:
                        test_pieces_failed += 1
                    pbar.update(1)

    print("\nBuilding ASAP test MUSTER cache...")
    test_asap_filegroups = sorted(
        [pair for pair in test_pairs if pair[0] == 'asap'],
        key=lambda fg: os.path.relpath(fg[1], ASAP_PATH).replace('\\', '/'),
    )
    cache_records = []
    if test_asap_filegroups:
        with Pool(processes=NUM_WORKERS) as pool:
            with tqdm(total=len(test_asap_filegroups), desc='ASAP cache', unit='piece') as pbar:
                for record in pool.imap(build_asap_eval_cache_record, test_asap_filegroups):
                    if record is not None:
                        cache_records.append(record)
                    pbar.update(1)
    cache_records = sorted(cache_records, key=lambda record: record['perf_path'])
    with open(test_asap_cache_tmp, 'w', encoding='utf-8') as f_cache:
        for record in cache_records:
            f_cache.write(json.dumps(record) + '\n')

    _replace_atomic(split_tmp, SPLIT_FILE)
    _replace_atomic(train_output_tmp, TRAIN_OUTPUT)
    _replace_atomic(train_asap_tmp, TRAIN_ASAP_OUTPUT)
    _replace_atomic(train_atepp_tmp, TRAIN_ATEPP_OUTPUT)
    _replace_atomic(test_output_tmp, TEST_OUTPUT)
    _replace_atomic(test_asap_tmp, TEST_ASAP_OUTPUT)
    _replace_atomic(test_asap_cache_tmp, TEST_ASAP_MUSTER_CACHE)
    
    print(f"Test: {test_sequences_total} sequences from {test_pieces_success} pieces ({test_asap_success} ASAP, {test_atepp_success} ATEPP), {test_pieces_failed} failed")
    print(f"ASAP test MUSTER cache: {len(cache_records)} pieces")
    
    # Verification
    print("\n" + "="*80)
    print("VERIFICATION")
    print("="*80)
    
    if train_sequences_total > 0:
        with open(TRAIN_OUTPUT, 'r') as f:
            first_line = f.readline().strip()
            tokens_part = first_line.split('|')[0].strip()
            first_seq = [int(x) for x in tokens_part.split()]
        
        print(f"First training sequence length: {len(first_seq)} tokens")
        print("No mode/bootstrap tokens are serialized")
        
        control_count = 0
        score_count = 0
        rest_count = 0
        
        for i in range(min(100, len(first_seq) // 3)):
            pos = i * 3
            if pos + 2 >= len(first_seq):
                break
            
            t0 = first_seq[pos]
            t2 = first_seq[pos + 2]
            
            if t0 >= CONTROL_OFFSET:
                control_count += 1
            elif t2 == REST:
                rest_count += 1
            elif t2 >= NOTE_OFFSET:
                score_count += 1
        
        print(f"\nFirst 100 triplets breakdown:")
        print(f"  Control triplets: {control_count}")
        print(f"  Score triplets (notes): {score_count}")
        print(f"  Score triplets (REST): {rest_count}")
    else:
        print("No sequences generated!")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Training sequences: {train_sequences_total} from {train_pieces_success}/{len(train_pairs)} pieces")
    print(f"  ASAP: {train_asap_success} pieces")
    print(f"  ATEPP: {train_atepp_success} pieces")
    if train_pieces_success > 0:
        print(f"  Average sequences per piece: {train_sequences_total/train_pieces_success:.1f}")
    print(f"Test sequences: {test_sequences_total} from {test_pieces_success}/{len(test_pairs)} pieces")
    print(f"  ASAP: {test_asap_success} pieces")
    print(f"  ATEPP: {test_atepp_success} pieces")
    if test_pieces_success > 0:
        print(f"  Average sequences per piece: {test_sequences_total/test_pieces_success:.1f}")
    print(f"Total sequences: {train_sequences_total + test_sequences_total}")
    print(f"\nOutput files:")
    print(f"  {TRAIN_OUTPUT}")
    print(f"  {TRAIN_ASAP_OUTPUT}")
    print(f"  {TRAIN_ATEPP_OUTPUT}")
    print(f"  {TEST_OUTPUT}")
    print(f"  {TEST_ASAP_OUTPUT}")
    print(f"  {TEST_ASAP_MUSTER_CACHE}")
    print(f"  {SPLIT_FILE}")
    print("\nTokenization complete!")
