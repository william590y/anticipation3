"""
Tokenize ASAP and ATEPP datasets using sliding window to extract all possible 1024-token sequences.

This combines both datasets:
- ASAP: Uses beat annotations for precise alignment
- ATEPP: Uses DTW (Dynamic Time Warping) alignment algorithm for score-performance alignment

Score normalization ENFORCES 0.5 second beat spacing regardless of original tempo.
Performance/control times preserve original tempo but are shifted to start at 0.

Uses parallel processing with 128 workers for efficiency.
"""

import os
import pandas as pd
from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import mido
from music21 import converter as m21_converter
import tempfile
import warnings

from anticipation.config import *
from anticipation.vocab import *
from anticipation import ops
from anticipation.convert import midi_to_events
from alignment import align_tokens2, load_annotation_file

# Suppress music21 warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Try to import Cython-optimized DTW, fall back to pure Python
try:
    from dtw_alignment import dtw_align as cython_dtw_align, dtw_traceback as cython_dtw_traceback
    USE_CYTHON_DTW = True
    print("Using Cython-optimized DTW alignment")
except ImportError:
    USE_CYTHON_DTW = False
    print("Cython DTW not available, using pure Python (slower)")
    print("To compile Cython module, run: python setup_dtw.py build_ext --inplace")

# Number of parallel workers
NUM_WORKERS = 128

# Dataset paths
ASAP_PATH = 'asap-dataset-master'
ASAP_META_CSV = os.path.join(ASAP_PATH, 'metadata.csv')

ATEPP_PATH = 'ATEPP'  # Base folder containing the dataset
ATEPP_DATA_PATH = os.path.join(ATEPP_PATH, 'ATEPP-1.2')  # Subfolder with actual MIDI/score files
ATEPP_META_CSV = os.path.join(ATEPP_PATH, 'ATEPP-metadata-1.2.csv')

# Output paths
TRAIN_OUTPUT = 'data/train_combined_dtw.txt'
TEST_OUTPUT = 'data/test_combined_dtw.txt'
SPLIT_FILE = 'data/combined_split_dtw.txt'

print(f"Combined ASAP + ATEPP Tokenization (DTW for ATEPP)")
print(f"=" * 60)
print(f"Configuration:")
print(f"  Workers: {NUM_WORKERS}")
print(f"  Context size: {CONTEXT_SIZE}")
print(f"  Prefix controls: 33 (fixed)")
print(f"  Output format: space-separated tokens (one sequence per line)")
print()

# ============================================================================
# DTW Alignment Algorithm (Pure Python version of Cython code)
# ============================================================================

def dtw_align(score, perf, ds, lmbda):
    """
    DTW alignment algorithm - pure Python implementation.
    
    Args:
        score: np.array of shape (N, 129) where [:, :128] are note features and [:, 128] is timing
        perf: np.array of shape (M, 128) note features
        ds: time step size
        lmbda: regularization parameter for tempo deviation
    
    Returns:
        L: np.array of shape (N, M) cost matrix
    """
    score = np.asarray(score, dtype=np.float32)
    perf = np.asarray(perf, dtype=np.float32)
    
    score_timing = score[:, 128]
    prior = (ds * len(perf)) / np.cumsum(score_timing)[-1]  # slope = rise/run
    
    L = np.full((len(score), len(perf)), np.inf, dtype=np.float32)
    
    # Precompute local cost of aligning score[j] with perf[k]
    local_cost = np.empty((len(score), len(perf)), dtype=np.float32)
    for j in range(len(score)):
        for k in range(len(perf)):
            local_cost[j, k] = np.sum(np.abs(score[j, :128] - perf[k, :128]))
    
    # Base case j = 0
    sj = score_timing[0]
    incremental_cost = np.float32(0)
    for k in range(len(perf)):
        instantaneous_tempo = (k * ds) / sj
        tmp = instantaneous_tempo - prior
        R = lmbda * tmp * tmp
        incremental_cost += local_cost[0, k]
        L[0, k] = incremental_cost * ds + R
    
    L[0, 0] = 0  # base case
    
    # Fill DP table
    for j in range(1, len(score)):
        sj = score_timing[j]
        for k in range(len(perf)):
            incremental_cost = np.float32(0)
            
            for m in reversed(range(0, k + 1)):
                instantaneous_tempo = ((k - m) * ds) / sj
                tmp = instantaneous_tempo - prior
                R = lmbda * tmp * tmp
                
                cost = L[j - 1, m] + incremental_cost * ds + R
                
                if cost < L[j, k]:
                    L[j, k] = cost
                
                incremental_cost += local_cost[j, m]
    
    return L


def dtw_traceback(score, perf, L, ds, lmbda):
    """
    Traceback through DTW cost matrix to find optimal alignment.
    
    Args:
        score: np.array of shape (N, 129)
        perf: np.array of shape (M, 128)
        L: cost matrix from dtw_align
        ds: time step size
        lmbda: regularization parameter
    
    Returns:
        A: list of (score_idx, perf_idx) alignment pairs
        C: list of costs
    """
    ds = np.float32(ds)
    lmbda = np.float32(lmbda)
    
    score = np.asarray(score, dtype=np.float32)
    perf = np.asarray(perf, dtype=np.float32)
    
    score_timing = score[:, 128]
    prior = (ds * np.float32(len(perf))) / np.cumsum(score_timing)[-1]
    
    A, C = [], []
    k = len(perf) - 1
    
    for j in reversed(range(1, len(score))):
        sj = score_timing[j]
        incremental_cost = np.float32(0)
        
        for m in reversed(range(0, k + 1)):
            instantaneous_tempo = ((np.float32(k) - np.float32(m)) * ds) / sj
            
            tmp = instantaneous_tempo - prior
            R = lmbda * tmp * tmp
            
            if np.isclose(L[j, k], L[j - 1, m] + incremental_cost * ds + R, rtol=1e-5, atol=1e-5):
                A.append((j, k))
                C.append(L[j, k])
                k = m
                break
            
            for i in range(128):
                tmp = score[j, i] - perf[m, i]
                incremental_cost += abs(tmp)
        else:
            # No match found, this shouldn't happen in a valid DTW
            pass
    
    A.append((0, k))
    C.append(L[0, k])
    
    return list(reversed(A)), list(reversed(C))


# ============================================================================
# ASAP Dataset Loading
# ============================================================================

asap_datafiles = []
asap_score_paths = []
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
            asap_piece_names.append(row['midi_performance'])
    
    print(f"[ASAP] Found {len(asap_datafiles)} valid pieces with all required files")
else:
    print(f"[ASAP] Metadata not found at {ASAP_META_CSV}, skipping ASAP dataset")

# ============================================================================
# ATEPP Dataset Loading
# ============================================================================

atepp_datafiles = []
atepp_score_paths = []
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
all_piece_names = asap_piece_names + atepp_piece_names

print(f"\nTotal: {len(all_datafiles)} pieces ({len(asap_datafiles)} ASAP + {len(atepp_datafiles)} ATEPP)")

if len(all_datafiles) == 0:
    print("ERROR: No valid pieces found. Please check dataset paths.")
    exit(1)

# Split by unique score to avoid data leakage
rng = np.random.default_rng(42)
unique_scores = list(sorted(set(all_score_paths)))
rng.shuffle(unique_scores)
n_test = int(np.ceil(0.2 * len(unique_scores)))
test_scores = set(unique_scores[:n_test])
train_scores = set(unique_scores[n_test:])

train_pairs = []
test_pairs = []
train_piece_names = []
test_piece_names = []

for df_entry, score, piece_name in zip(all_datafiles, all_score_paths, all_piece_names):
    if score in test_scores:
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
with open(SPLIT_FILE, 'w') as f:
    f.write(f"# Total pieces: {len(all_datafiles)} (train: {len(train_pairs)}, test: {len(test_pairs)})\n")
    f.write(f"# ASAP pieces: {len(asap_datafiles)}\n")
    f.write(f"# ATEPP pieces: {len(atepp_datafiles)}\n\n")
    
    f.write(f"=== TRAINING PIECES ===\n")
    for piece_name in sorted(train_piece_names):
        f.write(f"./{piece_name}\n")
    
    f.write(f"\n=== TEST PIECES ===\n")
    for piece_name in sorted(test_piece_names):
        f.write(f"./{piece_name}\n")

print(f"Split file written: {SPLIT_FILE}\n")


# ============================================================================
# Alignment Functions
# ============================================================================

def midi_to_piano_roll(midi_path, time_resolution=100):
    """
    Convert MIDI file to piano roll representation for DTW alignment.
    
    Returns:
        features: np.array of shape (T, 129) where T is number of time frames
                  [:, :128] are note activations, [:, 128] is timing info
        note_events: list of (time_sec, duration, pitch, frame_idx) for each note
    """
    try:
        events = midi_to_events(midi_path, quantize=False)
        
        if len(events) == 0:
            return None, []
        
        # Parse events into note list
        notes = []
        for i in range(len(events) // 3):
            time_units = events[3 * i]
            dur = events[3 * i + 1] - DUR_OFFSET
            pitch = events[3 * i + 2] - NOTE_OFFSET
            time_sec = time_units / TIME_RESOLUTION
            dur_sec = dur / TIME_RESOLUTION
            notes.append((time_sec, dur_sec, pitch))
        
        if len(notes) == 0:
            return None, []
        
        # Determine time range
        max_time = max(n[0] + n[1] for n in notes)
        num_frames = int(np.ceil(max_time * time_resolution)) + 1
        
        # Create piano roll
        piano_roll = np.zeros((num_frames, 128), dtype=np.float32)
        for time_sec, dur_sec, pitch in notes:
            if 0 <= pitch < 128:
                start_frame = int(time_sec * time_resolution)
                end_frame = int((time_sec + dur_sec) * time_resolution)
                piano_roll[start_frame:end_frame + 1, pitch] = 1.0
        
        # Add timing column (inter-frame interval = 1/time_resolution seconds)
        timing = np.full((num_frames, 1), 1.0 / time_resolution, dtype=np.float32)
        features = np.hstack([piano_roll, timing])
        
        # Build note events with frame indices
        note_events = []
        for time_sec, dur_sec, pitch in notes:
            frame_idx = int(time_sec * time_resolution)
            note_events.append((time_sec, dur_sec, pitch, frame_idx))
        
        return features, note_events
        
    except Exception as e:
        return None, []


def align_atepp_dtw(perf_midi, score_musicxml, ds=0.01, lmbda=1.0):
    """
    Align ATEPP performance MIDI with MusicXML score using DTW.
    
    Args:
        perf_midi: path to performance MIDI file
        score_musicxml: path to MusicXML score file
        ds: time step size for DTW
        lmbda: tempo regularization parameter
    
    Returns: List of [perf_triplet, perf_idx, score_triplet, score_idx]
    """
    try:
        # Convert MusicXML to MIDI for consistent processing
        score = m21_converter.parse(score_musicxml)
        with tempfile.NamedTemporaryFile(suffix='.mid', delete=False) as f:
            temp_score_midi = f.name
        score.write('midi', fp=temp_score_midi)
        
        # Convert both to piano roll representations
        perf_features, perf_notes = midi_to_piano_roll(perf_midi)
        score_features, score_notes = midi_to_piano_roll(temp_score_midi)
        
        # Clean up temp file
        try:
            os.unlink(temp_score_midi)
        except:
            pass
        
        if perf_features is None or score_features is None:
            return []
        
        if len(perf_notes) == 0 or len(score_notes) == 0:
            return []
        
        # Run DTW alignment (use Cython if available)
        if USE_CYTHON_DTW:
            # Cython version expects memory views
            score_f32 = np.ascontiguousarray(score_features, dtype=np.float32)
            perf_f32 = np.ascontiguousarray(perf_features[:, :128], dtype=np.float32)
            L = cython_dtw_align(score_f32, perf_f32, ds, lmbda)
            alignment, costs = cython_dtw_traceback(score_f32, perf_f32, L, ds, lmbda)
        else:
            L = dtw_align(score_features, perf_features[:, :128], ds, lmbda)
            alignment, costs = dtw_traceback(score_features, perf_features[:, :128], L, ds, lmbda)
        
        # Build frame-to-frame mapping from alignment
        # alignment is list of (score_frame, perf_frame) pairs
        frame_map = {}
        for score_frame, perf_frame in alignment:
            if score_frame not in frame_map:
                frame_map[score_frame] = perf_frame
        
        # Match notes based on frame alignment
        matched_tuples = []
        used_perf_indices = set()
        
        for s_idx, (s_time, s_dur, s_pitch, s_frame) in enumerate(score_notes):
            # Find corresponding performance frame
            if s_frame in frame_map:
                target_p_frame = frame_map[s_frame]
            else:
                # Interpolate from nearby frames
                nearby = [f for f in frame_map.keys() if abs(f - s_frame) <= 5]
                if nearby:
                    closest = min(nearby, key=lambda f: abs(f - s_frame))
                    target_p_frame = frame_map[closest]
                else:
                    continue
            
            # Find best matching performance note near target frame with same pitch
            best_match = None
            best_dist = np.inf
            best_p_idx = None
            
            for p_idx, (p_time, p_dur, p_pitch, p_frame) in enumerate(perf_notes):
                if p_idx in used_perf_indices:
                    continue
                if p_pitch != s_pitch:
                    continue
                
                frame_dist = abs(p_frame - target_p_frame)
                if frame_dist < best_dist and frame_dist <= 50:  # within 0.5 sec at 100fps
                    best_dist = frame_dist
                    best_match = (p_time, p_dur, p_pitch)
                    best_p_idx = p_idx
            
            if best_match is not None:
                used_perf_indices.add(best_p_idx)
                
                # Convert to token format
                perf_triplet = [
                    round(best_match[0] * TIME_RESOLUTION) + ATIME_OFFSET,
                    round(best_match[1] * TIME_RESOLUTION) + ADUR_OFFSET,
                    best_match[2] + ANOTE_OFFSET
                ]
                score_triplet = [
                    round(s_time * TIME_RESOLUTION) + TIME_OFFSET,
                    round(s_dur * TIME_RESOLUTION) + DUR_OFFSET,
                    s_pitch + NOTE_OFFSET
                ]
                matched_tuples.append([perf_triplet, best_p_idx, score_triplet, s_idx])
        
        # Sort by performance index
        matched_tuples.sort(key=lambda x: x[1])
        
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
        matched_tuples = align_tokens2(perf_midi, score_midi, perf_beats, score_beats, skip_Nones=True)
        
        if len(matched_tuples) < 20:
            return []
        
        # Load score beat annotations for time normalization
        score_annotations = load_annotation_file(score_beats)
        score_beat_times = [anno[0] for anno in score_annotations]
        
        TARGET_BEAT_INTERVAL = 0.5
        
        # Pre-normalize ALL score triplets using beat mapping
        normalized_matched_tuples = []
        for match in matched_tuples:
            perf_triplet = match[0]
            score_triplet = match[2]
            
            if score_triplet[0] is not None:
                original_time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
                original_duration_sec = (score_triplet[1] - DUR_OFFSET) / TIME_RESOLUTION
                pitch = score_triplet[2]
                
                normalized_time_sec = 0.0
                time_scale_factor = 1.0
                
                if score_beat_times and len(score_beat_times) >= 2:
                    if original_time_sec < score_beat_times[0]:
                        beat_duration = score_beat_times[1] - score_beat_times[0]
                        if beat_duration > 0:
                            progress = (original_time_sec - score_beat_times[0]) / beat_duration
                            time_scale_factor = TARGET_BEAT_INTERVAL / beat_duration
                        else:
                            progress = 0
                            time_scale_factor = 1.0
                        normalized_time_sec = 0.0 + progress * TARGET_BEAT_INTERVAL
                    else:
                        found = False
                        for i in range(len(score_beat_times) - 1):
                            if score_beat_times[i] <= original_time_sec <= score_beat_times[i + 1]:
                                beat_duration = score_beat_times[i + 1] - score_beat_times[i]
                                if beat_duration > 0:
                                    progress = (original_time_sec - score_beat_times[i]) / beat_duration
                                    time_scale_factor = TARGET_BEAT_INTERVAL / beat_duration
                                else:
                                    progress = 0
                                    time_scale_factor = 1.0
                                normalized_time_sec = i * TARGET_BEAT_INTERVAL + progress * TARGET_BEAT_INTERVAL
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
                                progress = 0
                                time_scale_factor = 1.0
                            normalized_time_sec = last_beat_idx * TARGET_BEAT_INTERVAL + progress * TARGET_BEAT_INTERVAL
                else:
                    normalized_time_sec = original_time_sec - (score_beat_times[0] if score_beat_times else 0)
                    time_scale_factor = 1.0
                
                normalized_duration_sec = original_duration_sec * time_scale_factor
                
                normalized_time_units = round(normalized_time_sec * TIME_RESOLUTION)
                normalized_duration_units = round(normalized_duration_sec * TIME_RESOLUTION)
                normalized_time_units = max(0, normalized_time_units)
                normalized_duration_units = max(0, normalized_duration_units)
                normalized_score = [
                    normalized_time_units + TIME_OFFSET,
                    normalized_duration_units + DUR_OFFSET,
                    pitch
                ]
            else:
                normalized_score = score_triplet
            
            normalized_matched_tuples.append([perf_triplet, match[1], normalized_score, match[3]])
        
        return _build_sequences(normalized_matched_tuples, prefix_controls)
        
    except Exception as e:
        return []


def tokenize_sliding_windows_atepp(filegroup, prefix_controls=33):
    """
    Tokenize an ATEPP performance-score pair using DTW alignment.
    """
    _, perf_midi, score_musicxml, composition_id = filegroup
    
    try:
        # Align using DTW algorithm
        matched_tuples = align_atepp_dtw(perf_midi, score_musicxml, ds=0.01, lmbda=1.0)
        
        if len(matched_tuples) < 20:
            return []
        
        # For ATEPP without beat annotations, we normalize score time 
        # by estimating beats from note density
        # Simpler approach: just shift score times to start at 0
        
        normalized_matched_tuples = []
        
        # Get min score time
        score_times = [m[2][0] - TIME_OFFSET for m in matched_tuples if m[2][0] is not None]
        min_score_time = min(score_times) if score_times else 0
        
        for match in matched_tuples:
            perf_triplet = match[0]
            score_triplet = match[2]
            
            if score_triplet[0] is not None:
                # Shift score time to start at 0
                normalized_time = score_triplet[0] - min_score_time
                normalized_score = [
                    max(0, normalized_time),
                    score_triplet[1],
                    score_triplet[2]
                ]
            else:
                normalized_score = score_triplet
            
            normalized_matched_tuples.append([perf_triplet, match[1], normalized_score, match[3]])
        
        return _build_sequences(normalized_matched_tuples, prefix_controls)
        
    except Exception as e:
        return []


def _build_sequences(normalized_matched_tuples, prefix_controls=33):
    """
    Build 1024-token sequences from normalized matched tuples.
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
        
        # Extract already-normalized score triplets
        score_triplets = [match[2] for match in subset]
        
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
        
        # Prepend 3 SEPs
        interleaved_tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]
        
        max_body = EVENT_SIZE * M
        if len(interleaved_tokens) < max_body:
            break
        
        interleaved_tokens = interleaved_tokens[:max_body]
        
        if ops.max_time(interleaved_tokens, seconds=False) >= MAX_TIME:
            continue
        
        sequence = [ANTICIPATE] + interleaved_tokens
        
        assert len(sequence) == CONTEXT_SIZE, f"Expected {CONTEXT_SIZE} tokens, got {len(sequence)}"
        
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
    
    train_sequences_total = 0
    train_pieces_success = 0
    train_pieces_failed = 0
    train_asap_success = 0
    train_atepp_success = 0
    
    with open(TRAIN_OUTPUT, 'w') as f_train:
        with Pool(processes=NUM_WORKERS) as pool:
            with tqdm(total=len(train_pairs), desc='Train', unit='piece') as pbar:
                for sequences, count, dataset_type in pool.imap_unordered(process_single_piece, train_pairs):
                    if count > 0:
                        for seq in sequences:
                            f_train.write(seq + '\n')
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
    
    with open(TEST_OUTPUT, 'w') as f_test:
        with Pool(processes=NUM_WORKERS) as pool:
            with tqdm(total=len(test_pairs), desc='Test', unit='piece') as pbar:
                for sequences, count, dataset_type in pool.imap_unordered(process_single_piece, test_pairs):
                    if count > 0:
                        for seq in sequences:
                            f_test.write(seq + '\n')
                        test_sequences_total += count
                        test_pieces_success += 1
                        if dataset_type == 'asap':
                            test_asap_success += 1
                        else:
                            test_atepp_success += 1
                    else:
                        test_pieces_failed += 1
                    pbar.update(1)
    
    print(f"Test: {test_sequences_total} sequences from {test_pieces_success} pieces ({test_asap_success} ASAP, {test_atepp_success} ATEPP), {test_pieces_failed} failed")
    
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
        print(f"Mode token: {first_seq[0]} (expected {ANTICIPATE})")
        print(f"Bootstrap: {first_seq[1:4]} (expected {[SEPARATOR, SEPARATOR, SEPARATOR]})")
        
        control_count = 0
        score_count = 0
        rest_count = 0
        
        for i in range(min(100, (len(first_seq) - 4) // 3)):
            pos = 4 + i * 3
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
    print(f"  {TEST_OUTPUT}")
    print(f"  {SPLIT_FILE}")
    print("\nTokenization complete!")
