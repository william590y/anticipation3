"""
Evaluate checkpoint using MUSTER (Music Score Transcription Error Rate) metrics.

MUSTER provides edit-distance-based metrics for evaluating music score transcription:
- Pitch Error Rate (PER)
- Missing Note Rate (MNR)  
- Extra Note Rate (ENR)
- Onset Time Error Rate (OTER)
- Offset Time Error Rate (OFTER)
- Mean of above (MER)
- Voice Error Rate (VER)

Reference: https://amtevaluation.github.io/

Usage:
    python evaluate_muster.py --checkpoint checkpoint-1750 --num-examples 10
"""

import os
import sys
import json
import subprocess
import tempfile
import shutil
import torch
import random
import numpy as np
from pathlib import Path
from transformers import AutoModelForCausalLM
from anticipation.vocab import *
from anticipation.config import *
from anticipation.convert import events_to_midi
from tqdm import tqdm
import music21
from music21 import converter

# Configuration
DEFAULT_CHECKPOINT = 'checkpoint-1750'
TEST_FILE = 'data/test_combined.txt'
OUTPUT_BASE = 'muster_evaluation_results'
NUM_EXAMPLES = 10  # Randomly sample sequences
RANDOM_SEED = 42
K_PREFIX = 33  # Number of control+rest pairs in prefix
ALTERNATING_START = 4 + K_PREFIX * 6  # = 202

# MUSTER programs location
MUSTER_DIR = Path(__file__).parent / 'MUSTER'
MUSTER_PROGRAMS = MUSTER_DIR / 'Programs'

# Determine executable extension based on platform
IS_WINDOWS = sys.platform.startswith('win')
EXE_EXT = '.exe' if IS_WINDOWS else ''


def get_muster_exe(name):
    """Get path to MUSTER executable, handling platform differences."""
    return str(MUSTER_PROGRAMS / f'{name}{EXE_EXT}')


def compile_muster_linux():
    """Compile MUSTER programs on Linux."""
    print("Compiling MUSTER programs for Linux...")
    os.makedirs(MUSTER_PROGRAMS, exist_ok=True)
    
    code_dir = MUSTER_DIR / 'Code'
    compilations = [
        ('Fmt3xToSpr_v220118.cpp', 'Fmt3xToSpr'),
        ('ScoreMatchEvaluation_VoicePlus_v220118.cpp', 'ScoreMatchEvaluation_VoicePlus'),
        ('MusicXMLToFmt3x_v170104.cpp', 'MusicXMLToFmt3x'),
        ('MusicXMLToHMM_v170104.cpp', 'MusicXMLToHMM'),
        ('ScorePerfmMatcher_v170503.cpp', 'ScorePerfmMatcher'),
        ('ErrorDetection_v170503.cpp', 'ErrorDetection'),
        ('RealignmentMOHMM_v190326.cpp', 'RealignmentMOHMM'),
    ]
    
    for src, name in compilations:
        src_path = code_dir / src
        out_path = MUSTER_PROGRAMS / name
        if not out_path.exists():
            print(f"  Compiling {name}...")
            result = subprocess.run(
                ['g++', '-O2', str(src_path), '-o', str(out_path)],
                capture_output=True, text=True
            )
            if result.returncode != 0:
                print(f"    ERROR: {result.stderr}")
                return False
    
    print("MUSTER compilation complete.")
    return True


def check_muster_installation():
    """Verify MUSTER programs are compiled."""
    required_programs = [
        'MusicXMLToFmt3x',
        'MusicXMLToHMM',
        'Fmt3xToSpr',
        'ScorePerfmMatcher',
        'ErrorDetection',
        'RealignmentMOHMM',
        'ScoreMatchEvaluation_VoicePlus'
    ]
    
    missing = []
    for prog in required_programs:
        exe_path = MUSTER_PROGRAMS / f'{prog}{EXE_EXT}'
        if not exe_path.exists():
            missing.append(prog)
    
    if missing:
        if IS_WINDOWS:
            print(f"ERROR: Missing MUSTER programs: {missing}")
            print("Please compile MUSTER first by running compile.sh in the MUSTER folder")
            sys.exit(1)
        else:
            # Try to compile on Linux
            print(f"Missing MUSTER programs: {missing}")
            if not compile_muster_linux():
                print("ERROR: Failed to compile MUSTER programs")
                sys.exit(1)
            # Re-check
            still_missing = [p for p in required_programs 
                           if not (MUSTER_PROGRAMS / f'{p}{EXE_EXT}').exists()]
            if still_missing:
                print(f"ERROR: Still missing after compilation: {still_missing}")
                sys.exit(1)
    
    return True


def load_model(checkpoint_path):
    """Load model from checkpoint."""
    print(f"Loading model from {checkpoint_path}...")
    model = AutoModelForCausalLM.from_pretrained(checkpoint_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    print(f"  Model loaded on {device}")
    return model, device


def parse_sequence(line):
    """Parse a sequence from the test file."""
    if '|' in line:
        token_str, _ = line.split('|')
        tokens = [int(t) for t in token_str.strip().split()]
    else:
        tokens = [int(t) for t in line.strip().split()]
    tokens = [max(0, t) for t in tokens]
    return tokens


def extract_components(tokens, score_start_idx):
    """Extract performance, score, and control information from interleaved sequence."""
    performance = []
    for i in range(4, score_start_idx, 6):
        if i + 2 < len(tokens):
            ctrl_time = tokens[i] - ATIME_OFFSET
            ctrl_dur = tokens[i + 1] - ADUR_OFFSET
            ctrl_pitch = tokens[i + 2] - ANOTE_OFFSET
            performance.append([ctrl_time, ctrl_dur, ctrl_pitch])
    
    alternating = tokens[score_start_idx:]
    score_triplets = []
    
    pos = 0
    while pos + 2 < len(alternating):
        t0, t1, t2 = alternating[pos], alternating[pos+1], alternating[pos+2]
        
        if t0 < CONTROL_OFFSET and t1 < CONTROL_OFFSET and t2 < CONTROL_OFFSET and t2 != REST:
            score_triplets.append([t0, t1, t2])
            pos += 3
            
            if pos + 2 < len(alternating):
                c0, c1, c2 = alternating[pos], alternating[pos+1], alternating[pos+2]
                if c0 >= CONTROL_OFFSET and c1 >= CONTROL_OFFSET and c2 >= CONTROL_OFFSET:
                    performance.append([c0 - ATIME_OFFSET, c1 - ADUR_OFFSET, c2 - ANOTE_OFFSET])
                    pos += 3
                else:
                    break
            else:
                break
        else:
            break
    
    return performance, score_triplets


def autoregressive_generate_score(model, tokens, score_start_idx, device):
    """Generate score tokens autoregressively while keeping control tokens fixed."""
    context = list(tokens[:score_start_idx])
    
    pos = score_start_idx
    while pos + 5 < len(tokens):
        if (tokens[pos] < CONTROL_OFFSET and 
            tokens[pos+1] < CONTROL_OFFSET and 
            tokens[pos+2] < CONTROL_OFFSET and
            tokens[pos+2] != REST):
            
            with torch.no_grad():
                input_tensor = torch.tensor([context], device=device)
                outputs = model(input_tensor)
                pred_time = outputs.logits[0, -1, :].argmax().item()
                context.append(pred_time)
                
                input_tensor = torch.tensor([context], device=device)
                outputs = model(input_tensor)
                pred_dur = outputs.logits[0, -1, :].argmax().item()
                context.append(pred_dur)
                
                input_tensor = torch.tensor([context], device=device)
                outputs = model(input_tensor)
                pred_pitch = outputs.logits[0, -1, :].argmax().item()
                context.append(pred_pitch)
            
            pos += 3
            
            if pos + 2 < len(tokens):
                context.extend([tokens[pos], tokens[pos+1], tokens[pos+2]])
                pos += 3
        else:
            context.append(tokens[pos])
            pos += 1
    
    return context


def triplets_to_events(triplets):
    """Convert list of [time, dur, pitch] triplets to flat event list."""
    events = []
    for t in triplets:
        events.extend(t)
    return events


def normalize_triplet_times(triplets):
    """Normalize triplet times to start at 0 and sort by time."""
    if not triplets:
        return triplets
    triplets = sorted(triplets, key=lambda t: t[0])
    min_time = min(t[0] - TIME_OFFSET for t in triplets)
    return [[t[0] - min_time, t[1], t[2]] for t in triplets]


def triplets_to_musicxml(triplets, xml_path):
    """
    Convert score triplets directly to single-part MusicXML (bypassing MIDI entirely).
    
    Builds a music21 Score from scratch with a single Part/Voice so MUSTER's
    HMM converter sees exactly one channel (nevts_ch.size()==1).
    
    Time resolution: 10ms bins (TIME_RESOLUTION=100 bins/sec).
    We use 120 BPM so 1 quarter = 0.5s = 50 bins.
    
    Args:
        triplets: list of [time_token, dur_token, pitch_token] (with vocab offsets)
        xml_path: output MusicXML path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from fractions import Fraction
        from xml.etree import ElementTree as ET
        from music21.musicxml.m21ToXml import ScoreExporter
        
        BINS_PER_SECOND = TIME_RESOLUTION        # 100
        BPM = 120
        BINS_PER_QUARTER = int(BINS_PER_SECOND * 60 / BPM)  # 50
        
        # Decode triplets to (onset_bins, dur_bins, midi_pitch)
        notes = []
        for t in triplets:
            onset_bins = t[0] - TIME_OFFSET
            dur_bins   = t[1] - DUR_OFFSET
            pitch      = t[2] - NOTE_OFFSET
            if pitch < 0 or pitch > 127:
                continue
            if dur_bins <= 0:
                dur_bins = 1
            notes.append((onset_bins, dur_bins, pitch))
        
        if not notes:
            return False
        
        # Sort by onset
        notes.sort(key=lambda x: x[0])
        
        # Build music21 score with a single part and voice
        s = music21.stream.Score()
        s.insert(0, music21.tempo.MetronomeMark(number=BPM))
        
        part = music21.stream.Part()
        part.insert(0, music21.instrument.Piano())
        
        # Insert notes at offset in quarter notes
        for onset_bins, dur_bins, pitch in notes:
            onset_quarters = Fraction(onset_bins, BINS_PER_QUARTER)
            dur_quarters   = Fraction(dur_bins,   BINS_PER_QUARTER)
            
            n = music21.note.Note()
            n.pitch.midi = pitch
            n.quarterLength = float(dur_quarters)
            if n.quarterLength <= 0:
                n.quarterLength = 0.25
            
            part.insert(float(onset_quarters), n)
        
        part.makeRests(fillGaps=True, inPlace=True)
        part.makeMeasures(inPlace=True)
        part.makeNotation(inPlace=True, cautionaryNotImmediateRepeat=False)
        
        s.insert(0, part)
        
        # Export to MusicXML 3.0
        exporter = ScoreExporter(s)
        xml_root = exporter.parse()
        xml_root.set('version', '3.0')
        
        xml_str  = b'<?xml version="1.0" encoding="UTF-8"?>\n'
        xml_str += b'<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
        xml_str += ET.tostring(xml_root, encoding='unicode').encode('utf-8')
        
        with open(xml_path, 'wb') as f:
            f.write(xml_str)
        
        return True
    except Exception as e:
        print(f"    Warning: Could not create MusicXML from triplets: {e}")
        import traceback
        traceback.print_exc()
        return False


def midi_to_musicxml(midi_path, xml_path):
    """Legacy MIDI-based conversion (kept for reference but not used)."""
    try:
        score = converter.parse(midi_path)
        score = score.quantize(quarterLengthDivisors=[4, 3, 2])
        for part in score.parts:
            part.makeNotation(inPlace=True, cautionaryNotImmediateRepeat=False)
        from music21.musicxml.m21ToXml import ScoreExporter
        from xml.etree import ElementTree as ET
        exporter = ScoreExporter(score)
        xml_root = exporter.parse()
        xml_root.set('version', '3.0')
        xml_str  = b'<?xml version="1.0" encoding="UTF-8"?>\n'
        xml_str += b'<!DOCTYPE score-partwise PUBLIC "-//Recordare//DTD MusicXML 3.0 Partwise//EN" "http://www.musicxml.org/dtds/partwise.dtd">\n'
        xml_str += ET.tostring(xml_root, encoding='unicode').encode('utf-8')
        with open(xml_path, 'wb') as f:
            f.write(xml_str)
        return True
    except Exception as e:
        print(f"    Warning: Could not convert {midi_path} to MusicXML: {e}")
        return False
        import traceback
        traceback.print_exc()
        return False


def save_midi(events, filepath):
    """Save events as MIDI file."""
    try:
        midi_obj = events_to_midi(events)
        midi_obj.save(filepath)
        return True
    except Exception as e:
        print(f"    Warning: Could not save {filepath}: {e}")
        return False


def run_muster_evaluation(gt_xml_path, pred_xml_path, output_prefix, work_dir):
    """
    Run MUSTER evaluation on a pair of MusicXML files.
    
    Args:
        gt_xml_path: Path to ground truth MusicXML
        pred_xml_path: Path to predicted MusicXML
        output_prefix: Prefix for output files
        work_dir: Working directory for intermediate files
        
    Returns:
        Dictionary with MUSTER metrics, or None if evaluation failed
    """
    try:
        # Use simple names within work directory
        gt_name = "gt"
        pred_name = "est"
        
        # Copy XML files to work directory with simple names
        gt_work = work_dir / f"{gt_name}.xml"
        pred_work = work_dir / f"{pred_name}.xml"
        shutil.copy(gt_xml_path, gt_work)
        shutil.copy(pred_xml_path, pred_work)
        
        # Verify files exist
        if not gt_work.exists():
            print(f"    GT file not copied: {gt_work}")
            return None
        if not pred_work.exists():
            print(f"    EST file not copied: {pred_work}")
            return None
        
        # Run MUSTER pipeline using relative filenames within work_dir
        # 1. Convert estimated score to score performance representation
        r = subprocess.run([
            get_muster_exe('MusicXMLToFmt3x'),
            f'{pred_name}.xml',
            f'{pred_name}_fmt3x.txt'
        ], cwd=str(work_dir), capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    Step 1 failed (MusicXMLToFmt3x EST): {r.stderr}")
            return None
        
        r = subprocess.run([
            get_muster_exe('Fmt3xToSpr'),
            f'{pred_name}_fmt3x.txt',
            f'{pred_name}_spr.txt'
        ], cwd=str(work_dir), capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    Step 2 failed (Fmt3xToSpr): {r.stderr}")
            return None
        
        # 2. Convert ground truth to HMM and Fmt3x
        r = subprocess.run([
            get_muster_exe('MusicXMLToHMM'),
            f'{gt_name}.xml',
            f'{gt_name}_hmm.txt'
        ], cwd=str(work_dir), capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    Step 3 failed (MusicXMLToHMM GT): {r.stderr}")
            return None
        
        r = subprocess.run([
            get_muster_exe('MusicXMLToFmt3x'),
            f'{gt_name}.xml',
            f'{gt_name}_fmt3x.txt'
        ], cwd=str(work_dir), capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    Step 4 failed (MusicXMLToFmt3x GT): {r.stderr}")
            return None
        
        # 3. Run score-performance matching
        r = subprocess.run([
            get_muster_exe('ScorePerfmMatcher'),
            f'{gt_name}_hmm.txt',
            f'{pred_name}_spr.txt',
            f'{pred_name}_pre_match.txt',
            '0.01'
        ], cwd=str(work_dir), capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    Step 5 failed (ScorePerfmMatcher): {r.stderr}")
            return None
        
        # 4. Error detection
        r = subprocess.run([
            get_muster_exe('ErrorDetection'),
            f'{gt_name}_fmt3x.txt',
            f'{gt_name}_hmm.txt',
            f'{pred_name}_pre_match.txt',
            f'{pred_name}_err_match.txt'
        ], cwd=str(work_dir), capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    Step 6 failed (ErrorDetection): {r.stderr}")
            return None
        
        # 5. Realignment
        r = subprocess.run([
            get_muster_exe('RealignmentMOHMM'),
            f'{gt_name}_fmt3x.txt',
            f'{gt_name}_hmm.txt',
            f'{pred_name}_err_match.txt',
            f'{pred_name}_auto_match.txt',
            '0.3'
        ], cwd=str(work_dir), capture_output=True, text=True)
        if r.returncode != 0:
            print(f"    Step 7 failed (RealignmentMOHMM): {r.stderr}")
            return None
        
        # 6. Score match evaluation
        result = subprocess.run([
            get_muster_exe('ScoreMatchEvaluation_VoicePlus'),
            f'{gt_name}_fmt3x.txt',
            f'{pred_name}_fmt3x.txt',
            f'{pred_name}_auto_match.txt',
            f'{pred_name}_err_detail.txt',
            '-1'
        ], cwd=str(work_dir), capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"    Step 8 failed (ScoreMatchEvaluation): {result.stderr}")
            return None
        
        # Parse output - MUSTER outputs header line then values
        # Format: "PitchER,MissRate,...:  val1 val2 ..."
        output_line = result.stdout.strip()
        if output_line and ':' in output_line:
            # Extract values after the colon
            values_str = output_line.split(':')[-1].strip()
            values = values_str.split()
            if len(values) >= 8:
                metrics = {
                    'pitch_error_rate': float(values[0]),
                    'missing_note_rate': float(values[1]),
                    'extra_note_rate': float(values[2]),
                    'onset_time_error_rate': float(values[3]),
                    'offset_time_error_rate': float(values[4]),
                    'mean_error_rate': float(values[5]),
                    'voice_error_rate': float(values[6]),
                    'mean_error_rate_with_voice': float(values[7])
                }
                if len(values) >= 13:
                    metrics.update({
                        'voice_precision': float(values[8]),
                        'voice_recall': float(values[9]),
                        'voice_f_measure': float(values[10]),
                        'note_value_scale_error': float(values[11]),
                        'hand_error_rate': float(values[12])
                    })
                
                # Save raw output
                with open(work_dir / f'{output_prefix}_ER.txt', 'w') as f:
                    f.write(output_line)
                
                return metrics
        
        print(f"    Warning: Could not parse MUSTER output: {output_line[:100] if output_line else '(empty)'}")
        return None
        
    except Exception as e:
        print(f"    Warning: Error processing {pred_xml_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def evaluate_checkpoint_muster(checkpoint_path, test_lines, original_indices, output_dir):
    """
    Evaluate a checkpoint using MUSTER metrics.
    
    Returns aggregate MUSTER statistics.
    """
    model, device = load_model(checkpoint_path)
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Aggregate MUSTER metrics
    aggregate_metrics = {
        'pitch_error_rate': [],
        'missing_note_rate': [],
        'extra_note_rate': [],
        'onset_time_error_rate': [],
        'offset_time_error_rate': [],
        'mean_error_rate': [],
        'voice_error_rate': [],
        'mean_error_rate_with_voice': []
    }
    
    per_sequence_metrics = []
    num_successful = 0
    num_failed = 0
    
    for i, (line, orig_idx) in enumerate(tqdm(zip(test_lines, original_indices), 
                                               total=len(test_lines),
                                               desc=f"Evaluating {checkpoint_path}")):
        tokens = parse_sequence(line)
        
        if len(tokens) <= ALTERNATING_START:
            num_failed += 1
            continue
        
        score_start_idx = ALTERNATING_START
        
        # Extract ground truth score
        gt_perf, gt_score = extract_components(tokens, score_start_idx)
        
        if len(gt_score) < 5:  # Need minimum notes for meaningful MUSTER evaluation
            num_failed += 1
            continue
        
        # Generate predictions
        try:
            predicted_tokens = autoregressive_generate_score(model, tokens, score_start_idx, device)
        except Exception as e:
            print(f"  Sequence {orig_idx}: Generation failed - {e}")
            num_failed += 1
            continue
        
        # Extract predicted score
        _, pred_score = extract_components(predicted_tokens, score_start_idx)
        
        if len(pred_score) < 3:
            print(f"  Sequence {orig_idx}: Too few predicted notes")
            num_failed += 1
            continue
        
        # Create sequence directory
        seq_dir = Path(output_dir) / f'sequence_{orig_idx:04d}'
        os.makedirs(seq_dir, exist_ok=True)
        
        # Normalize triplets (shift onset to start at 0)
        gt_score_normalized  = normalize_triplet_times(gt_score)
        pred_score_normalized = normalize_triplet_times(pred_score)
        
        # Save MIDI for reference
        save_midi(triplets_to_events(gt_score_normalized),   str(seq_dir / 'ground_truth_score.mid'))
        save_midi(triplets_to_events(pred_score_normalized), str(seq_dir / 'output_score.mid'))
        
        # Convert DIRECTLY from triplets to single-part MusicXML (bypasses MIDI
        # multi-channel issue that causes MUSTER's HMM assertion to fail)
        gt_xml_path   = seq_dir / 'ground_truth_score.xml'
        pred_xml_path = seq_dir / 'output_score.xml'
        
        if not triplets_to_musicxml(gt_score_normalized, str(gt_xml_path)):
            num_failed += 1
            continue
        if not triplets_to_musicxml(pred_score_normalized, str(pred_xml_path)):
            num_failed += 1
            continue
        
        # Run MUSTER evaluation
        work_dir = seq_dir / 'muster_work'
        os.makedirs(work_dir, exist_ok=True)
        
        metrics = run_muster_evaluation(
            gt_xml_path, pred_xml_path,
            f'seq_{orig_idx:04d}',
            work_dir
        )
        
        if metrics:
            # Save per-sequence metrics
            metrics['original_index'] = orig_idx
            metrics['num_gt_notes'] = len(gt_score)
            metrics['num_pred_notes'] = len(pred_score)
            per_sequence_metrics.append(metrics)
            
            # Update aggregates
            for key in aggregate_metrics:
                if key in metrics:
                    aggregate_metrics[key].append(metrics[key])
            
            num_successful += 1
            
            # Save metrics for this sequence
            with open(seq_dir / 'muster_metrics.json', 'w') as f:
                json.dump(metrics, f, indent=2)
        else:
            num_failed += 1
    
    # Compute aggregate statistics
    final_aggregate = {
        'num_sequences_evaluated': num_successful,
        'num_sequences_failed': num_failed,
    }
    
    for key, values in aggregate_metrics.items():
        if values:
            final_aggregate[f'{key}_mean'] = np.mean(values)
            final_aggregate[f'{key}_std'] = np.std(values)
            final_aggregate[f'{key}_min'] = np.min(values)
            final_aggregate[f'{key}_max'] = np.max(values)
    
    # Save aggregate statistics
    with open(Path(output_dir) / 'aggregate_muster_stats.json', 'w') as f:
        json.dump(final_aggregate, f, indent=2)
    
    # Save per-sequence statistics
    with open(Path(output_dir) / 'per_sequence_muster_stats.json', 'w') as f:
        json.dump(per_sequence_metrics, f, indent=2)
    
    return final_aggregate


def print_muster_summary(checkpoint_name, stats):
    """Print MUSTER summary statistics."""
    print(f"\n{'='*70}")
    print(f"MUSTER Results for {checkpoint_name}")
    print(f"{'='*70}")
    print(f"  Sequences evaluated: {stats['num_sequences_evaluated']}")
    print(f"  Sequences failed: {stats['num_sequences_failed']}")
    print()
    
    metrics_to_show = [
        ('pitch_error_rate', 'Pitch Error Rate (PER)'),
        ('missing_note_rate', 'Missing Note Rate (MNR)'),
        ('extra_note_rate', 'Extra Note Rate (ENR)'),
        ('onset_time_error_rate', 'Onset Time Error Rate (OTER)'),
        ('offset_time_error_rate', 'Offset Time Error Rate (OFTER)'),
        ('mean_error_rate', 'Mean Error Rate (MER)'),
        ('voice_error_rate', 'Voice Error Rate (VER)'),
        ('mean_error_rate_with_voice', 'Mean with Voice (MER+V)'),
    ]
    
    print("  MUSTER Metrics (lower is better):")
    print("  " + "-"*50)
    for key, name in metrics_to_show:
        mean_key = f'{key}_mean'
        std_key = f'{key}_std'
        if mean_key in stats:
            print(f"  {name:<30} {stats[mean_key]:>8.2f}% (±{stats[std_key]:.2f})")
    print()


def main(checkpoint=None, test_file=None, num_examples=None):
    if checkpoint is None:
        checkpoint = DEFAULT_CHECKPOINT
    if test_file is None:
        test_file = TEST_FILE
    if num_examples is None:
        num_examples = NUM_EXAMPLES
    
    print("="*80)
    print("MUSTER EVALUATION")
    print("="*80)
    print(f"Checkpoint: {checkpoint}")
    print(f"Test file: {test_file}")
    print()
    
    # Check MUSTER installation
    check_muster_installation()
    
    # Check test file exists
    if not os.path.exists(test_file):
        print(f"ERROR: Test file not found: {test_file}")
        sys.exit(1)
    
    # Check checkpoint exists
    if not os.path.exists(checkpoint):
        print(f"ERROR: Checkpoint not found: {checkpoint}")
        sys.exit(1)
    
    # Load test data
    print(f"Loading test data from {test_file}...")
    with open(test_file, 'r') as f:
        all_lines = [line.strip() for line in f if line.strip()]
    print(f"  Found {len(all_lines)} total test sequences")
    
    # Randomly sample sequences
    random.seed(RANDOM_SEED)
    if num_examples is not None and num_examples < len(all_lines):
        sampled_indices = random.sample(range(len(all_lines)), num_examples)
        sampled_indices.sort()
        test_lines = [all_lines[i] for i in sampled_indices]
        print(f"  Randomly sampled {num_examples} sequences (seed={RANDOM_SEED})")
    else:
        test_lines = all_lines
        sampled_indices = list(range(len(all_lines)))
    print()
    
    # Create output directory
    output_dir = Path(OUTPUT_BASE) / checkpoint
    os.makedirs(output_dir, exist_ok=True)
    
    # Save sampled indices
    with open(output_dir / 'sampled_indices.json', 'w') as f:
        json.dump({
            'seed': RANDOM_SEED, 
            'num_samples': len(sampled_indices), 
            'indices': sampled_indices
        }, f, indent=2)
    
    # Evaluate
    stats = evaluate_checkpoint_muster(checkpoint, test_lines, sampled_indices, str(output_dir))
    
    # Print summary
    print_muster_summary(checkpoint, stats)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print("\nEach sequence folder contains:")
    print("  • ground_truth_score.mid / .xml - Ground truth score")
    print("  • output_score.mid / .xml - Model predictions")
    print("  • muster_metrics.json - Per-sequence MUSTER metrics")
    print("  • muster_work/ - Intermediate MUSTER files")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate checkpoint using MUSTER metrics')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT,
                        help=f'Path to checkpoint (default: {DEFAULT_CHECKPOINT})')
    parser.add_argument('--test-file', type=str, default=TEST_FILE,
                        help=f'Path to test file (default: {TEST_FILE})')
    parser.add_argument('--num-examples', type=int, default=NUM_EXAMPLES,
                        help=f'Number of examples to evaluate (default: {NUM_EXAMPLES})')
    args = parser.parse_args()
    
    main(checkpoint=args.checkpoint, test_file=args.test_file, num_examples=args.num_examples)
