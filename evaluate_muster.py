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
NUM_EXAMPLES = 25  # Randomly sample sequences
RANDOM_SEED = 41
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


def autoregressive_generate_score(model, tokens, score_start_idx, device,
                                   forced=False, forced_max_attempts=1000,
                                   beam_size=1, temperature=0.0):
    """
    Generate score tokens autoregressively while keeping control tokens fixed.

    Modes (mutually exclusive, checked in this order):

    beam_size > 1  – token-level beam search over score tokens.  At each of the
        3 score-token slots the top-beam_size continuations from every active
        beam are considered, and the global top-beam_size are kept.  Control
        triplets are identical (GT) across all beams.  The highest
        log-probability beam is returned at the end.

    forced=True  – for each score triplet position, sample a full
        (time, dur, pitch) triplet autoregressively.  If the generated PITCH
        does not match the GT pitch, roll the context back and resample the
        whole triplet.  Repeat until the pitch matches or forced_max_attempts
        triplet draws are exhausted; on budget exhaustion the last predicted
        time/dur are kept and only the GT pitch is injected.

    default  – greedy (argmax) decoding at each score token.

    temperature  – when > 0, the greedy path samples from softmax(logits/T)
        instead of taking the argmax.  Also applied in forced mode (changes
        the sampling distribution) and beam search (rescales log-probs).
        temperature=0 (default) always uses argmax/pure beam.

    Returns:
        (predicted_token_list, stats_dict)

        stats_dict keys when forced:
            'total_triplet_attempts' – total triplet draws across all positions
            'positions_forced'       – positions where GT pitch was injected
        stats_dict keys when beam:
            'beam_log_prob'          – log-prob of the winning beam
    """

    # ------------------------------------------------------------------ #
    #  BEAM SEARCH
    # ------------------------------------------------------------------ #
    if beam_size > 1:
        # Each beam is (context_list, cumulative_log_prob)
        beams = [(list(tokens[:score_start_idx]), 0.0)]

        pos = score_start_idx
        while pos + 5 < len(tokens):
            is_score = (
                tokens[pos]   < CONTROL_OFFSET and
                tokens[pos+1] < CONTROL_OFFSET and
                tokens[pos+2] < CONTROL_OFFSET and
                tokens[pos+2] != REST
            )
            if is_score:
                # Expand beams over each of the 3 score token slots
                for _slot in range(3):
                    candidates = []
                    for ctx, lp in beams:
                        with torch.no_grad():
                            inp = torch.tensor([ctx], device=device)
                            logits = model(inp).logits[0, -1, :]
                            if temperature > 0:
                                logits = logits / temperature
                            log_probs = torch.log_softmax(logits, dim=-1)
                        top_lps, top_toks = torch.topk(log_probs, beam_size)
                        for tok, tlp in zip(top_toks.tolist(), top_lps.tolist()):
                            candidates.append((ctx + [tok], lp + tlp))
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    beams = candidates[:beam_size]

                pos += 3
                # All beams receive the same GT control triplet
                if pos + 2 < len(tokens):
                    beams = [(ctx + [tokens[pos], tokens[pos+1], tokens[pos+2]], lp)
                             for ctx, lp in beams]
                    pos += 3
            else:
                beams = [(ctx + [tokens[pos]], lp) for ctx, lp in beams]
                pos += 1

        best_ctx, best_lp = max(beams, key=lambda x: x[1])
        return best_ctx, {'beam_log_prob': best_lp}

    # ------------------------------------------------------------------ #
    #  FORCED / GREEDY
    # ------------------------------------------------------------------ #
    context = list(tokens[:score_start_idx])
    stats = {'total_triplet_attempts': 0, 'positions_forced': 0}

    pos = score_start_idx
    while pos + 5 < len(tokens):
        if (tokens[pos]   < CONTROL_OFFSET and
            tokens[pos+1] < CONTROL_OFFSET and
            tokens[pos+2] < CONTROL_OFFSET and
            tokens[pos+2] != REST):

            gt_pitch = tokens[pos+2]

            if forced:
                # Sample a full triplet; retry until pitch matches GT
                matched = False
                for _attempt in range(forced_max_attempts):
                    stats['total_triplet_attempts'] += 1
                    ctx_before = list(context)      # save for rollback
                    with torch.no_grad():
                        def _sample(ctx):
                            inp = torch.tensor([ctx], device=device)
                            logits = model(inp).logits[0, -1, :]
                            if temperature > 0:
                                logits = logits / temperature
                            return torch.multinomial(torch.softmax(logits, dim=-1), 1).item()

                        tok_t = _sample(context); context.append(tok_t)
                        tok_d = _sample(context); context.append(tok_d)
                        tok_p = _sample(context); context.append(tok_p)

                    if tok_p == gt_pitch:
                        matched = True
                        break
                    context = ctx_before   # pitch wrong – roll back

                if not matched:
                    # Budget exhausted: keep last predicted time/dur, inject GT pitch
                    context[-1] = gt_pitch
                    stats['positions_forced'] += 1
            else:
                # Greedy / temperature-sampled decoding
                with torch.no_grad():
                    def _decode(ctx):
                        inp = torch.tensor([ctx], device=device)
                        logits = model(inp).logits[0, -1, :]
                        if temperature > 0:
                            logits = logits / temperature
                            return torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
                        return logits.argmax().item()

                    context.append(_decode(context))
                    context.append(_decode(context))
                    context.append(_decode(context))

            pos += 3

            # Add ground truth control triplet
            if pos + 2 < len(tokens):
                context.extend([tokens[pos], tokens[pos+1], tokens[pos+2]])
                pos += 3
        else:
            context.append(tokens[pos])
            pos += 1

    return context, stats


def autoregressive_generate_full_piece(model, full_gt_tokens, score_start_idx, device,
                                        forced=False, forced_max_attempts=1000,
                                        beam_size=1, temperature=0.0):
    """
    Generate score tokens for an ENTIRE piece using a sliding context window.

    `full_gt_tokens` may be arbitrarily longer than CONTEXT_SIZE (1024). The
    fixed header (first `score_start_idx` tokens) is always kept in the context.
    Whenever `context` grows to CONTEXT_SIZE, the first half of the alternating
    section is dropped, aligned to 6-token (score+control pair) boundaries:

        context = header + alternating_section[half:]

    GT control tokens are always inserted from `full_gt_tokens`.

    Supports the same decoding modes as `autoregressive_generate_score`
    (greedy, temperature, forced, beam).

    Returns:
        pred_score_triplets: list of [time_tok, dur_tok, pitch_tok] with vocab offsets
        stats: {num_slides, total_triplet_attempts, positions_forced, beam_log_prob}
    """
    header  = list(full_gt_tokens[:score_start_idx])
    context = list(header)
    pred_score_triplets = []
    stats = {
        'num_slides':              0,
        'total_triplet_attempts':  0,
        'positions_forced':        0,
        'beam_log_prob':           0.0,
    }

    def _run_model(ctx):
        """Single forward pass; returns logits for the last position."""
        with torch.no_grad():
            return model(torch.tensor([ctx], device=device)).logits[0, -1, :]

    def _sample_tok(ctx):
        logits = _run_model(ctx)
        if temperature > 0:
            logits = logits / temperature
        return torch.multinomial(torch.softmax(logits, dim=-1), 1).item()

    def _greedy_tok(ctx):
        logits = _run_model(ctx)
        if temperature > 0:
            logits = logits / temperature
            return torch.multinomial(torch.softmax(logits, dim=-1), 1).item()
        return logits.argmax().item()

    pos = score_start_idx
    while pos + 2 < len(full_gt_tokens):
        t0, t1, t2 = full_gt_tokens[pos], full_gt_tokens[pos+1], full_gt_tokens[pos+2]
        is_score = (t0 < CONTROL_OFFSET and t1 < CONTROL_OFFSET and
                    t2 < CONTROL_OFFSET and t2 != REST)

        if is_score:
            gt_pitch = t2

            if forced:
                matched = False
                for _attempt in range(forced_max_attempts):
                    stats['total_triplet_attempts'] += 1
                    ctx_before = list(context)
                    tok_t = _sample_tok(context); context.append(tok_t)
                    tok_d = _sample_tok(context); context.append(tok_d)
                    tok_p = _sample_tok(context); context.append(tok_p)
                    if tok_p == gt_pitch:
                        matched = True
                        break
                    context = ctx_before
                if not matched:
                    context[-1] = gt_pitch
                    stats['positions_forced'] += 1
                pred_score_triplets.append([context[-3], context[-2], context[-1]])

            elif beam_size > 1:
                beams = [(list(context), 0.0)]
                for _slot in range(3):
                    candidates = []
                    for ctx, lp in beams:
                        with torch.no_grad():
                            logits = _run_model(ctx)
                            if temperature > 0:
                                logits = logits / temperature
                            log_probs = torch.log_softmax(logits, dim=-1)
                        top_lps, top_toks = torch.topk(log_probs, beam_size)
                        for tok, tlp in zip(top_toks.tolist(), top_lps.tolist()):
                            candidates.append((ctx + [tok], lp + tlp))
                    candidates.sort(key=lambda x: x[1], reverse=True)
                    beams = candidates[:beam_size]
                best_ctx, best_lp = max(beams, key=lambda x: x[1])
                stats['beam_log_prob'] += best_lp
                pred_score_triplets.append([best_ctx[-3], best_ctx[-2], best_ctx[-1]])
                context = best_ctx

            else:
                context.append(_greedy_tok(context))
                context.append(_greedy_tok(context))
                context.append(_greedy_tok(context))
                pred_score_triplets.append([context[-3], context[-2], context[-1]])

            pos += 3

            # Append GT control triplet
            if pos + 2 < len(full_gt_tokens):
                context.extend([full_gt_tokens[pos],
                                 full_gt_tokens[pos+1],
                                 full_gt_tokens[pos+2]])
                pos += 3

        else:
            # Control/non-score token: append GT directly
            context.append(full_gt_tokens[pos])
            pos += 1

        # ---- Slide if context is at capacity ----
        if len(context) >= CONTEXT_SIZE:
            alt  = context[score_start_idx:]
            # Drop the older half, aligned to 6-token (score+ctrl pair) boundaries
            half = (len(alt) // 2) // 6 * 6
            if half > 0:
                context = header + alt[half:]
                stats['num_slides'] += 1

    return pred_score_triplets, stats


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
    Convert score triplets directly to single-part MusicXML using raw ElementTree.
    
    Bypasses music21's notation pipeline entirely (avoids makeBeams/makeNotation
    crashes) and guarantees a single part/voice for MUSTER's HMM converter.
    
    Time resolution: 10ms bins (TIME_RESOLUTION=100 bins/sec).
    We use 120 BPM so 1 quarter = 0.5s = 50 bins.  divisions=50 means each
    MusicXML <duration> unit is 1 bin = 10ms.
    
    Args:
        triplets: list of [time_token, dur_token, pitch_token] (with vocab offsets)
        xml_path: output MusicXML path
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from xml.etree.ElementTree import Element, SubElement, ElementTree, indent
        
        BINS_PER_SECOND  = TIME_RESOLUTION   # 100
        BPM              = 120
        BINS_PER_QUARTER = BINS_PER_SECOND * 60 // BPM  # 50
        DIVISIONS        = BINS_PER_QUARTER              # 50 units per quarter note
        BINS_PER_MEASURE = BINS_PER_QUARTER * 4          # 200 bins per 4/4 bar
        
        # Decode tokens
        notes = []
        for t in triplets:
            onset = t[0] - TIME_OFFSET
            dur   = t[1] - DUR_OFFSET
            pitch = t[2] - NOTE_OFFSET
            if pitch < 0 or pitch > 127:
                continue
            if dur <= 0:
                dur = 1
            notes.append((onset, dur, pitch))
        
        if not notes:
            return False
        
        notes.sort(key=lambda x: x[0])
        total_bins = max(onset + dur for onset, dur, _ in notes)
        num_measures = max(1, (total_bins + BINS_PER_MEASURE - 1) // BINS_PER_MEASURE)
        
        # Pitch helpers
        MIDI_TO_STEP  = ['C','D','E','F','G','A','B']
        MIDI_TO_ALTER = [0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0]   # sharps
        SEMITONES_IN_OCTAVE = 12

        def midi_to_pitch_elements(parent, midi):
            pitch_el  = SubElement(parent, 'pitch')
            pc        = midi % SEMITONES_IN_OCTAVE
            octave    = midi // SEMITONES_IN_OCTAVE - 1
            # Note name table (with sharps)
            names  = ['C','C','D','D','E','F','F','G','G','A','A','B']
            alters = [ 0,  1,  0,  1,  0,  0,  1,  0,  1,  0,  1,  0]
            SubElement(pitch_el, 'step').text    = names[pc]
            if alters[pc]:
                SubElement(pitch_el, 'alter').text = '1'
            SubElement(pitch_el, 'octave').text  = str(octave)
        
        # Duration-to-type mapping (in bins at DIVISIONS=50 per quarter)
        def dur_to_type(dur_bins):
            """Return closest MusicXML note type (no dots)."""
            quarter = DIVISIONS
            mapping = [
                (quarter * 8,  'breve'),
                (quarter * 4,  'whole'),
                (quarter * 2,  'half'),
                (quarter,      'quarter'),
                (quarter // 2, 'eighth'),
                (quarter // 4, '16th'),
                (quarter // 8, '32nd'),
            ]
            best_type, best_dur = 'quarter', quarter
            best_dist = abs(dur_bins - quarter)
            for d, t in mapping:
                if d > 0 and abs(dur_bins - d) < best_dist:
                    best_dist = abs(dur_bins - d)
                    best_type = t
                    best_dur  = d
            # Clamp duration to match type (MUSTER needs consistent dur/type)
            return best_type, best_dur
        
        # Build XML
        root = Element('score-partwise', version='3.0')
        
        # Part list
        part_list = SubElement(root, 'part-list')
        sp = SubElement(part_list, 'score-part', id='P1')
        SubElement(sp, 'part-name').text = 'Piano'
        
        # Part
        part_el = SubElement(root, 'part', id='P1')
        
        # Group notes by measure
        for m_idx in range(num_measures):
            m_start = m_idx * BINS_PER_MEASURE
            m_end   = m_start + BINS_PER_MEASURE
            
            measure_el = SubElement(part_el, 'measure', number=str(m_idx + 1))
            
            # Attributes on first measure
            if m_idx == 0:
                attrs = SubElement(measure_el, 'attributes')
                SubElement(attrs, 'divisions').text = str(DIVISIONS)
                key_el = SubElement(attrs, 'key')
                SubElement(key_el, 'fifths').text = '0'
                time_el = SubElement(attrs, 'time')
                SubElement(time_el, 'beats').text = '4'
                SubElement(time_el, 'beat-type').text = '4'
                clef_el = SubElement(attrs, 'clef')
                SubElement(clef_el, 'sign').text = 'G'
                SubElement(clef_el, 'line').text  = '2'
            
            # Collect notes in this measure
            m_notes = [(o, d, p) for o, d, p in notes if o >= m_start and o < m_end]
            
            # Emit notes sorted by onset; use <chord> for simultaneous notes
            prev_onset = None
            for onset, dur, pitch in m_notes:
                note_el = SubElement(measure_el, 'note')
                # Chord tag if same onset as previous
                if prev_onset is not None and onset == prev_onset:
                    SubElement(note_el, 'chord')
                midi_to_pitch_elements(note_el, pitch)
                
                note_type, clamped_dur = dur_to_type(dur)
                SubElement(note_el, 'duration').text = str(clamped_dur)
                SubElement(note_el, 'type').text = note_type
                prev_onset = onset
            
            # If no notes, emit a whole rest so the measure is not empty
            if not m_notes:
                rest_el = SubElement(measure_el, 'note')
                SubElement(rest_el, 'rest')
                SubElement(rest_el, 'duration').text = str(DIVISIONS * 4)
                SubElement(rest_el, 'type').text = 'whole'
        
        # Write to file
        tree = ElementTree(root)
        try:
            indent(tree, space='  ')   # Python 3.9+
        except TypeError:
            pass   # older Python — no pretty-printing, still valid XML
        
        header = (b'<?xml version="1.0" encoding="UTF-8"?>\n'
                  b'<!DOCTYPE score-partwise PUBLIC '
                  b'"-//Recordare//DTD MusicXML 3.0 Partwise//EN" '
                  b'"http://www.musicxml.org/dtds/partwise.dtd">\n')
        with open(xml_path, 'wb') as f:
            f.write(header)
            tree.write(f, encoding='utf-8', xml_declaration=False)
        
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


def evaluate_checkpoint_muster(checkpoint_path, test_lines, original_indices, output_dir,
                               forced=False, forced_max_attempts=1000, beam_size=1,
                               temperature=0.0):
    """
    Evaluate a checkpoint using MUSTER metrics.

    Args:
        forced: retry-triplet-until-pitch-matches oracle decoding.
        forced_max_attempts: max triplet draws per position before injecting GT pitch.
        beam_size: beam width for beam-search decoding (1 = greedy).
        temperature: sampling temperature (0 = greedy/argmax).

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
    if forced:
        aggregate_metrics['forced_total_triplet_attempts'] = []
        aggregate_metrics['forced_positions_forced']       = []
    if beam_size > 1:
        aggregate_metrics['beam_log_prob'] = []
    
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
            predicted_tokens, gen_stats = autoregressive_generate_score(
                model, tokens, score_start_idx, device,
                forced=forced, forced_max_attempts=forced_max_attempts,
                beam_size=beam_size, temperature=temperature
            )
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
            if forced:
                metrics['forced_total_triplet_attempts'] = gen_stats['total_triplet_attempts']
                metrics['forced_positions_forced']       = gen_stats['positions_forced']
            if beam_size > 1:
                metrics['beam_log_prob'] = gen_stats.get('beam_log_prob', 0.0)
            per_sequence_metrics.append(metrics)
            
            # Update aggregates
            for key in aggregate_metrics:
                if key in metrics:
                    aggregate_metrics[key].append(metrics[key])
            if forced:
                aggregate_metrics['forced_total_triplet_attempts'].append(gen_stats['total_triplet_attempts'])
                aggregate_metrics['forced_positions_forced'].append(gen_stats['positions_forced'])
            if beam_size > 1:
                aggregate_metrics['beam_log_prob'].append(gen_stats.get('beam_log_prob', 0.0))
            
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
            final_aggregate[f'{key}_mean'] = float(np.mean(values))
            final_aggregate[f'{key}_std']  = float(np.std(values))
            final_aggregate[f'{key}_min']  = float(np.min(values))
            final_aggregate[f'{key}_max']  = float(np.max(values))
    
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
    
    # Forced decoding stats
    if 'forced_total_triplet_attempts_mean' in stats:
        print()
        print("  Forced Decoding Stats:")
        print("  " + "-"*50)
        print(f"  {'Avg triplet attempts/seq':<30} {stats['forced_total_triplet_attempts_mean']:>8.1f}")
        print(f"  {'Avg positions GT-injected':<30} {stats['forced_positions_forced_mean']:>8.1f}")

    # Beam search stats
    if 'beam_log_prob_mean' in stats:
        print()
        print("  Beam Search Stats:")
        print("  " + "-"*50)
        print(f"  {'Avg best-beam log-prob':<30} {stats['beam_log_prob_mean']:>8.2f}")
    print()


def main(checkpoint=None, test_file=None, num_examples=None,
         forced=False, forced_max_attempts=1000, beam_size=1, temperature=0.0):
    if checkpoint is None:
        checkpoint = DEFAULT_CHECKPOINT
    if test_file is None:
        test_file = TEST_FILE
    if num_examples is None:
        num_examples = NUM_EXAMPLES
    
    print("="*80)
    temp_tag = f" [TEMP={temperature}]" if temperature > 0 else ""
    mode_tag = " [FORCED/ORACLE]" if forced else (f" [BEAM={beam_size}]" if beam_size > 1 else "")
    print("MUSTER EVALUATION" + mode_tag + temp_tag)
    print("="*80)
    print(f"Checkpoint: {checkpoint}")
    print(f"Test file: {test_file}")
    if forced:
        print(f"Forced decoding: ON (max {forced_max_attempts} triplet attempts per position)")
    if beam_size > 1:
        print(f"Beam search: ON (beam_size={beam_size})")
    if temperature > 0:
        print(f"Temperature: {temperature}")
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
    
    # Create output directory (separate subdir per mode)
    temp_suffix = f'_temp{temperature}' if temperature > 0 else ''
    if forced:
        subdir = f'{checkpoint}_forced{temp_suffix}'
    elif beam_size > 1:
        subdir = f'{checkpoint}_beam{beam_size}{temp_suffix}'
    else:
        subdir = f'{checkpoint}{temp_suffix}'
    output_dir = Path(OUTPUT_BASE) / subdir
    os.makedirs(output_dir, exist_ok=True)
    
    # Save sampled indices
    with open(output_dir / 'sampled_indices.json', 'w') as f:
        json.dump({
            'seed': RANDOM_SEED, 
            'num_samples': len(sampled_indices), 
            'indices': sampled_indices
        }, f, indent=2)
    
    # Evaluate
    stats = evaluate_checkpoint_muster(
        checkpoint, test_lines, sampled_indices, str(output_dir),
        forced=forced, forced_max_attempts=forced_max_attempts,
        beam_size=beam_size, temperature=temperature
    )
    
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
    parser.add_argument('--forced', action='store_true',
                        help='Forced/oracle decoding: sample from model until GT token is drawn, '
                             'guaranteeing the model stays on the correct path')
    parser.add_argument('--forced-max-attempts', type=int, default=1000,
                        help='Max triplet draws per position before injecting GT pitch (default: 1000)')
    parser.add_argument('--beam', type=int, default=1, metavar='BEAM_SIZE',
                        help='Beam size for beam-search decoding (default: 1 = greedy). '
                             'Mutually exclusive with --forced.')
    parser.add_argument('--temperature', type=float, default=0.0,
                        help='Sampling temperature (default: 0 = greedy argmax). '
                             'Values < 1 sharpen the distribution; > 1 flatten it.')
    args = parser.parse_args()

    if args.forced and args.beam > 1:
        print('ERROR: --forced and --beam are mutually exclusive.')
        sys.exit(1)

    main(checkpoint=args.checkpoint, test_file=args.test_file, num_examples=args.num_examples,
         forced=args.forced, forced_max_attempts=args.forced_max_attempts,
         beam_size=args.beam, temperature=args.temperature)
