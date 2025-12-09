"""
Examine the actual interleaved token sequence produced by tokenization
to ensure it matches expectations.
"""

import os
import pandas as pd
import numpy as np
from anticipation.config import *
from anticipation.vocab import *
from alignment import align_tokens2, load_annotation_file

def examine_interleaved_sequence():
    """Generate and examine the actual interleaved sequence"""
    
    # Load first piece
    asap_annotations = pd.read_csv(os.path.join('asap-dataset-master', "metadata.csv"))
    row = asap_annotations.iloc[0]
    piece_name = f"{row['composer']} - {row['title']}"
    
    midi_score_filename = row["midi_score"]
    midi_performance_filename = row["midi_performance"]
    
    file1 = os.path.join('asap-dataset-master', midi_performance_filename)
    file2 = os.path.join('asap-dataset-master', midi_score_filename)
    file3 = file1.replace(".mid", "_annotations.txt")
    file4 = file2.replace(".mid", "_annotations.txt")
    
    print("="*80)
    print(f"INTERLEAVED SEQUENCE EXAMINATION: {piece_name}")
    print("="*80)
    
    # Run alignment
    print("\nRunning alignment...")
    matched_tuples = align_tokens2(file1, file2, file3, file4, skip_Nones=True)
    print(f"Matched {len(matched_tuples)} notes")
    
    # Load beat annotations for normalization
    score_annotations = load_annotation_file(file4)
    score_beat_times = [anno[0] for anno in score_annotations]
    
    # Pre-normalize ALL score triplets
    print("\nNormalizing score times and durations...")
    normalized_matched_tuples = []
    for match in matched_tuples:
        perf_triplet = match[0]
        score_triplet = match[2]
        
        if score_triplet[0] is not None:
            # Decode
            original_time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
            original_duration_sec = (score_triplet[1] - DUR_OFFSET) / TIME_RESOLUTION
            pitch = score_triplet[2]
            
            # Normalize time using beat mapping
            normalized_time_sec = 0.0
            time_scale_factor = 1.0
            
            if score_beat_times and len(score_beat_times) >= 2:
                if original_time_sec < score_beat_times[0]:
                    beat_duration = score_beat_times[1] - score_beat_times[0]
                    if beat_duration > 0:
                        progress = (original_time_sec - score_beat_times[0]) / beat_duration
                        time_scale_factor = 0.5 / beat_duration
                    else:
                        progress = 0
                        time_scale_factor = 1.0
                    normalized_time_sec = 0.0 + progress * 0.5
                else:
                    found = False
                    for i in range(len(score_beat_times) - 1):
                        if score_beat_times[i] <= original_time_sec <= score_beat_times[i + 1]:
                            beat_duration = score_beat_times[i + 1] - score_beat_times[i]
                            if beat_duration > 0:
                                progress = (original_time_sec - score_beat_times[i]) / beat_duration
                                time_scale_factor = 0.5 / beat_duration
                            else:
                                progress = 0
                                time_scale_factor = 1.0
                            normalized_time_sec = i * 0.5 + progress * 0.5
                            found = True
                            break
                    
                    if not found:
                        last_beat_idx = len(score_beat_times) - 1
                        last_beat_duration = score_beat_times[-1] - score_beat_times[-2]
                        if last_beat_duration > 0:
                            progress = (original_time_sec - score_beat_times[-1]) / last_beat_duration
                            time_scale_factor = 0.5 / last_beat_duration
                        else:
                            progress = 0
                            time_scale_factor = 1.0
                        normalized_time_sec = last_beat_idx * 0.5 + progress * 0.5
            else:
                normalized_time_sec = original_time_sec
                time_scale_factor = 1.0
            
            # Scale duration
            normalized_duration_sec = original_duration_sec * time_scale_factor
            
            # Convert back to quantized units
            normalized_time_units = round(normalized_time_sec * TIME_RESOLUTION)
            normalized_duration_units = round(normalized_duration_sec * TIME_RESOLUTION)
            normalized_score = [
                normalized_time_units + TIME_OFFSET,
                normalized_duration_units + DUR_OFFSET,
                pitch
            ]
        else:
            normalized_score = score_triplet
        
        normalized_matched_tuples.append([perf_triplet, match[1], normalized_score, match[3]])
    
    # Build interleaved sequence (from a starting point)
    print("\nBuilding interleaved sequence starting from index 0...")
    prefix_controls = 33
    k = min(prefix_controls, len(normalized_matched_tuples))
    
    subset = normalized_matched_tuples[0:]
    
    # Extract and normalize performance triplets (normalize to start at 0)
    perf_triplets = [[match[0][0] - CONTROL_OFFSET, match[0][1] - CONTROL_OFFSET, match[0][2] - CONTROL_OFFSET] 
                     for match in subset]
    if perf_triplets:
        perf_min_time = min(triplet[0] for triplet in perf_triplets)
        perf_triplets = [[triplet[0] - perf_min_time, triplet[1], triplet[2]] for triplet in perf_triplets]
    
    # Extract normalized score triplets
    score_triplets = [match[2] for match in subset]
    
    # Build interleaved stream
    interleaved_tokens = []
    
    # Add ANTICIPATE token
    interleaved_tokens.append(ANTICIPATE)
    
    # Add 3 SEP tokens
    interleaved_tokens.extend([SEPARATOR, SEPARATOR, SEPARATOR])
    
    # Prefix: k control+rest pairs
    for i in range(k):
        perf_triplet = perf_triplets[i]
        # Add control triplet (re-add CONTROL_OFFSET to all 3 elements)
        interleaved_tokens.extend([
            perf_triplet[0] + CONTROL_OFFSET,
            perf_triplet[1] + CONTROL_OFFSET,
            perf_triplet[2] + CONTROL_OFFSET
        ])
        # Add rest triplet
        cc_time = perf_triplet[0]
        interleaved_tokens.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])
    
    # Main body: alternate score/control using notes [0:] for scores and notes [k:] for controls
    for i in range(len(subset)):
        score_triplet = score_triplets[i]
        # Add score triplet if it exists
        if score_triplet[0] is not None:
            interleaved_tokens.extend(score_triplet)
        # Add next control if available
        ii = i + k
        if ii < len(subset):
            perf_triplet = perf_triplets[ii]
            interleaved_tokens.extend([
                perf_triplet[0] + CONTROL_OFFSET,
                perf_triplet[1] + CONTROL_OFFSET,
                perf_triplet[2] + CONTROL_OFFSET
            ])
    
    print(f"\nTotal interleaved tokens: {len(interleaved_tokens)}")
    print(f"Expected structure:")
    print(f"  1 ANTICIPATE token")
    print(f"  3 SEPARATOR tokens")
    print(f"  {k} control+rest pairs = {k*6} tokens")
    print(f"  {len(subset)} score triplets + {len(subset)-k} additional controls (alternating)")
    print(f"  Total should be: 1 + 3 + {k*6} + roughly {len(subset)*3 + (len(subset)-k)*3} = {1 + 3 + k*6 + len(subset)*3 + (len(subset)-k)*3}")
    
    # Examine the sequence in detail
    print("\n" + "="*80)
    print("SEQUENCE STRUCTURE ANALYSIS")
    print("="*80)
    
    print(f"\n[Position 0] ANTICIPATE token: {interleaved_tokens[0]}")
    print(f"  Expected: {ANTICIPATE}")
    print(f"  Match: {interleaved_tokens[0] == ANTICIPATE}")
    
    print(f"\n[Positions 1-3] Three SEPARATOR tokens:")
    print(f"  Token 1: {interleaved_tokens[1]} (expected {SEPARATOR}, match: {interleaved_tokens[1] == SEPARATOR})")
    print(f"  Token 2: {interleaved_tokens[2]} (expected {SEPARATOR}, match: {interleaved_tokens[2] == SEPARATOR})")
    print(f"  Token 3: {interleaved_tokens[3]} (expected {SEPARATOR}, match: {interleaved_tokens[3] == SEPARATOR})")
    
    print(f"\n[Positions 4-{3+k*6}] First {k} control+rest pairs:")
    print(f"{'Idx':<5} {'Pos':<8} {'Time':<12} {'Duration':<12} {'Pitch':<12} {'Time(s)':<12} {'Dur(s)':<12}")
    print("-" * 85)
    for i in range(min(10, k)):
        pos = 4 + i*6  # After ANTICIPATE + SEP SEP SEP
        time_token = interleaved_tokens[pos]
        dur_token = interleaved_tokens[pos+1]
        pitch_token = interleaved_tokens[pos+2]
        
        # Decode (control triplet has CONTROL_OFFSET on all 3 elements)
        time_sec = (time_token - CONTROL_OFFSET) / TIME_RESOLUTION
        dur_sec = (dur_token - CONTROL_OFFSET - DUR_OFFSET) / TIME_RESOLUTION
        
        print(f"{i:<5} {pos:<8} {time_token:<12} {dur_token:<12} {pitch_token:<12} {time_sec:<12.3f} {dur_sec:<12.3f}")
    
    score_start = 4 + k*6  # After ANTICIPATE + SEP SEP SEP + k control+rest pairs
    print(f"\n[Positions {score_start}+] Score (anticipated) triplets (alternating with controls):")
    print(f"{'Idx':<5} {'Pos':<8} {'Time':<12} {'Duration':<12} {'Pitch':<12} {'Time(s)':<12} {'Dur(s)':<12}")
    print("-" * 85)
    for i in range(min(10, len(subset))):
        pos = score_start + i*6  # Each iteration: score triplet (3) + control triplet (3)
        if pos < len(interleaved_tokens):
            time_token = interleaved_tokens[pos]
            dur_token = interleaved_tokens[pos+1]
            pitch_token = interleaved_tokens[pos+2]
            
            # Decode
            time_sec = (time_token - TIME_OFFSET) / TIME_RESOLUTION
            dur_sec = (dur_token - DUR_OFFSET) / TIME_RESOLUTION
            
            print(f"{i:<5} {pos:<8} {time_token:<12} {dur_token:<12} {pitch_token:<12} {time_sec:<12.3f} {dur_sec:<12.3f}")
    
    # Check properties
    print("\n" + "="*80)
    print("PROPERTY CHECKS")
    print("="*80)
    
    issues = []
    
    # 1. Check starting pattern
    if interleaved_tokens[0] != ANTICIPATE:
        issues.append(f"First token should be ANTICIPATE ({ANTICIPATE}), got {interleaved_tokens[0]}")
    
    # SEP SEP SEP should be at positions 1-3 (right after ANTICIPATE)
    if interleaved_tokens[1:4] != [SEPARATOR, SEPARATOR, SEPARATOR]:
        issues.append(f"Tokens at positions 1-3 should be SEP SEP SEP, got {interleaved_tokens[1:4]}")
    
    # 2. Check that control times start at 0 (first control+rest pair starts at position 4)
    first_control_pos = 4
    first_control_time = interleaved_tokens[first_control_pos]
    first_control_time_sec = (first_control_time - CONTROL_OFFSET) / TIME_RESOLUTION
    if abs(first_control_time_sec) > 0.001:
        issues.append(f"First control time should be ~0, got {first_control_time_sec:.3f}s")
    
    # 3. Check that control times are monotonic (in the prefix control+rest pairs)
    for i in range(k-1):
        pos = 4 + i*6  # Control triplets in prefix, each control+rest pair is 6 tokens
        time1 = (interleaved_tokens[pos] - CONTROL_OFFSET) / TIME_RESOLUTION
        time2 = (interleaved_tokens[pos+6] - CONTROL_OFFSET) / TIME_RESOLUTION
        if time2 < time1:
            issues.append(f"Control times not monotonic: position {pos} ({time1:.3f}s) > position {pos+6} ({time2:.3f}s)")
    
    # 4. Check that score times start and are spaced by ~0.5 sec beats
    # Score triplets start after: ANTICIPATE + SEP SEP SEP + k control+rest pairs
    score_start_pos = 4 + k*6  
    first_score_time = interleaved_tokens[score_start_pos]
    first_score_time_sec = (first_score_time - TIME_OFFSET) / TIME_RESOLUTION
    
    # Find first beat-aligned score notes
    beat_notes = []
    for i in range(min(20, len(subset))):
        # Alternating: score (3 tokens) + control (3 tokens)
        pos = score_start_pos + i*6
        if pos >= len(interleaved_tokens):
            break
        time_sec = (interleaved_tokens[pos] - TIME_OFFSET) / TIME_RESOLUTION
        # Check if close to a multiple of 0.5
        nearest_beat = round(time_sec / 0.5)
        if abs(time_sec - nearest_beat * 0.5) < 0.01:
            beat_notes.append((i, time_sec, nearest_beat))
    
    if beat_notes:
        print(f"\nBeat-aligned score notes (first 10):")
        print(f"{'Note':<8} {'Time(s)':<12} {'Beat#':<8}")
        for note_idx, time_sec, beat_num in beat_notes[:10]:
            print(f"{note_idx:<8} {time_sec:<12.4f} {beat_num:<8}")
    
    # Print summary header
    print(f"\nValidation Summary:")
    
    # Print summary header
    print(f"\nValidation Summary:")
    
    # 5. Check score times are monotonic
    for i in range(min(20, len(subset)-1)):
        pos = score_start_pos + i*6
        if pos+6 >= len(interleaved_tokens):
            break
        time1 = (interleaved_tokens[pos] - TIME_OFFSET) / TIME_RESOLUTION
        time2 = (interleaved_tokens[pos+6] - TIME_OFFSET) / TIME_RESOLUTION
        if time2 < time1:
            issues.append(f"Score times not monotonic: position {pos} ({time1:.3f}s) > position {pos+6} ({time2:.3f}s)")
    
    # 6. Check control triplet format (positions 4 to 4+k*6, every 6 tokens)
    for i in range(k):
        ctrl_pos = 4 + i*6
        ctrl_time = interleaved_tokens[ctrl_pos]
        ctrl_dur = interleaved_tokens[ctrl_pos+1]
        ctrl_pitch = interleaved_tokens[ctrl_pos+2]
        
        # All 3 should be in CONTROL range (CONTROL_OFFSET to SPECIAL_OFFSET)
        if not (CONTROL_OFFSET <= ctrl_time < SPECIAL_OFFSET and
                CONTROL_OFFSET <= ctrl_dur < SPECIAL_OFFSET and
                CONTROL_OFFSET <= ctrl_pitch < SPECIAL_OFFSET):
            issues.append(f"Control triplet at pos {ctrl_pos} not in CONTROL range: {ctrl_time}, {ctrl_dur}, {ctrl_pitch}")
    
    # 7. Check rest triplet format (positions 7, 13, 19... in prefix)
    for i in range(k):
        rest_pos = 4 + i*6 + 3  # After each control triplet
        rest_time = interleaved_tokens[rest_pos]
        rest_dur = interleaved_tokens[rest_pos+1]
        rest_pitch = interleaved_tokens[rest_pos+2]
        
        # Should be: TIME_OFFSET + time, DUR_OFFSET + 0, REST
        if rest_dur != DUR_OFFSET:
            issues.append(f"Rest at pos {rest_pos} should have dur={DUR_OFFSET}, got {rest_dur}")
        if rest_pitch != REST:
            issues.append(f"Rest at pos {rest_pos} should be REST ({REST}), got {rest_pitch}")
    
    # 8. Check score triplets in main body (alternating with controls)
    for i in range(min(20, len(subset))):
        score_pos = score_start_pos + i*6
        if score_pos+2 >= len(interleaved_tokens):
            break
        
        score_time = interleaved_tokens[score_pos]
        score_dur = interleaved_tokens[score_pos+1]
        score_pitch = interleaved_tokens[score_pos+2]
        
        # All 3 should be < CONTROL_OFFSET
        if not (score_time < CONTROL_OFFSET and
                score_dur < CONTROL_OFFSET and
                score_pitch < CONTROL_OFFSET):
            issues.append(f"Score triplet at pos {score_pos} should be < CONTROL_OFFSET: {score_time}, {score_dur}, {score_pitch}")
        
        # Check ranges
        if not (0 <= score_time < TIME_OFFSET + MAX_TIME):
            issues.append(f"Score time at pos {score_pos} out of range: {score_time}")
        if not (DUR_OFFSET <= score_dur < DUR_OFFSET + 1000):
            issues.append(f"Score duration at pos {score_pos} out of range: {score_dur}")
        if not (NOTE_OFFSET <= score_pitch <= REST):
            issues.append(f"Score pitch at pos {score_pos} out of range: {score_pitch}")
    
    # 9. Check control triplets in main body (after each score, positions 205, 211, 217...)
    for i in range(min(20, len(subset)-k)):
        ctrl_pos = score_start_pos + i*6 + 3  # After each score triplet
        if ctrl_pos+2 >= len(interleaved_tokens):
            break
        
        ctrl_time = interleaved_tokens[ctrl_pos]
        ctrl_dur = interleaved_tokens[ctrl_pos+1]
        ctrl_pitch = interleaved_tokens[ctrl_pos+2]
        
        # All 3 should be in CONTROL range
        if not (CONTROL_OFFSET <= ctrl_time < SPECIAL_OFFSET and
                CONTROL_OFFSET <= ctrl_dur < SPECIAL_OFFSET and
                CONTROL_OFFSET <= ctrl_pitch < SPECIAL_OFFSET):
            issues.append(f"Control triplet at pos {ctrl_pos} not in CONTROL range: {ctrl_time}, {ctrl_dur}, {ctrl_pitch}")
    
    if issues:
        print(f"\nFOUND {len(issues)} ISSUES:")
        for issue in issues[:20]:  # Show first 20
            print(f"  - {issue}")
        if len(issues) > 20:
            print(f"  ... and {len(issues) - 20} more issues")
    else:
        print(f"\nALL CHECKS PASSED!")
        print("Format verification:")
        print(f"  - Position 0: ANTICIPATE ({ANTICIPATE})")
        print(f"  - Positions 1-3: SEP SEP SEP ({SEPARATOR})")
        print(f"  - Positions 4-{4+k*6-1}: {k} control+rest pairs (each 6 tokens)")
        print(f"  - Position {score_start_pos}+: Alternating score/control triplets")
        print("\nToken validation:")
        print("  - Control triplets: all 3 elements in CONTROL range [27513, 55025)")
        print("  - Rest triplets: duration=10000, pitch=27512 (REST)")
        print("  - Score triplets: all 3 elements < 27513")
        print("\nTiming validation:")
        print("  - Control times start at 0 and monotonic")
        print("  - Score times monotonic")
        print(f"  - Found {len(beat_notes)} beat-aligned notes at 0.5 sec intervals")
    
    print("\n" + "="*80)
    print("EXAMINATION COMPLETE")
    print("="*80)
    
    return not bool(issues)

if __name__ == "__main__":
    success = examine_interleaved_sequence()
    exit(0 if success else 1)
