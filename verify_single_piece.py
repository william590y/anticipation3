"""
Comprehensive single-piece verification:
1) Format is followed correctly (ANTICIPATE + SEP SEP SEP + control+rest pairs + alternating)
2) Pitch accuracy is 100% in the interleaved sequence
3) 1 beat = 0.5s and both score/performance start at time zero
"""

import os
import pandas as pd
import numpy as np
from anticipation.config import *
from anticipation.vocab import *
from anticipation.convert import midi_to_events
from alignment import align_tokens2, load_annotation_file

def verify_single_piece():
    """Run full pipeline on one piece and verify all requirements"""
    
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
    print(f"SINGLE PIECE VERIFICATION: {piece_name}")
    print("="*80)
    print()
    
    # Load annotations
    score_annotations = load_annotation_file(file4)
    perf_annotations = load_annotation_file(file3)
    score_beat_times = np.array([anno[0] for anno in score_annotations])
    perf_beat_times = np.array([anno[0] for anno in perf_annotations])
    
    print(f"Score beats: {len(score_beat_times)}")
    print(f"Performance beats: {len(perf_beat_times)}")
    print(f"Score first beat: {score_beat_times[0]:.3f}s")
    print(f"Performance first beat: {perf_beat_times[0]:.3f}s")
    print()
    
    # Run alignment
    print("Running alignment...")
    matched_tuples = align_tokens2(file1, file2, file3, file4, skip_Nones=True)
    print(f"Matched {len(matched_tuples)} notes")
    print()
    
    # Extract performance and score triplets
    perf_triplets = []
    score_triplets = []
    
    for match in matched_tuples:
        perf_triplet = match[0]
        score_triplet = match[2]
        
        if score_triplet[0] is not None:
            perf_triplets.append(perf_triplet)
            score_triplets.append(score_triplet)
    
    print(f"Valid triplets: {len(perf_triplets)}")
    print()
    
    # =========================================================================
    # REQUIREMENT 3: Normalize to 0.5s per beat and start at time zero
    # =========================================================================
    print("="*80)
    print("REQUIREMENT 3: Normalize to 0.5s per beat, start at time zero")
    print("="*80)
    
    # Normalize performance triplets
    norm_perf_triplets = []
    for perf_triplet in perf_triplets:
        # Decode (remove CONTROL_OFFSET from ALL 3 elements)
        time_sec = (perf_triplet[0] - CONTROL_OFFSET - TIME_OFFSET) / TIME_RESOLUTION
        dur_sec = (perf_triplet[1] - CONTROL_OFFSET - DUR_OFFSET) / TIME_RESOLUTION
        pitch = perf_triplet[2] - CONTROL_OFFSET - NOTE_OFFSET
        
        # Normalize time (find which beat interval and scale)
        norm_time_sec = 0.0
        scale_factor = 1.0
        
        if len(perf_beat_times) >= 2:
            if time_sec < perf_beat_times[0]:
                # Before first beat
                beat_duration = perf_beat_times[1] - perf_beat_times[0]
                if beat_duration > 0:
                    progress = (time_sec - perf_beat_times[0]) / beat_duration
                    scale_factor = 0.5 / beat_duration
                else:
                    progress = 0
                norm_time_sec = 0.0 + progress * 0.5
            else:
                # Find which beats this falls between
                found = False
                for i in range(len(perf_beat_times) - 1):
                    if perf_beat_times[i] <= time_sec <= perf_beat_times[i + 1]:
                        beat_duration = perf_beat_times[i + 1] - perf_beat_times[i]
                        if beat_duration > 0:
                            progress = (time_sec - perf_beat_times[i]) / beat_duration
                            scale_factor = 0.5 / beat_duration
                        else:
                            progress = 0
                        norm_time_sec = i * 0.5 + progress * 0.5
                        found = True
                        break
                
                if not found:
                    # After last beat
                    last_beat_idx = len(perf_beat_times) - 1
                    last_beat_duration = perf_beat_times[-1] - perf_beat_times[-2] if len(perf_beat_times) >= 2 else 1.0
                    if last_beat_duration > 0:
                        progress = (time_sec - perf_beat_times[-1]) / last_beat_duration
                        scale_factor = 0.5 / last_beat_duration
                    else:
                        progress = 0
                    norm_time_sec = last_beat_idx * 0.5 + progress * 0.5
        
        norm_dur_sec = dur_sec * scale_factor
        
        # Convert back to tokens (add offsets, no CONTROL_OFFSET yet)
        norm_time = round(norm_time_sec * TIME_RESOLUTION) + TIME_OFFSET
        norm_dur = round(norm_dur_sec * TIME_RESOLUTION) + DUR_OFFSET
        norm_pitch = pitch + NOTE_OFFSET
        
        norm_perf_triplets.append([norm_time, norm_dur, norm_pitch])
    
    # Normalize score triplets
    norm_score_triplets = []
    for score_triplet in score_triplets:
        # Decode
        time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
        dur_sec = (score_triplet[1] - DUR_OFFSET) / TIME_RESOLUTION
        pitch = score_triplet[2] - NOTE_OFFSET
        
        # Normalize time (find which beat interval and scale)
        norm_time_sec = 0.0
        scale_factor = 1.0
        
        if len(score_beat_times) >= 2:
            if time_sec < score_beat_times[0]:
                # Before first beat
                beat_duration = score_beat_times[1] - score_beat_times[0]
                if beat_duration > 0:
                    progress = (time_sec - score_beat_times[0]) / beat_duration
                    scale_factor = 0.5 / beat_duration
                else:
                    progress = 0
                norm_time_sec = 0.0 + progress * 0.5
            else:
                # Find which beats this falls between
                found = False
                for i in range(len(score_beat_times) - 1):
                    if score_beat_times[i] <= time_sec <= score_beat_times[i + 1]:
                        beat_duration = score_beat_times[i + 1] - score_beat_times[i]
                        if beat_duration > 0:
                            progress = (time_sec - score_beat_times[i]) / beat_duration
                            scale_factor = 0.5 / beat_duration
                        else:
                            progress = 0
                        norm_time_sec = i * 0.5 + progress * 0.5
                        found = True
                        break
                
                if not found:
                    # After last beat
                    last_beat_idx = len(score_beat_times) - 1
                    last_beat_duration = score_beat_times[-1] - score_beat_times[-2] if len(score_beat_times) >= 2 else 1.0
                    if last_beat_duration > 0:
                        progress = (time_sec - score_beat_times[-1]) / last_beat_duration
                        scale_factor = 0.5 / last_beat_duration
                    else:
                        progress = 0
                    norm_time_sec = last_beat_idx * 0.5 + progress * 0.5
        
        norm_dur_sec = dur_sec * scale_factor
        
        # Convert back to tokens
        norm_time = round(norm_time_sec * TIME_RESOLUTION) + TIME_OFFSET
        norm_dur = round(norm_dur_sec * TIME_RESOLUTION) + DUR_OFFSET
        norm_pitch = pitch + NOTE_OFFSET
        
        norm_score_triplets.append([norm_time, norm_dur, norm_pitch])
    
    # Check first times are at zero
    perf_first_time = (norm_perf_triplets[0][0] - TIME_OFFSET) / TIME_RESOLUTION
    score_first_time = (norm_score_triplets[0][0] - TIME_OFFSET) / TIME_RESOLUTION
    
    print(f"Performance first note time: {perf_first_time:.6f}s")
    print(f"Score first note time: {score_first_time:.6f}s")
    
    if abs(perf_first_time) < 0.01 and abs(score_first_time) < 0.01:
        print("✅ PASS: Both start at time zero")
    else:
        print("❌ FAIL: Not starting at time zero")
    
    # Check beat spacing in normalized score
    beat_notes = []
    for i, score_triplet in enumerate(norm_score_triplets[:20]):
        time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
        # Check if close to a beat time (0.0, 0.5, 1.0, 1.5, ...)
        beat_num = round(time_sec / 0.5)
        expected_time = beat_num * 0.5
        if abs(time_sec - expected_time) < 0.02:  # Within 20ms
            beat_notes.append((beat_num, time_sec, expected_time))
    
    print(f"\nFound {len(beat_notes)} beat-aligned notes in first 20 notes:")
    print(f"{'Beat#':<8} {'Actual':>10} {'Expected':>10} {'Error':>10}")
    for beat_num, actual, expected in beat_notes:
        error = actual - expected
        print(f"{beat_num:<8} {actual:>10.4f} {expected:>10.4f} {error:>+10.4f}")
    
    if all(abs(actual - expected) < 0.01 for _, actual, expected in beat_notes):
        print("✅ PASS: Beat spacing is 0.5s")
    else:
        print("❌ FAIL: Beat spacing is not 0.5s")
    
    print()
    
    # =========================================================================
    # REQUIREMENT 2: Pitch accuracy is 100%
    # =========================================================================
    print("="*80)
    print("REQUIREMENT 2: Pitch accuracy is 100% in interleaved sequence")
    print("="*80)
    
    # Extract pitches
    perf_pitches = [(p[2] - NOTE_OFFSET) for p in norm_perf_triplets]
    score_pitches = [(p[2] - NOTE_OFFSET) for p in norm_score_triplets]
    
    # Check they match
    min_len = min(len(perf_pitches), len(score_pitches))
    matches = sum(1 for i in range(min_len) if perf_pitches[i] == score_pitches[i])
    accuracy = 100.0 * matches / min_len if min_len > 0 else 0.0
    
    print(f"Matched notes: {min_len}")
    print(f"Pitch matches: {matches}/{min_len}")
    print(f"Pitch accuracy: {accuracy:.2f}%")
    
    if accuracy == 100.0:
        print("✅ PASS: Pitch accuracy is 100%")
    else:
        print("❌ FAIL: Pitch accuracy is not 100%")
        print("\nFirst 10 mismatches:")
        mismatch_count = 0
        for i in range(min_len):
            if perf_pitches[i] != score_pitches[i]:
                print(f"  Note {i}: perf={perf_pitches[i]}, score={score_pitches[i]}")
                mismatch_count += 1
                if mismatch_count >= 10:
                    break
    
    print()
    
    # =========================================================================
    # REQUIREMENT 1: Format is followed correctly
    # =========================================================================
    print("="*80)
    print("REQUIREMENT 1: Format is followed correctly")
    print("="*80)
    
    # Build interleaved sequence with control+rest pairs
    # Use tokenize-asap-sliding.py logic
    
    # First, add CONTROL_OFFSET to performance triplets
    control_triplets = []
    for perf_triplet in norm_perf_triplets:
        control_triplets.append([
            perf_triplet[0] + CONTROL_OFFSET,
            perf_triplet[1] + CONTROL_OFFSET,
            perf_triplet[2] + CONTROL_OFFSET
        ])
    
    # Build control+rest pairs (k=33 pairs)
    k = 33
    interleaved_tokens = []
    
    for i in range(k):
        # Control triplet
        if i < len(control_triplets):
            interleaved_tokens.extend(control_triplets[i])
        
        # Rest triplet
        if i < len(norm_score_triplets):
            rest_time = norm_score_triplets[i][0]
            interleaved_tokens.extend([rest_time, DUR_OFFSET + 0, REST])
    
    # Add alternating score/control triplets
    M = 341  # Maximum triplets in body (1023 tokens / 3)
    remaining_triplets = M - k * 2  # 275 triplets remaining
    
    for i in range(k, min(len(norm_score_triplets), k + remaining_triplets // 2)):
        # Score triplet
        interleaved_tokens.extend(norm_score_triplets[i])
        
        # Control triplet
        if i < len(control_triplets):
            interleaved_tokens.extend(control_triplets[i])
    
    # Prepend SEP SEP SEP
    interleaved_tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]
    
    # Trim to exactly 1023 tokens
    interleaved_tokens = interleaved_tokens[:1023]
    
    # Add ANTICIPATE at position 0
    sequence = [ANTICIPATE] + interleaved_tokens
    
    print(f"Sequence length: {len(sequence)} (expected 1024)")
    print(f"Position 0: {sequence[0]} (expected {ANTICIPATE} = ANTICIPATE)")
    print(f"Positions 1-3: {sequence[1:4]} (expected [{SEPARATOR}, {SEPARATOR}, {SEPARATOR}] = SEP SEP SEP)")
    print()
    
    # Verify format
    format_ok = True
    
    if len(sequence) != 1024:
        print(f"❌ FAIL: Sequence length is {len(sequence)}, expected 1024")
        format_ok = False
    else:
        print("✅ PASS: Sequence length is 1024")
    
    if sequence[0] != ANTICIPATE:
        print(f"❌ FAIL: Position 0 is {sequence[0]}, expected {ANTICIPATE}")
        format_ok = False
    else:
        print("✅ PASS: Position 0 is ANTICIPATE")
    
    if sequence[1:4] != [SEPARATOR, SEPARATOR, SEPARATOR]:
        print(f"❌ FAIL: Positions 1-3 are {sequence[1:4]}, expected SEP SEP SEP")
        format_ok = False
    else:
        print("✅ PASS: Positions 1-3 are SEP SEP SEP")
    
    # Check control+rest pairs (positions 4-201, 66 triplets = 33 pairs)
    print()
    print("Checking control+rest pairs (positions 4-201)...")
    control_rest_ok = True
    
    for i in range(33):
        base_pos = 4 + i * 6
        
        # Control triplet
        ctrl_time = sequence[base_pos]
        ctrl_dur = sequence[base_pos + 1]
        ctrl_pitch = sequence[base_pos + 2]
        
        if not (ctrl_time >= CONTROL_OFFSET and ctrl_dur >= CONTROL_OFFSET and ctrl_pitch >= CONTROL_OFFSET):
            print(f"❌ FAIL: Control triplet at position {base_pos} not all >= CONTROL_OFFSET")
            print(f"  Values: [{ctrl_time}, {ctrl_dur}, {ctrl_pitch}]")
            control_rest_ok = False
            break
        
        # Rest triplet
        rest_time = sequence[base_pos + 3]
        rest_dur = sequence[base_pos + 4]
        rest_pitch = sequence[base_pos + 5]
        
        if not (rest_time < CONTROL_OFFSET and rest_dur == DUR_OFFSET and rest_pitch == REST):
            print(f"❌ FAIL: Rest triplet at position {base_pos + 3} incorrect")
            print(f"  Values: [{rest_time}, {rest_dur}, {rest_pitch}]")
            print(f"  Expected: [<{CONTROL_OFFSET}, {DUR_OFFSET}, {REST}]")
            control_rest_ok = False
            break
    
    if control_rest_ok:
        print("✅ PASS: All 33 control+rest pairs are correctly formatted")
    
    # Check alternating pattern (positions 202+)
    print()
    print("Checking alternating score/control pattern (positions 202+)...")
    alternating_ok = True
    
    pos = 202
    pair_count = 0
    while pos + 5 < len(sequence):
        # Score triplet
        score_time = sequence[pos]
        score_dur = sequence[pos + 1]
        score_pitch = sequence[pos + 2]
        
        if not (score_time < CONTROL_OFFSET and score_dur < CONTROL_OFFSET and score_pitch < CONTROL_OFFSET):
            print(f"❌ FAIL: Score triplet at position {pos} not all < CONTROL_OFFSET")
            print(f"  Values: [{score_time}, {score_dur}, {score_pitch}]")
            alternating_ok = False
            break
        
        # Control triplet
        ctrl_time = sequence[pos + 3]
        ctrl_dur = sequence[pos + 4]
        ctrl_pitch = sequence[pos + 5]
        
        if not (ctrl_time >= CONTROL_OFFSET and ctrl_dur >= CONTROL_OFFSET and ctrl_pitch >= CONTROL_OFFSET):
            print(f"❌ FAIL: Control triplet at position {pos + 3} not all >= CONTROL_OFFSET")
            print(f"  Values: [{ctrl_time}, {ctrl_dur}, {ctrl_pitch}]")
            alternating_ok = False
            break
        
        pos += 6
        pair_count += 1
        
        if pair_count >= 10:  # Check first 10 pairs
            break
    
    if alternating_ok:
        print(f"✅ PASS: First {pair_count} score/control pairs are correctly alternating")
    
    print()
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    all_passed = (accuracy == 100.0 and 
                  abs(perf_first_time) < 0.01 and 
                  abs(score_first_time) < 0.01 and
                  all(abs(actual - expected) < 0.01 for _, actual, expected in beat_notes) and
                  format_ok and control_rest_ok and alternating_ok)
    
    if all_passed:
        print("✅ ALL REQUIREMENTS PASSED!")
    else:
        print("❌ SOME REQUIREMENTS FAILED")
    
    print()
    return all_passed

if __name__ == "__main__":
    success = verify_single_piece()
    exit(0 if success else 1)
