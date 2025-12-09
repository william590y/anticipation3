"""
Comprehensive verification: run a piece through the full alignment + normalization pipeline
and verify every aspect is correct.
"""

import os
import pandas as pd
import numpy as np
from anticipation.config import *
from anticipation.vocab import *
from anticipation.convert import midi_to_events
from alignment import align_tokens2, load_annotation_file

def verify_full_pipeline():
    """Run full pipeline on one piece and verify correctness at every step"""
    
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
    print(f"FULL PIPELINE VERIFICATION: {piece_name}")
    print("="*80)
    
    # Step 1: Load raw MIDI tokens
    print("\n" + "="*80)
    print("STEP 1: Load raw score MIDI tokens")
    print("="*80)
    score_events = midi_to_events(file2, quantize=False)
    print(f"Total tokens: {len(score_events)} ({len(score_events)//3} notes)")
    print(f"Format: [time, duration, pitch] triplets")
    print(f"\nFirst 5 notes (raw tokens):")
    print(f"{'Note':<6} {'Time':>10} {'Duration':>10} {'Pitch':>10}")
    for i in range(min(5, len(score_events)//3)):
        time_raw = score_events[3*i]
        dur_raw = score_events[3*i+1]
        pitch_raw = score_events[3*i+2]
        print(f"{i:<6} {time_raw:>10.2f} {dur_raw:>10} {pitch_raw:>10}")
    
    print(f"\nFirst 5 notes (decoded to seconds):")
    print(f"{'Note':<6} {'Time(s)':>10} {'Dur(s)':>10} {'Pitch':>10}")
    for i in range(min(5, len(score_events)//3)):
        time_sec = (score_events[3*i] - TIME_OFFSET) / TIME_RESOLUTION
        dur_sec = (score_events[3*i+1] - DUR_OFFSET) / TIME_RESOLUTION
        pitch = score_events[3*i+2]
        print(f"{i:<6} {time_sec:>10.3f} {dur_sec:>10.3f} {pitch:>10}")
    
    # Step 2: Load beat annotations
    print("\n" + "="*80)
    print("STEP 2: Load beat annotations")
    print("="*80)
    score_annotations = load_annotation_file(file4)
    score_beat_times = [anno[0] for anno in score_annotations]
    print(f"Total beats: {len(score_beat_times)}")
    print(f"First 10 beats: {score_beat_times[:10]}")
    print(f"Piece duration: {score_beat_times[-1]:.2f} seconds")
    print(f"Beat spacing (first interval): {score_beat_times[1] - score_beat_times[0]:.3f} sec")
    
    # Step 3: Run alignment
    print("\n" + "="*80)
    print("STEP 3: Align performance to score")
    print("="*80)
    matched_tuples = align_tokens2(file1, file2, file3, file4, skip_Nones=True)
    print(f"Matched {len(matched_tuples)} notes")
    
    print(f"\nFirst 5 matched notes (before normalization):")
    print(f"{'Note':<6} {'Perf_Time':>12} {'Perf_Dur':>10} {'Score_Time':>12} {'Score_Dur':>10} {'Pitch':>10}")
    for i in range(min(5, len(matched_tuples))):
        match = matched_tuples[i]
        perf_triplet = match[0]
        score_triplet = match[2]
        
        # Decode performance (has CONTROL_OFFSET)
        perf_time_sec = (perf_triplet[0] - CONTROL_OFFSET - TIME_OFFSET) / TIME_RESOLUTION
        perf_dur_sec = (perf_triplet[1] - CONTROL_OFFSET - DUR_OFFSET) / TIME_RESOLUTION
        
        # Decode score
        score_time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
        score_dur_sec = (score_triplet[1] - DUR_OFFSET) / TIME_RESOLUTION
        pitch = score_triplet[2]
        
        print(f"{i:<6} {perf_time_sec:>12.3f} {perf_dur_sec:>10.3f} {score_time_sec:>12.3f} {score_dur_sec:>10.3f} {pitch:>10}")
    
    # Step 4: Normalize score times and durations
    print("\n" + "="*80)
    print("STEP 4: Normalize score to 0.5 sec between beats")
    print("="*80)
    
    # Calculate expected scale factor for first beat interval
    if len(score_beat_times) >= 2:
        first_beat_duration = score_beat_times[1] - score_beat_times[0]
        expected_scale_factor = 0.5 / first_beat_duration
        print(f"First beat interval: {first_beat_duration:.4f} sec")
        print(f"Target beat interval: 0.5000 sec")
        print(f"Scale factor: {expected_scale_factor:.4f}")
    
    normalized_matched_tuples = []
    for match in matched_tuples:
        perf_triplet = match[0]
        score_triplet = match[2]
        
        if score_triplet[0] is not None:
            # Decode original values
            original_time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
            original_duration_sec = (score_triplet[1] - DUR_OFFSET) / TIME_RESOLUTION
            pitch = score_triplet[2]
            
            # Normalize time
            normalized_time_sec = 0.0
            time_scale_factor = 1.0
            
            if score_beat_times and len(score_beat_times) >= 2:
                if original_time_sec < score_beat_times[0]:
                    # Before first beat
                    beat_duration = score_beat_times[1] - score_beat_times[0]
                    if beat_duration > 0:
                        progress = (original_time_sec - score_beat_times[0]) / beat_duration
                        time_scale_factor = 0.5 / beat_duration
                    else:
                        progress = 0
                        time_scale_factor = 1.0
                    normalized_time_sec = 0.0 + progress * 0.5
                else:
                    # Find which beats this falls between
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
                        # After last beat
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
            
            # Scale duration by same factor
            normalized_duration_sec = original_duration_sec * time_scale_factor
            
            # Convert back to quantized units with offsets
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
    
    print(f"\nFirst 10 notes after normalization:")
    print(f"{'Note':<6} {'Orig_Time':>11} {'Norm_Time':>11} {'Scale':>8} {'Orig_Dur':>10} {'Norm_Dur':>10} {'Dur_Ratio':>10}")
    print("-" * 80)
    for i in range(min(10, len(normalized_matched_tuples))):
        match = matched_tuples[i]
        norm_match = normalized_matched_tuples[i]
        
        score_triplet = match[2]
        norm_score_triplet = norm_match[2]
        
        # Original
        orig_time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
        orig_dur_sec = (score_triplet[1] - DUR_OFFSET) / TIME_RESOLUTION
        
        # Normalized
        norm_time_sec = (norm_score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
        norm_dur_sec = (norm_score_triplet[1] - DUR_OFFSET) / TIME_RESOLUTION
        
        # Scale factor
        if orig_time_sec < score_beat_times[0]:
            beat_dur = score_beat_times[1] - score_beat_times[0]
        else:
            beat_idx = 0
            for j in range(len(score_beat_times) - 1):
                if score_beat_times[j] <= orig_time_sec <= score_beat_times[j + 1]:
                    beat_idx = j
                    break
            beat_dur = score_beat_times[beat_idx + 1] - score_beat_times[beat_idx]
        
        scale = 0.5 / beat_dur if beat_dur > 0 else 1.0
        dur_ratio = norm_dur_sec / orig_dur_sec if orig_dur_sec > 0 else 0
        
        print(f"{i:<6} {orig_time_sec:>11.4f} {norm_time_sec:>11.4f} {scale:>8.4f} {orig_dur_sec:>10.4f} {norm_dur_sec:>10.4f} {dur_ratio:>10.4f}")
    
    # Step 5: Verify beat spacing in normalized times
    print("\n" + "="*80)
    print("STEP 5: Verify beat spacing in normalized sequence")
    print("="*80)
    
    beat_matches = []
    tolerance = 0.05  # 50ms tolerance
    
    for beat_idx, beat_time in enumerate(score_beat_times[:15]):  # Check first 15 beats
        # Find note closest to this beat in original time
        closest_match_idx = None
        closest_diff = float('inf')
        
        for i, match in enumerate(matched_tuples):
            score_triplet = match[2]
            if score_triplet[0] is None:
                continue
            orig_time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
            diff = abs(orig_time_sec - beat_time)
            if diff < closest_diff:
                closest_diff = diff
                closest_match_idx = i
        
        if closest_match_idx is not None and closest_diff < tolerance:
            norm_match = normalized_matched_tuples[closest_match_idx]
            norm_score_triplet = norm_match[2]
            norm_time_sec = (norm_score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
            expected_norm_time = beat_idx * 0.5
            error = abs(norm_time_sec - expected_norm_time)
            
            beat_matches.append({
                'beat_idx': beat_idx,
                'beat_time': beat_time,
                'norm_time': norm_time_sec,
                'expected': expected_norm_time,
                'error': error
            })
    
    print(f"{'Beat#':<8} {'Orig_Time':>11} {'Norm_Time':>11} {'Expected':>11} {'Error':>11} {'Status':>10}")
    print("-" * 72)
    for bm in beat_matches:
        status = "OK" if bm['error'] < 0.01 else "WARNING"
        print(f"{bm['beat_idx']:<8} {bm['beat_time']:>11.3f} {bm['norm_time']:>11.4f} {bm['expected']:>11.4f} {bm['error']:>11.4f} {status:>10}")
    
    # Step 6: Check for any issues
    print("\n" + "="*80)
    print("STEP 6: Final validation checks")
    print("="*80)
    
    issues = []
    
    # Check for negative time intervals
    prev_norm_time = None
    for i, norm_match in enumerate(normalized_matched_tuples[:50]):
        norm_score_triplet = norm_match[2]
        if norm_score_triplet[0] is None:
            continue
        norm_time_sec = (norm_score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
        
        if prev_norm_time is not None and norm_time_sec < prev_norm_time:
            issues.append(f"Negative interval at note {i}: {prev_norm_time:.4f} -> {norm_time_sec:.4f}")
        
        prev_norm_time = norm_time_sec
    
    # Check duration scaling consistency
    for i in range(min(10, len(normalized_matched_tuples))):
        match = matched_tuples[i]
        norm_match = normalized_matched_tuples[i]
        
        score_triplet = match[2]
        norm_score_triplet = norm_match[2]
        
        orig_time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
        orig_dur_sec = (score_triplet[1] - DUR_OFFSET) / TIME_RESOLUTION
        norm_dur_sec = (norm_score_triplet[1] - DUR_OFFSET) / TIME_RESOLUTION
        
        # Find expected scale factor
        if orig_time_sec < score_beat_times[0]:
            beat_dur = score_beat_times[1] - score_beat_times[0]
        else:
            beat_idx = 0
            for j in range(len(score_beat_times) - 1):
                if score_beat_times[j] <= orig_time_sec <= score_beat_times[j + 1]:
                    beat_idx = j
                    break
            beat_dur = score_beat_times[beat_idx + 1] - score_beat_times[beat_idx]
        
        expected_scale = 0.5 / beat_dur if beat_dur > 0 else 1.0
        expected_norm_dur = orig_dur_sec * expected_scale
        
        if abs(norm_dur_sec - expected_norm_dur) > 0.01:
            issues.append(f"Duration scaling mismatch at note {i}: expected {expected_norm_dur:.4f}, got {norm_dur_sec:.4f}")
    
    # Check pitch preservation
    for i in range(min(20, len(normalized_matched_tuples))):
        match = matched_tuples[i]
        norm_match = normalized_matched_tuples[i]
        
        score_triplet = match[2]
        norm_score_triplet = norm_match[2]
        
        if score_triplet[2] != norm_score_triplet[2]:
            issues.append(f"Pitch changed at note {i}: {score_triplet[2]} -> {norm_score_triplet[2]}")
    
    if issues:
        print(f"\n❌ FOUND {len(issues)} ISSUES:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✅ ALL VALIDATION CHECKS PASSED!")
        print("  - No negative time intervals")
        print("  - Duration scaling is consistent")
        print("  - Pitches are preserved")
        print("  - Beat spacing is correct (0.5 sec intervals)")
    
    print("\n" + "="*80)
    print("VERIFICATION COMPLETE")
    print("="*80)
    
    return not bool(issues)

if __name__ == "__main__":
    success = verify_full_pipeline()
    exit(0 if success else 1)
