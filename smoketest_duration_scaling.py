"""
Smoketest to verify duration scaling works correctly.
Tests on a single piece to check:
1. Score times are normalized to 0.5 sec between beats
2. Durations are scaled by the same factor as times
3. No obvious errors in the normalization logic
"""

import os
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, os.path.dirname(__file__))

from anticipation.config import *
from anticipation.vocab import *
from alignment import align_tokens2, load_annotation_file

ASAP_PATH = 'asap-dataset-master'

def smoketest_single_piece():
    """Test duration scaling on a single piece"""
    
    # Get ASAP dataset paths
    asap_annotations = pd.read_csv(os.path.join(ASAP_PATH, "metadata.csv"))
    
    # Pick first piece
    row = asap_annotations.iloc[0]
    piece_name = f"{row['composer']} - {row['title']}"
    
    midi_score_filename = row["midi_score"]
    midi_performance_filename = row["midi_performance"]
    
    file1 = os.path.join(ASAP_PATH, midi_performance_filename)
    file2 = os.path.join(ASAP_PATH, midi_score_filename)
    file3 = file1.replace(".mid", "_annotations.txt")
    file4 = file2.replace(".mid", "_annotations.txt")
    
    print(f"\nTesting piece: {piece_name}")
    print(f"Score MIDI: {midi_score_filename}")
    print(f"Performance MIDI: {midi_performance_filename}")
    
    # Align
    print("\nAligning...")
    matched_tuples = align_tokens2(file1, file2, file3, file4, skip_Nones=True)
    print(f"Got {len(matched_tuples)} matched notes")
    
    if len(matched_tuples) < 10:
        print("ERROR: Not enough matched notes!")
        return
    
    # Load beat annotations
    print("\nLoading beat annotations...")
    score_annotations = load_annotation_file(file4)
    score_beat_times = [anno[0] for anno in score_annotations]
    print(f"Found {len(score_beat_times)} beats")
    print(f"First 5 beat times: {score_beat_times[:5]}")
    
    if len(score_beat_times) < 2:
        print("ERROR: Not enough beats!")
        return
    
    # Calculate expected scale factor for first beat interval
    first_beat_duration = score_beat_times[1] - score_beat_times[0]
    expected_scale_factor = 0.5 / first_beat_duration
    print(f"\nFirst beat interval: {first_beat_duration:.4f} sec")
    print(f"Expected scale factor: {expected_scale_factor:.4f}")
    
    # Normalize score triplets (same logic as main code)
    print("\nNormalizing first 10 score triplets...")
    print("\n{:<6} {:<12} {:<12} {:<12} {:<12} {:<12} {:<12} {:<8}".format(
        "Note", "Orig_Time", "Norm_Time", "Scale", "Orig_Dur", "Norm_Dur", "Dur_Ratio", "Pitch"))
    print("-" * 98)
    
    for idx in range(min(10, len(matched_tuples))):
        match = matched_tuples[idx]
        score_triplet = match[2]
        
        if score_triplet[0] is None:
            continue
        
        # Original values - CHECK UNITS!
        # Triplet format: [time, duration, pitch]
        original_time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
        original_duration_sec = (score_triplet[1] - DUR_OFFSET) / TIME_RESOLUTION  # triplet[1] is duration!
        pitch = score_triplet[2]  # triplet[2] is pitch!
        
        print(f"  DEBUG: score_triplet = {score_triplet}, triplet[1] = {score_triplet[1]}, triplet[1]-DUR_OFFSET = {score_triplet[1]-DUR_OFFSET}, dur_sec = {original_duration_sec:.4f}")
        
        # Normalize time
        normalized_time_sec = 0.0
        time_scale_factor = 1.0
        
        if score_beat_times and len(score_beat_times) >= 2:
            if original_time_sec < score_beat_times[0]:
                # Before first beat - scale relative to first beat
                beat_duration = score_beat_times[1] - score_beat_times[0]
                if beat_duration > 0:
                    # How far before first beat as fraction of beat duration
                    progress = (original_time_sec - score_beat_times[0]) / beat_duration  # negative
                    time_scale_factor = 0.5 / beat_duration
                else:
                    progress = 0
                    time_scale_factor = 1.0
                normalized_time_sec = 0.0 + progress * 0.5  # Will be negative
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
        
        # Scale duration by same factor
        normalized_duration_sec = original_duration_sec * time_scale_factor
        
        # Check ratio
        duration_ratio = normalized_duration_sec / original_duration_sec if original_duration_sec > 0 else 0
        
        print("{:<6} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f} {:<8}".format(
            idx,
            original_time_sec,
            normalized_time_sec,
            time_scale_factor,
            original_duration_sec,
            normalized_duration_sec,
            duration_ratio,
            pitch
        ))
    
    # Verify beat spacing
    print("\n\nVerifying beat spacing in normalized times...")
    beat_notes = []
    for i, beat_time in enumerate(score_beat_times[:10]):  # Check first 10 beats
        # Find note closest to this beat
        closest_idx = None
        closest_diff = float('inf')
        
        for idx, match in enumerate(matched_tuples):
            score_triplet = match[2]
            if score_triplet[0] is None:
                continue
            original_time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
            diff = abs(original_time_sec - beat_time)
            if diff < closest_diff:
                closest_diff = diff
                closest_idx = idx
        
        if closest_idx is not None and closest_diff < 0.05:  # Within 50ms
            match = matched_tuples[closest_idx]
            score_triplet = match[2]
            original_time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
            
            # Calculate normalized time for this note
            if i == 0:
                if score_beat_times[0] > 0:
                    ratio = original_time_sec / score_beat_times[0]
                else:
                    ratio = 0
                normalized_time_sec = 0.0 + ratio * 0.5
            else:
                beat_duration = score_beat_times[i] - score_beat_times[i-1]
                if beat_duration > 0:
                    progress = (original_time_sec - score_beat_times[i-1]) / beat_duration
                else:
                    progress = 0
                normalized_time_sec = (i-1) * 0.5 + progress * 0.5
            
            beat_notes.append((i, beat_time, original_time_sec, normalized_time_sec, closest_diff))
    
    print("\n{:<6} {:<12} {:<12} {:<12} {:<12}".format(
        "Beat#", "Beat_Time", "Note_Time", "Norm_Time", "Error"))
    print("-" * 66)
    for beat_num, beat_time, note_time, norm_time, error in beat_notes:
        expected_norm_time = beat_num * 0.5
        print("{:<6} {:<12.4f} {:<12.4f} {:<12.4f} {:<12.4f}".format(
            beat_num, beat_time, note_time, norm_time, error))
        if abs(norm_time - expected_norm_time) > 0.05:
            print(f"  WARNING: Expected {expected_norm_time:.4f}, got {norm_time:.4f}")
    
    # Check if intervals between consecutive normalized times look reasonable
    print("\n\nChecking interval consistency...")
    print("\n{:<6} {:<12} {:<12} {:<12} {:<12}".format(
        "Note", "Orig_Time", "Norm_Time", "Interval", "Pitch"))
    print("-" * 66)
    
    intervals = []
    prev_norm_time = None
    prev_idx = None
    
    for idx in range(min(50, len(matched_tuples))):
        match = matched_tuples[idx]
        score_triplet = match[2]
        
        if score_triplet[0] is None:
            continue
        
        original_time_sec = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
        pitch = score_triplet[2]
        
        # Calculate normalized time (same logic)
        normalized_time_sec = 0.0
        if score_beat_times and len(score_beat_times) >= 2:
            if original_time_sec < score_beat_times[0]:
                # Before first beat
                beat_duration = score_beat_times[1] - score_beat_times[0]
                progress = (original_time_sec - score_beat_times[0]) / beat_duration if beat_duration > 0 else 0
                normalized_time_sec = 0.0 + progress * 0.5
            else:
                found = False
                for i in range(len(score_beat_times) - 1):
                    if score_beat_times[i] <= original_time_sec <= score_beat_times[i + 1]:
                        beat_duration = score_beat_times[i + 1] - score_beat_times[i]
                        progress = (original_time_sec - score_beat_times[i]) / beat_duration if beat_duration > 0 else 0
                        normalized_time_sec = i * 0.5 + progress * 0.5
                        found = True
                        break
                
                if not found:
                    last_beat_idx = len(score_beat_times) - 1
                    last_beat_duration = score_beat_times[-1] - score_beat_times[-2]
                    progress = (original_time_sec - score_beat_times[-1]) / last_beat_duration if last_beat_duration > 0 else 0
                    normalized_time_sec = last_beat_idx * 0.5 + progress * 0.5
        
        interval_str = "-"
        if prev_norm_time is not None:
            interval = normalized_time_sec - prev_norm_time
            intervals.append(interval)
            interval_str = f"{interval:.4f}"
            
            # Flag negative or suspicious intervals
            if interval < 0:
                interval_str += " [NEG]"
            elif interval == 0:
                interval_str += " [ZERO]"
        
        print("{:<6} {:<12.4f} {:<12.4f} {:<12} {:<12}".format(
            idx, original_time_sec, normalized_time_sec, interval_str, pitch))
        
        prev_norm_time = normalized_time_sec
        prev_idx = idx
    
    if intervals:
        print(f"Intervals between consecutive notes (first 49):")
        print(f"  Min: {min(intervals):.4f} sec")
        print(f"  Max: {max(intervals):.4f} sec")
        print(f"  Mean: {np.mean(intervals):.4f} sec")
        print(f"  Median: {np.median(intervals):.4f} sec")
        print(f"  Std: {np.std(intervals):.4f} sec")
        
        # Count how many are close to multiples of 0.125 (eighth notes at 0.5 beat spacing)
        multiples_0125 = sum(1 for x in intervals if abs(x - round(x / 0.125) * 0.125) < 0.02)
        print(f"  Close to multiples of 0.125 sec: {multiples_0125}/{len(intervals)} ({100*multiples_0125/len(intervals):.1f}%)")

if __name__ == "__main__":
    smoketest_single_piece()
