"""
Analyze timing alignment between score and performance (control) tokens.

Checks if score tokens frequently have different time coordinates than their
corresponding control tokens, and whether they progress at different rates.
"""
import random
from anticipation.vocab import CONTROL_OFFSET, TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET, REST, ANTICIPATE, SEPARATOR
from anticipation.config import TIME_RESOLUTION
import numpy as np
import matplotlib.pyplot as plt

def analyze_sequence_timing(tokens):
    """
    Analyze timing differences between aligned score and control triplets.
    
    Returns:
        time_diffs: list of (score_time - control_time) for each aligned pair
        score_times: list of score time values
        control_times: list of control time values
        score_positions: list of score triplet positions
        control_positions: list of control triplet positions
        pace_ratios: list of (score_time_delta / control_time_delta) for consecutive pairs
    """
    # Find all score and control triplet positions (skip special tokens)
    score_triplets = []
    control_triplets = []
    
    i = 4  # Skip [ANTICIPATE, SEP, SEP, SEP]
    while i < len(tokens) - 2:
        time_tok, dur_tok, note_tok = tokens[i], tokens[i+1], tokens[i+2]
        
        if (time_tok >= TIME_OFFSET and time_tok < CONTROL_OFFSET and 
            dur_tok >= DUR_OFFSET and dur_tok < CONTROL_OFFSET and 
            note_tok >= NOTE_OFFSET and note_tok < CONTROL_OFFSET and
            note_tok != REST):
            # Score triplet (non-REST)
            score_time = time_tok - TIME_OFFSET
            score_triplets.append((i, score_time, dur_tok - DUR_OFFSET, note_tok - NOTE_OFFSET))
            i += 3
        elif (time_tok >= CONTROL_OFFSET and 
              dur_tok >= CONTROL_OFFSET and 
              note_tok >= CONTROL_OFFSET):
            # Control triplet
            control_time = (time_tok - CONTROL_OFFSET - TIME_OFFSET) if time_tok >= CONTROL_OFFSET + TIME_OFFSET else (time_tok - CONTROL_OFFSET)
            control_dur = (dur_tok - CONTROL_OFFSET - DUR_OFFSET) if dur_tok >= CONTROL_OFFSET + DUR_OFFSET else (dur_tok - CONTROL_OFFSET)
            control_note = (note_tok - CONTROL_OFFSET - NOTE_OFFSET) if note_tok >= CONTROL_OFFSET + NOTE_OFFSET else (note_tok - CONTROL_OFFSET)
            
            if control_note != REST:
                control_triplets.append((i, control_time, control_dur, control_note))
            i += 3
        else:
            i += 1
    
    # Analyze aligned pairs (score[i] should align with control[i])
    time_diffs = []
    score_times = []
    control_times = []
    score_positions = []
    control_positions = []
    
    num_pairs = min(len(score_triplets), len(control_triplets))
    for i in range(num_pairs):
        score_pos, score_time, score_dur, score_note = score_triplets[i]
        control_pos, control_time, control_dur, control_note = control_triplets[i]
        
        time_diff = score_time - control_time
        time_diffs.append(time_diff)
        score_times.append(score_time)
        control_times.append(control_time)
        score_positions.append(score_pos)
        control_positions.append(control_pos)
    
    # Compute pace ratios (how fast score progresses vs control)
    pace_ratios = []
    for i in range(1, num_pairs):
        score_delta = score_times[i] - score_times[i-1]
        control_delta = control_times[i] - control_times[i-1]
        
        # Avoid division by zero
        if control_delta > 0:
            pace_ratio = score_delta / control_delta
            pace_ratios.append(pace_ratio)
    
    return time_diffs, score_times, control_times, score_positions, control_positions, pace_ratios

def main():
    train_file = 'data/train_normalized.txt'
    test_file = 'data/test_normalized.txt'
    num_sequences = 100
    
    print("="*80)
    print("TIMING ALIGNMENT ANALYSIS")
    print("="*80)
    print(f"Analyzing {num_sequences} sequences from train and test data")
    print()
    
    # Analyze both train and test
    for data_name, data_file in [('Train', train_file), ('Test', test_file)]:
        print(f"\n{'='*80}")
        print(f"{data_name} Data: {data_file}")
        print(f"{'='*80}")
        
        # Load sequences
        with open(data_file, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]
        
        print(f"Total sequences: {len(lines)}")
        
        # Sample sequences
        random.seed(42)
        sampled_lines = random.sample(lines, min(num_sequences, len(lines)))
        
        # Analyze timing for each sequence
        all_time_diffs = []
        all_score_times = []
        all_control_times = []
        all_position_diffs = []
        all_pace_ratios = []
        
        for line in sampled_lines:
            if '|' in line:
                token_part = line.split('|')[0].strip()
            else:
                token_part = line
            
            tokens = [int(t) for t in token_part.split()]
            
            time_diffs, score_times, control_times, score_positions, control_positions, pace_ratios = analyze_sequence_timing(tokens)
            
            if len(time_diffs) > 0:
                all_time_diffs.extend(time_diffs)
                all_score_times.extend(score_times)
                all_control_times.extend(control_times)
                all_pace_ratios.extend(pace_ratios)
                
                # Also track position differences in the token sequence
                position_diffs = [s - c for s, c in zip(score_positions, control_positions)]
                all_position_diffs.extend(position_diffs)
        
        if len(all_time_diffs) == 0:
            print("No aligned triplets found!")
            continue
        
        # Convert to seconds (TIME_RESOLUTION = bins per second, e.g., 100)
        all_time_diffs_sec = [t / TIME_RESOLUTION for t in all_time_diffs]
        all_score_times_sec = [t / TIME_RESOLUTION for t in all_score_times]
        all_control_times_sec = [t / TIME_RESOLUTION for t in all_control_times]
        
        # Statistics
        print(f"\nTotal aligned pairs analyzed: {len(all_time_diffs)}")
        print(f"\nTime Difference Statistics (score_time - control_time):")
        print(f"  Mean: {np.mean(all_time_diffs_sec):.3f} seconds ({np.mean(all_time_diffs):.2f} units)")
        print(f"  Std:  {np.std(all_time_diffs_sec):.3f} seconds ({np.std(all_time_diffs):.2f} units)")
        print(f"  Min:  {np.min(all_time_diffs_sec):.3f} seconds ({np.min(all_time_diffs):.2f} units)")
        print(f"  Max:  {np.max(all_time_diffs_sec):.3f} seconds ({np.max(all_time_diffs):.2f} units)")
        print(f"  Median: {np.median(all_time_diffs_sec):.3f} seconds ({np.median(all_time_diffs):.2f} units)")
        
        print(f"\nScore Time Statistics:")
        print(f"  Mean: {np.mean(all_score_times_sec):.3f} seconds ({np.mean(all_score_times):.2f} units)")
        print(f"  Std:  {np.std(all_score_times_sec):.3f} seconds ({np.std(all_score_times):.2f} units)")
        print(f"  Min:  {np.min(all_score_times_sec):.3f} seconds ({np.min(all_score_times):.2f} units)")
        print(f"  Max:  {np.max(all_score_times_sec):.3f} seconds ({np.max(all_score_times):.2f} units)")
        
        print(f"\nControl Time Statistics:")
        print(f"  Mean: {np.mean(all_control_times_sec):.3f} seconds ({np.mean(all_control_times):.2f} units)")
        print(f"  Std:  {np.std(all_control_times_sec):.3f} seconds ({np.std(all_control_times):.2f} units)")
        print(f"  Min:  {np.min(all_control_times_sec):.3f} seconds ({np.min(all_control_times):.2f} units)")
        print(f"  Max:  {np.max(all_control_times_sec):.3f} seconds ({np.max(all_control_times):.2f} units)")
        
        print(f"\nPace Ratio Statistics (score_delta / control_delta):")
        print(f"  Mean: {np.mean(all_pace_ratios):.3f} (1.0 = same pace)")
        print(f"  Std:  {np.std(all_pace_ratios):.3f}")
        print(f"  Min:  {np.min(all_pace_ratios):.3f}")
        print(f"  Max:  {np.max(all_pace_ratios):.3f}")
        print(f"  Median: {np.median(all_pace_ratios):.3f}")
        
        # Analyze how many are close to 1.0 (same pace)
        close_to_1 = sum(1 for r in all_pace_ratios if 0.9 <= r <= 1.1)
        within_factor_2 = sum(1 for r in all_pace_ratios if 0.5 <= r <= 2.0)
        print(f"  Ratios close to 1.0 (0.9-1.1): {close_to_1}/{len(all_pace_ratios)} ({close_to_1/len(all_pace_ratios)*100:.1f}%)")
        print(f"  Ratios within factor of 2 (0.5-2.0): {within_factor_2}/{len(all_pace_ratios)} ({within_factor_2/len(all_pace_ratios)*100:.1f}%)")
        
        print(f"\nToken Position Difference Statistics (score_pos - control_pos):")
        print(f"  Mean: {np.mean(all_position_diffs):.2f}")
        print(f"  Std:  {np.std(all_position_diffs):.2f}")
        print(f"  Min:  {np.min(all_position_diffs):.2f}")
        print(f"  Max:  {np.max(all_position_diffs):.2f}")
        
        # Check how many have exact alignment (time diff = 0)
        exact_matches = sum(1 for d in all_time_diffs if d == 0)
        print(f"\nExact time matches (diff = 0): {exact_matches}/{len(all_time_diffs)} ({exact_matches/len(all_time_diffs)*100:.1f}%)")
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f'{data_name} Data - Timing Alignment Analysis', fontsize=16)
        
        # Plot 1: Time difference distribution (in seconds)
        ax = axes[0, 0]
        ax.hist(all_time_diffs_sec, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(0, color='red', linestyle='--', linewidth=2, label='Perfect alignment')
        ax.set_xlabel('Time Difference (seconds)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Time Differences')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Score vs Control times scatter (in seconds)
        ax = axes[0, 1]
        ax.scatter(all_control_times_sec, all_score_times_sec, alpha=0.3, s=10)
        
        # Add y=x line for perfect alignment
        min_time = min(min(all_score_times_sec), min(all_control_times_sec))
        max_time = max(max(all_score_times_sec), max(all_control_times_sec))
        ax.plot([min_time, max_time], [min_time, max_time], 'r--', linewidth=2, label='Perfect alignment (y=x)')
        
        ax.set_xlabel('Control Time (seconds)')
        ax.set_ylabel('Score Time (seconds)')
        ax.set_title('Score Time vs Control Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Pace ratio distribution
        ax = axes[0, 2]
        ax.hist(all_pace_ratios, bins=50, alpha=0.7, edgecolor='black')
        ax.axvline(1.0, color='red', linestyle='--', linewidth=2, label='Same pace (ratio=1.0)')
        ax.axvline(2.0, color='orange', linestyle=':', linewidth=2, label='2x faster')
        ax.axvline(0.5, color='orange', linestyle=':', linewidth=2, label='2x slower')
        ax.set_xlabel('Pace Ratio (score_delta / control_delta)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Pace Ratios')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Position difference distribution
        ax = axes[1, 0]
        ax.hist(all_position_diffs, bins=50, alpha=0.7, edgecolor='black')
        ax.set_xlabel('Token Position Difference (score_pos - control_pos)')
        ax.set_ylabel('Frequency')
        ax.set_title('Distribution of Token Position Differences')
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Time difference vs sequence position
        ax = axes[1, 1]
        positions = list(range(len(all_time_diffs_sec)))
        ax.scatter(positions, all_time_diffs_sec, alpha=0.3, s=10)
        ax.axhline(0, color='red', linestyle='--', linewidth=2)
        ax.set_xlabel('Triplet Index (across all sequences)')
        ax.set_ylabel('Time Difference (seconds)')
        ax.set_title('Time Difference Over Sequence Progression')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Pace ratio over sequence progression
        ax = axes[1, 2]
        pace_positions = list(range(len(all_pace_ratios)))
        ax.scatter(pace_positions, all_pace_ratios, alpha=0.3, s=10)
        ax.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Same pace')
        ax.axhline(2.0, color='orange', linestyle=':', linewidth=1.5)
        ax.axhline(0.5, color='orange', linestyle=':', linewidth=1.5)
        ax.set_xlabel('Triplet Transition Index (across all sequences)')
        ax.set_ylabel('Pace Ratio')
        ax.set_title('Pace Ratio Over Sequence Progression')
        ax.set_ylim([0, min(5, max(all_pace_ratios))])  # Cap at 5 for readability
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = f'timing_alignment_{data_name.lower()}.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        print(f"\nPlot saved to: {plot_path}")
        plt.close()
    
    print(f"\n{'='*80}")
    print("Analysis complete!")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()
