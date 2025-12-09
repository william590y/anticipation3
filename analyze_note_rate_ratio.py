"""
Analyze the ratio of note rates between performance and score sequences.
This helps understand if performances tend to have more/fewer notes per second
compared to their corresponding scores.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from multiprocessing import Pool
from tqdm import tqdm
import pandas as pd

from alignment import align_tokens2
from anticipation.vocab import CONTROL_OFFSET, TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET
from anticipation.config import TIME_RESOLUTION

NUM_WORKERS = 32
ASAP_PATH = 'asap-dataset-master'


def analyze_note_rates(args):
    """
    Analyze note rates for a single piece.
    
    Args: tuple of (file1, file2, file3, file4, piece_name)
    
    Returns dict with:
        - piece_name: name of the piece
        - perf_note_rate: notes per second in performance
        - score_note_rate: notes per second in score
        - rate_ratio: perf_note_rate / score_note_rate
        - perf_duration: total duration in seconds
        - score_duration: total duration in seconds
        - perf_note_count: number of notes in performance
        - score_note_count: number of notes in score
    """
    file1, file2, file3, file4, piece_name = args
    
    try:
        matched_tuples = align_tokens2(file1, file2, file3, file4, skip_Nones=False)
        
        if not matched_tuples:
            return {'piece_name': piece_name, 'success': False, 'error': 'No matched tuples'}
        
        # Extract performance times (remove offsets, convert to seconds)
        perf_times = []
        for match in matched_tuples:
            perf_triplet = match[0]  # [CONTROL_OFFSET+time, dur, pitch]
            perf_time = (perf_triplet[0] - CONTROL_OFFSET - TIME_OFFSET) / TIME_RESOLUTION
            perf_times.append(perf_time)
        
        # Extract score times (only from matched notes)
        score_times = []
        for match in matched_tuples:
            score_triplet = match[2]  # [time, dur, pitch] or [None, None, None]
            if score_triplet[0] is not None:
                score_time = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
                score_times.append(score_time)
        
        if not perf_times or not score_times:
            return {'piece_name': piece_name, 'success': False, 'error': 'No valid times'}
        
        # Calculate durations (min to max time)
        perf_duration = max(perf_times) - min(perf_times)
        score_duration = max(score_times) - min(score_times)
        
        if perf_duration <= 0 or score_duration <= 0:
            return {'piece_name': piece_name, 'success': False, 'error': 'Invalid duration'}
        
        # Calculate note rates (notes per second)
        perf_note_count = len(perf_times)
        score_note_count = len(score_times)
        
        perf_note_rate = perf_note_count / perf_duration
        score_note_rate = score_note_count / score_duration
        
        rate_ratio = perf_note_rate / score_note_rate
        
        return {
            'piece_name': piece_name,
            'success': True,
            'perf_note_rate': perf_note_rate,
            'score_note_rate': score_note_rate,
            'rate_ratio': rate_ratio,
            'perf_duration': perf_duration,
            'score_duration': score_duration,
            'perf_note_count': perf_note_count,
            'score_note_count': score_note_count
        }
        
    except Exception as e:
        return {
            'piece_name': piece_name,
            'success': False,
            'error': str(e)
        }


def main():
    print("="*80)
    print("NOTE RATE RATIO ANALYSIS")
    print("="*80)
    print("Analyzing the ratio of note rates (notes/second) between performance and score")
    print()
    
    # Load ASAP metadata
    metadata_path = os.path.join(ASAP_PATH, 'metadata.csv')
    df = pd.read_csv(metadata_path)
    
    print(f"Found {len(df)} pieces in metadata")
    
    # Collect all valid datafiles
    datafiles = []
    piece_names = []
    
    for _, row in df.iterrows():
        file1 = os.path.join(ASAP_PATH, row['midi_performance'])
        file2 = os.path.join(ASAP_PATH, row['midi_score'])
        file3 = os.path.join(ASAP_PATH, row['performance_annotations'])
        file4 = os.path.join(ASAP_PATH, row['midi_score_annotations'])
        
        if all(os.path.exists(f) for f in [file1, file2, file3, file4]):
            datafiles.append((file1, file2, file3, file4))
            piece_names.append(row['midi_performance'])
    
    print(f"Found {len(datafiles)} valid pieces with all required files")
    print()
    
    # Analyze all pieces
    num_samples = min(100, len(datafiles))  # Analyze up to 100 pieces
    print(f"Analyzing {num_samples} pieces with {NUM_WORKERS} workers...")
    print()
    
    # Prepare arguments for parallel processing
    piece_args = [(files[0], files[1], files[2], files[3], name) 
                  for files, name in zip(datafiles[:num_samples], piece_names[:num_samples])]
    
    # Run parallel analysis
    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(analyze_note_rates, piece_args), 
                           total=num_samples, 
                           desc="Analyzing pieces"))
    
    # Separate successful and failed results
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"Successfully analyzed: {len(successful_results)}/{num_samples}")
    print(f"Failed pieces: {len(failed_results)}")
    
    if failed_results:
        print()
        print(f"Failed pieces ({len(failed_results)}):")
        for r in failed_results[:10]:
            print(f"  - {r['piece_name']}: {r['error']}")
        if len(failed_results) > 10:
            print(f"  ... and {len(failed_results) - 10} more")
    
    if not successful_results:
        print("No successful results to analyze.")
        return
    
    # Extract statistics
    rate_ratios = [r['rate_ratio'] for r in successful_results]
    perf_rates = [r['perf_note_rate'] for r in successful_results]
    score_rates = [r['score_note_rate'] for r in successful_results]
    
    print()
    print("Note Rate Ratio Statistics (performance / score):")
    print(f"  Mean ratio: {np.mean(rate_ratios):.3f}")
    print(f"  Median ratio: {np.median(rate_ratios):.3f}")
    print(f"  Std ratio: {np.std(rate_ratios):.3f}")
    print(f"  Min ratio: {np.min(rate_ratios):.3f}")
    print(f"  Max ratio: {np.max(rate_ratios):.3f}")
    print()
    print(f"  Ratios < 1.0 (perf slower): {sum(1 for r in rate_ratios if r < 1.0)}/{len(rate_ratios)} ({sum(1 for r in rate_ratios if r < 1.0)/len(rate_ratios)*100:.1f}%)")
    print(f"  Ratios ≈ 1.0 (0.95-1.05): {sum(1 for r in rate_ratios if 0.95 <= r <= 1.05)}/{len(rate_ratios)} ({sum(1 for r in rate_ratios if 0.95 <= r <= 1.05)/len(rate_ratios)*100:.1f}%)")
    print(f"  Ratios > 1.0 (perf faster): {sum(1 for r in rate_ratios if r > 1.0)}/{len(rate_ratios)} ({sum(1 for r in rate_ratios if r > 1.0)/len(rate_ratios)*100:.1f}%)")
    
    print()
    print("Performance Note Rates:")
    print(f"  Mean: {np.mean(perf_rates):.2f} notes/sec")
    print(f"  Median: {np.median(perf_rates):.2f} notes/sec")
    print(f"  Std: {np.std(perf_rates):.2f} notes/sec")
    
    print()
    print("Score Note Rates:")
    print(f"  Mean: {np.mean(score_rates):.2f} notes/sec")
    print(f"  Median: {np.median(score_rates):.2f} notes/sec")
    print(f"  Std: {np.std(score_rates):.2f} notes/sec")
    
    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Distribution of rate ratios
    axes[0, 0].hist(rate_ratios, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].axvline(np.mean(rate_ratios), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rate_ratios):.3f}')
    axes[0, 0].axvline(np.median(rate_ratios), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(rate_ratios):.3f}')
    axes[0, 0].axvline(1.0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Equal rates')
    axes[0, 0].set_xlabel('Rate Ratio (Performance / Score)')
    axes[0, 0].set_ylabel('Number of Pieces')
    axes[0, 0].set_title('Distribution of Note Rate Ratios')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Log-scale distribution
    axes[0, 1].hist(rate_ratios, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(np.mean(rate_ratios), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(rate_ratios):.3f}')
    axes[0, 1].axvline(1.0, color='black', linestyle='-', linewidth=1, alpha=0.5, label='Equal rates')
    axes[0, 1].set_xlabel('Rate Ratio (Performance / Score)')
    axes[0, 1].set_ylabel('Number of Pieces (log scale)')
    axes[0, 1].set_title('Distribution of Note Rate Ratios (Log Scale)')
    axes[0, 1].set_yscale('log')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: CDF of rate ratios
    sorted_ratios = np.sort(rate_ratios)
    cdf = np.arange(1, len(sorted_ratios) + 1) / len(sorted_ratios) * 100
    axes[0, 2].plot(sorted_ratios, cdf, linewidth=2)
    axes[0, 2].axvline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Equal rates')
    axes[0, 2].axhline(50, color='gray', linestyle=':', alpha=0.5)
    axes[0, 2].set_xlabel('Rate Ratio (Performance / Score)')
    axes[0, 2].set_ylabel('Cumulative Percentage (%)')
    axes[0, 2].set_title('CDF of Note Rate Ratios')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Performance vs Score rates scatter
    axes[1, 0].scatter(score_rates, perf_rates, alpha=0.5, s=30)
    
    # Add diagonal line (equal rates)
    max_rate = max(max(score_rates), max(perf_rates))
    axes[1, 0].plot([0, max_rate], [0, max_rate], 'k--', alpha=0.5, label='Equal rates')
    
    axes[1, 0].set_xlabel('Score Note Rate (notes/sec)')
    axes[1, 0].set_ylabel('Performance Note Rate (notes/sec)')
    axes[1, 0].set_title('Performance vs Score Note Rates')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Box plot comparison
    axes[1, 1].boxplot([perf_rates, score_rates], labels=['Performance', 'Score'])
    axes[1, 1].set_ylabel('Note Rate (notes/sec)')
    axes[1, 1].set_title('Note Rate Distributions')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Rate ratio vs piece index (sorted)
    sorted_indices = np.argsort(rate_ratios)
    sorted_rate_ratios = [rate_ratios[i] for i in sorted_indices]
    
    axes[1, 2].plot(range(len(sorted_rate_ratios)), sorted_rate_ratios, marker='o', markersize=3, linewidth=1)
    axes[1, 2].axhline(1.0, color='black', linestyle='--', linewidth=1, alpha=0.5, label='Equal rates')
    axes[1, 2].axhline(np.mean(rate_ratios), color='red', linestyle='--', linewidth=1, label=f'Mean: {np.mean(rate_ratios):.3f}')
    axes[1, 2].set_xlabel('Piece Index (sorted by ratio)')
    axes[1, 2].set_ylabel('Rate Ratio (Performance / Score)')
    axes[1, 2].set_title('Rate Ratios Across Pieces')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('note_rate_ratio_analysis.png', dpi=150, bbox_inches='tight')
    print()
    print(f"Visualization saved to note_rate_ratio_analysis.png")
    print()
    
    # Interpretation
    print("="*80)
    print("INTERPRETATION")
    print("="*80)
    print()
    print("Note rate ratio (performance / score):")
    print("  - Ratio < 1.0: Performance has fewer notes per second (slower, more sparse)")
    print("  - Ratio = 1.0: Same note density")
    print("  - Ratio > 1.0: Performance has more notes per second (faster, denser)")
    print()
    
    if np.mean(rate_ratios) < 0.95:
        print("⚠️  Performances tend to be SPARSER than scores (fewer notes per second)")
    elif np.mean(rate_ratios) > 1.05:
        print("⚠️  Performances tend to be DENSER than scores (more notes per second)")
    else:
        print("✓ Performances have similar note density to scores on average")
    
    print()
    print("This ratio is different from temporal pace ratio (which measures speed).")
    print("A piece can have the same note rate but different pace if notes are clustered differently.")
    print()
    print("="*80)


if __name__ == "__main__":
    main()
