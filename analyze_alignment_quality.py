"""
Analyze how often score notes match corresponding performance notes in tokenization.

This checks the alignment quality from align_tokens2 to understand:
1. How often performance and score notes are matched (same pitch)
2. How often they differ (alignment errors or expressive differences)
3. Distribution of matches across the dataset
"""

import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from alignment import align_tokens2
from anticipation.vocab import *
from anticipation.config import *


def analyze_single_alignment(filegroup, skip_Nones=True):
    """
    Analyze alignment quality for a single piece.
    
    Returns dict with:
    - total_matched: number of aligned pairs
    - pitch_matches: number where performance pitch == score pitch
    - pitch_mismatches: number where they differ
    - match_rate: pitch_matches / total_matched
    """
    file1, file2, file3, file4 = filegroup
    
    try:
        matched_tuples = align_tokens2(file1, file2, file3, file4, skip_Nones=skip_Nones)
    except Exception as e:
        return {'error': str(e), 'total_matched': 0, 'pitch_matches': 0, 'pitch_mismatches': 0}
    
    total_matched = 0
    pitch_matches = 0
    pitch_mismatches = 0
    
    for match in matched_tuples:
        perf_triplet = match[0]  # Performance: ALL elements have CONTROL_OFFSET added
        score_triplet = match[2]  # Score: [time, dur+DUR_OFFSET, note+NOTE_OFFSET] or [None, None, None]
        
        # Skip if no score match
        if score_triplet[0] is None:
            continue
        
        total_matched += 1
        
        # Extract raw pitch values
        # Performance: [time+CONTROL_OFFSET, dur+DUR_OFFSET+CONTROL_OFFSET, note+NOTE_OFFSET+CONTROL_OFFSET]
        # Score: [time, dur+DUR_OFFSET, note+NOTE_OFFSET]
        perf_note = perf_triplet[2] - NOTE_OFFSET - CONTROL_OFFSET  # Remove both offsets
        score_note = score_triplet[2] - NOTE_OFFSET
        
        if perf_note == score_note:
            pitch_matches += 1
        else:
            pitch_mismatches += 1
    
    match_rate = pitch_matches / total_matched if total_matched > 0 else 0.0
    
    return {
        'total_matched': total_matched,
        'pitch_matches': pitch_matches,
        'pitch_mismatches': pitch_mismatches,
        'match_rate': match_rate,
        'error': None
    }


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze alignment quality')
    parser.add_argument('--num-pieces', type=int, default=10,
                       help='Number of pieces to analyze (default: 10)')
    parser.add_argument('--asap-root', default='./asap-dataset-master',
                       help='Path to ASAP dataset root')
    args = parser.parse_args()
    
    print("=" * 80)
    print("ALIGNMENT QUALITY ANALYSIS")
    print("=" * 80)
    print()
    print("Analyzing how often score notes match performance notes in tokenization...")
    print()
    
    # Load ASAP dataset metadata
    asap_root = args.asap_root
    meta_csv = os.path.join(asap_root, 'metadata.csv')
    df = pd.read_csv(meta_csv)
    
    print(f"Found {len(df)} pieces in ASAP dataset")
    print(f"Testing on {args.num_pieces} pieces")
    print()
    
    # Build file tuples (limit to num_pieces)
    datafiles = []
    for idx, row in df.iterrows():
        if idx >= args.num_pieces:
            break
        file1 = os.path.join(asap_root, row['midi_performance'])
        file2 = os.path.join(asap_root, row['midi_score'])
        file3 = os.path.join(asap_root, row['performance_annotations'])
        file4 = os.path.join(asap_root, row['midi_score_annotations'])
        datafiles.append((file1, file2, file3, file4))
    
    # Analyze pieces
    results = []
    errors = 0
    
    print("Analyzing alignments...")
    for i, filegroup in enumerate(tqdm(datafiles, desc="Processing pieces")):
        result = analyze_single_alignment(filegroup, skip_Nones=True)
        if result['error']:
            errors += 1
        results.append(result)
    
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    
    # Filter out errors
    valid_results = [r for r in results if r['error'] is None and r['total_matched'] > 0]
    
    if errors > 0:
        print(f"⚠ {errors} pieces had errors during alignment")
        print()
    
    if not valid_results:
        print("❌ No valid results to analyze")
        return
    
    # Aggregate statistics
    total_pairs = sum(r['total_matched'] for r in valid_results)
    total_matches = sum(r['pitch_matches'] for r in valid_results)
    total_mismatches = sum(r['pitch_mismatches'] for r in valid_results)
    
    overall_match_rate = total_matches / total_pairs if total_pairs > 0 else 0.0
    
    print(f"Valid pieces analyzed: {len(valid_results)}")
    print(f"Total aligned pairs: {total_pairs:,}")
    print(f"Pitch matches: {total_matches:,} ({overall_match_rate*100:.2f}%)")
    print(f"Pitch mismatches: {total_mismatches:,} ({(1-overall_match_rate)*100:.2f}%)")
    print()
    
    # Per-piece statistics
    match_rates = [r['match_rate'] for r in valid_results]
    print("Per-piece match rate statistics:")
    print(f"  Mean: {np.mean(match_rates)*100:.2f}%")
    print(f"  Median: {np.median(match_rates)*100:.2f}%")
    print(f"  Std: {np.std(match_rates)*100:.2f}%")
    print(f"  Min: {np.min(match_rates)*100:.2f}%")
    print(f"  Max: {np.max(match_rates)*100:.2f}%")
    print()
    
    # Distribution
    print("Match rate distribution:")
    bins = [0, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    hist, _ = np.histogram(match_rates, bins=bins)
    
    for i in range(len(bins)-1):
        count = hist[i]
        pct = count / len(valid_results) * 100
        print(f"  {bins[i]*100:5.1f}% - {bins[i+1]*100:5.1f}%: {count:3d} pieces ({pct:5.1f}%)")
    print()
    
    # Find worst and best pieces
    sorted_results = sorted(zip(match_rates, range(len(valid_results))), key=lambda x: x[0])
    
    print("Bottom 5 pieces (lowest match rates):")
    for rate, idx in sorted_results[:5]:
        r = valid_results[idx]
        print(f"  {rate*100:6.2f}%: {r['pitch_matches']}/{r['total_matched']} matches")
    print()
    
    print("Top 5 pieces (highest match rates):")
    for rate, idx in sorted_results[-5:]:
        r = valid_results[idx]
        print(f"  {rate*100:6.2f}%: {r['pitch_matches']}/{r['total_matched']} matches")
    print()
    
    # Interpretation
    print("=" * 80)
    print("INTERPRETATION")
    print("=" * 80)
    print()
    
    if overall_match_rate > 0.95:
        print("✅ EXCELLENT: >95% of notes match - very high alignment quality")
    elif overall_match_rate > 0.90:
        print("✅ GOOD: 90-95% match rate - good alignment quality")
    elif overall_match_rate > 0.80:
        print("⚠ FAIR: 80-90% match rate - acceptable but some alignment issues")
    else:
        print("❌ POOR: <80% match rate - significant alignment issues")
    
    print()
    print("Note: Some mismatches may be intentional (e.g., ornamentations, trills)")
    print("in the performance that don't appear in the score, or vice versa.")
    print()
    
    # Save detailed results
    output_file = 'alignment_quality_results.txt'
    with open(output_file, 'w') as f:
        f.write("Alignment Quality Analysis Results\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total pieces: {len(valid_results)}\n")
        f.write(f"Total aligned pairs: {total_pairs:,}\n")
        f.write(f"Overall match rate: {overall_match_rate*100:.2f}%\n\n")
        
        f.write("Per-piece results:\n")
        for i, r in enumerate(valid_results):
            f.write(f"Piece {i+1}: {r['pitch_matches']}/{r['total_matched']} = {r['match_rate']*100:.2f}%\n")
    
    print(f"✓ Detailed results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
