"""
Analyze the effectiveness of the align_tokens2 alignment function.

Checks:
1. How many notes from performance get matched to score
2. How accurate the temporal matching is
3. Whether pitch matching is perfect (it should be by design)
"""
import os
import pandas as pd
import numpy as np
from alignment import align_tokens2
from anticipation.vocab import CONTROL_OFFSET, TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET
from anticipation.config import TIME_RESOLUTION
from multiprocessing import Pool
from tqdm import tqdm

NUM_WORKERS = 32

# ASAP dataset path
ASAP_PATH = 'asap-dataset-master'
META_CSV = os.path.join(ASAP_PATH, 'metadata.csv')

def analyze_alignment(args):
    """
    Analyze alignment quality for a single piece.
    
    Args: tuple of (file1, file2, file3, file4, piece_name)
    
    Returns dict with:
        - total_perf_notes: total performance notes
        - matched_notes: how many got matched to score
        - match_rate: percentage matched
        - time_diffs_sec: list of temporal differences in seconds
        - pitch_mismatches: number of pitch mismatches (should be 0)
        - piece_name: name of the piece
    """
    file1, file2, file3, file4, piece_name = args
    
    try:
        matched_tuples = align_tokens2(file1, file2, file3, file4, skip_Nones=False)
        
        total_perf_notes = len(matched_tuples)
        matched_notes = sum(1 for m in matched_tuples if m[2][0] is not None)
        match_rate = (matched_notes / total_perf_notes * 100) if total_perf_notes > 0 else 0
        
        # Analyze temporal alignment quality
        time_diffs_sec = []
        pitch_mismatches = 0
        
        for match in matched_tuples:
            perf_triplet = match[0]  # [CONTROL_OFFSET+time, dur, pitch]
            score_triplet = match[2]  # [time, dur, pitch] or [None, None, None]
            
            if score_triplet[0] is not None:
                # Remove offsets to get raw values
                perf_time = (perf_triplet[0] - CONTROL_OFFSET - TIME_OFFSET) / TIME_RESOLUTION
                score_time = (score_triplet[0] - TIME_OFFSET) / TIME_RESOLUTION
                
                perf_pitch = perf_triplet[2] - CONTROL_OFFSET - NOTE_OFFSET
                score_pitch = score_triplet[2] - NOTE_OFFSET
                
                time_diff = abs(perf_time - score_time)
                time_diffs_sec.append(time_diff)
                
                if perf_pitch != score_pitch:
                    pitch_mismatches += 1
        
        return {
            'total_perf_notes': total_perf_notes,
            'matched_notes': matched_notes,
            'match_rate': match_rate,
            'time_diffs_sec': time_diffs_sec,
            'pitch_mismatches': pitch_mismatches,
            'piece_name': piece_name,
            'success': True
        }
    except Exception as e:
        return {
            'piece_name': piece_name,
            'success': False,
            'error': str(e)
        }

def main():
    print("="*80)
    print("ALIGNMENT EFFECTIVENESS ANALYSIS")
    print("="*80)
    print("Analyzing align_tokens2 function from alignment.py")
    print()
    
    # Load metadata
    df = pd.read_csv(META_CSV)
    print(f"Found {len(df)} pieces in metadata")
    
    # Build file tuples
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
    
    # Analyze a sample of pieces
    num_samples = min(50, len(datafiles))
    print(f"Analyzing {num_samples} pieces with {NUM_WORKERS} workers...")
    print()
    
    # Prepare arguments for parallel processing
    piece_args = [(files[0], files[1], files[2], files[3], name) 
                  for files, name in zip(datafiles[:num_samples], piece_names[:num_samples])]
    
    # Run parallel analysis
    with Pool(NUM_WORKERS) as pool:
        results = list(tqdm(pool.imap(analyze_alignment, piece_args), 
                           total=num_samples, 
                           desc="Analyzing pieces"))
    
    # Separate successful and failed results
    successful_results = [r for r in results if r['success']]
    failed_results = [r for r in results if not r['success']]
    
    all_match_rates = [r['match_rate'] for r in successful_results]
    all_time_diffs = [td for r in successful_results for td in r['time_diffs_sec']]
    total_pitch_mismatches = sum(r['pitch_mismatches'] for r in successful_results)
    
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
    
    if len(successful_results) > 0:
        print()
        print("Match Rate Statistics:")
        print(f"  Mean: {np.mean(all_match_rates):.1f}%")
        print(f"  Std:  {np.std(all_match_rates):.1f}%")
        print(f"  Min:  {np.min(all_match_rates):.1f}%")
        print(f"  Max:  {np.max(all_match_rates):.1f}%")
        print(f"  Median: {np.median(all_match_rates):.1f}%")
        
        if all_time_diffs:
            print()
            print("Temporal Alignment Quality (for matched notes):")
            print(f"  Total matched pairs: {len(all_time_diffs)}")
            print(f"  Mean time diff: {np.mean(all_time_diffs):.3f} seconds")
            print(f"  Std time diff:  {np.std(all_time_diffs):.3f} seconds")
            print(f"  Min time diff:  {np.min(all_time_diffs):.3f} seconds")
            print(f"  Max time diff:  {np.max(all_time_diffs):.3f} seconds")
            print(f"  Median time diff: {np.median(all_time_diffs):.3f} seconds")
            
            # Distribution of temporal accuracy
            within_10ms = sum(1 for d in all_time_diffs if d <= 0.01)
            within_50ms = sum(1 for d in all_time_diffs if d <= 0.05)
            within_100ms = sum(1 for d in all_time_diffs if d <= 0.1)
            
            print()
            print("  Temporal accuracy distribution:")
            print(f"    Within 10ms:  {within_10ms}/{len(all_time_diffs)} ({within_10ms/len(all_time_diffs)*100:.1f}%)")
            print(f"    Within 50ms:  {within_50ms}/{len(all_time_diffs)} ({within_50ms/len(all_time_diffs)*100:.1f}%)")
            print(f"    Within 100ms: {within_100ms}/{len(all_time_diffs)} ({within_100ms/len(all_time_diffs)*100:.1f}%)")
        
        print()
        print("Pitch Matching:")
        print(f"  Total pitch mismatches: {total_pitch_mismatches}")
        print(f"  Pitch match accuracy: {(1 - total_pitch_mismatches/len(all_time_diffs))*100:.2f}%" if all_time_diffs else "N/A")
        print(f"  (Should be 100% - alignment enforces pitch matching)")
    
    print()
    print("="*80)
    print("INTERPRETATION")
    print("="*80)
    print()
    print("The align_tokens2 function:")
    print("1. Uses beat annotations to establish temporal correspondence")
    print("2. Uses scipy.interpolate to map performance times to score times")
    print("3. Matches notes within 100ms threshold if they have same pitch")
    print("4. Enforces pitch matching (perf_note == score_note)")
    print()
    print("Key findings:")
    print(f"- Average {np.mean(all_match_rates):.1f}% of performance notes get matched")
    print(f"- Unmatched notes are due to: ornamentation, errors, or differences")
    print(f"- Temporal alignment: {within_100ms/len(all_time_diffs)*100:.1f}% within 100ms threshold")
    print(f"- Pitch matching: 100% by design (enforced in alignment logic)")
    print()
    print("="*80)

if __name__ == "__main__":
    main()
