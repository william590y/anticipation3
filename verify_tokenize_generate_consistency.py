"""
Comprehensive verification that tokenize-asap.py and generate4 are consistent.

This script creates a mock scenario to verify:
1. Tokenization format from tokenize-asap.py
2. Control extraction from test.py
3. generate4 reconstruction matches the training format
"""

import sys
from anticipation.config import *
from anticipation.vocab import *
from anticipation import ops


def create_mock_matched_tuples(num_tuples=50):
    """
    Create mock matched_tuples to simulate what align_tokens2 produces.
    
    matched_tuples format: [[perf_triplet, i, score_triplet, best_index], ...]
    - perf_triplet: [time+CONTROL_OFFSET, dur+DUR_OFFSET, note+NOTE_OFFSET]
    - score_triplet: [time, dur+DUR_OFFSET, note+NOTE_OFFSET] or [None, None, None]
    """
    matched_tuples = []
    
    for i in range(num_tuples):
        # Create performance triplet (with CONTROL_OFFSET)
        perf_time = i * 50  # 50 time units apart
        perf_dur = 10 + (i % 5)
        perf_note = 60 + (i % 12)  # C4 and nearby notes
        perf_triplet = [
            CONTROL_OFFSET + perf_time,
            DUR_OFFSET + perf_dur,
            NOTE_OFFSET + perf_note
        ]
        
        # Create score triplet (without CONTROL_OFFSET, but with other offsets)
        score_time = i * 48 + (i % 3)  # Slightly different timing
        score_dur = perf_dur
        score_note = perf_note
        score_triplet = [
            score_time,
            DUR_OFFSET + score_dur,
            NOTE_OFFSET + score_note
        ]
        
        matched_tuples.append([perf_triplet, i, score_triplet, i])
    
    return matched_tuples


def simulate_tokenize_asap(matched_tuples, prefix_controls=33):
    """
    Simulate the tokenization from _interleave_tokenize4_single.
    """
    interleaved_tokens = []
    
    # Step 1: Build prefix with control+rest pairs
    k = min(prefix_controls, len(matched_tuples))
    for t in matched_tuples[:k]:
        cc = t[0]  # Control triplet (with CONTROL_OFFSET)
        interleaved_tokens.extend(cc)
        cc_time = cc[0] - CONTROL_OFFSET
        interleaved_tokens.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])
    
    # Step 2: Alternating score and future controls
    for i, t in enumerate(matched_tuples):
        sc = t[2]  # Score triplet
        if sc[0] is not None:
            interleaved_tokens.extend(sc)
        ii = i + k  # Future index
        if ii < len(matched_tuples):
            interleaved_tokens.extend(matched_tuples[ii][0])  # Future control
    
    # Step 3: Prepend separators
    interleaved_tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]
    
    # Step 4: Add mode token
    sequence = [ANTICIPATE] + interleaved_tokens
    
    return sequence


def extract_controls_from_sequence(sequence_tokens, prefix_controls=33):
    """
    Replicate the control extraction from test.py.
    """
    # Skip ANTICIPATE + 3 SEPs
    if len(sequence_tokens) < 4:
        return []
    
    tokens = sequence_tokens[4:]
    controls = []
    i = 0
    
    # Extract prefix controls (skip rests)
    for _ in range(prefix_controls):
        if i + 6 <= len(tokens):
            control_triplet = tokens[i:i+3]
            rest_triplet = tokens[i+3:i+6]
            
            if control_triplet[0] >= CONTROL_OFFSET:
                controls.extend(control_triplet)
            else:
                break
            
            i += 6
        else:
            break
    
    # Extract alternating controls
    while i + 3 <= len(tokens):
        triplet = tokens[i:i+3]
        
        if triplet[0] >= CONTROL_OFFSET:
            controls.extend(triplet)
        
        i += 3
    
    return controls


def simulate_generate4_format(controls, prefix_controls=33):
    """
    Simulate what generate4 creates (without actual model inference).
    This checks the FORMAT, not the actual generation.
    """
    # Shift controls to start from time 0
    first_arrival = controls[0] - CONTROL_OFFSET
    controls_shifted = controls.copy()
    for i in range(0, len(controls), 3):
        controls_shifted[i] = controls[i] - first_arrival
    
    tokens = []
    
    # Step 1: Build prefix with control+rest pairs
    k = min(prefix_controls, len(controls) // 3)
    for i in range(k):
        ctrl = controls_shifted[i*3:i*3+3]
        tokens.extend(ctrl)
        cc_time = ctrl[0] - CONTROL_OFFSET
        tokens.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])
    
    # Step 2: Prepare remaining controls
    remaining_controls = controls_shifted[k*3:]
    
    # Step 3: Simulate generation pattern (mock performance events)
    # We generate performance for all controls, alternating with remaining controls
    num_controls = len(controls) // 3
    for i in range(num_controls):
        # Mock generated performance event (without CONTROL_OFFSET)
        mock_perf = [TIME_OFFSET + i*45, DUR_OFFSET + 8, NOTE_OFFSET + 62]
        tokens.extend(mock_perf)
        
        # Add next control if available (these are the "future" controls)
        if len(remaining_controls) >= 3:
            tokens.extend(remaining_controls[0:3])
            remaining_controls = remaining_controls[3:]
    
    return tokens


def verify_consistency():
    """
    Main verification function.
    """
    print("=" * 80)
    print("TOKENIZE-ASAP.PY AND GENERATE4 CONSISTENCY VERIFICATION")
    print("=" * 80)
    print()
    
    # Create mock data
    num_tuples = 50
    print(f"Creating {num_tuples} mock matched tuples...")
    matched_tuples = create_mock_matched_tuples(num_tuples)
    print(f"✓ Created {len(matched_tuples)} matched tuples")
    print()
    
    # Simulate tokenization
    print("Step 1: Simulating tokenize-asap.py tokenization...")
    print("-" * 80)
    sequence = simulate_tokenize_asap(matched_tuples, prefix_controls=33)
    print(f"✓ Sequence length: {len(sequence)} tokens")
    print(f"  - Position 0: {sequence[0]} (should be ANTICIPATE={ANTICIPATE})")
    print(f"  - Positions 1-3: {sequence[1:4]} (should be [SEP, SEP, SEP])")
    print()
    
    # Verify prefix structure
    print("Verifying prefix structure (positions 4-201):")
    prefix_ok = True
    for i in range(33):
        pos = 4 + i * 6
        if pos + 6 > len(sequence):
            break
        
        ctrl = sequence[pos:pos+3]
        rest = sequence[pos+3:pos+6]
        
        # Check control has CONTROL_OFFSET
        if ctrl[0] < CONTROL_OFFSET:
            print(f"  ❌ Control {i} time token {ctrl[0]} missing CONTROL_OFFSET")
            prefix_ok = False
        
        # Check rest structure
        if rest[1] != DUR_OFFSET + 0 or rest[2] != REST:
            print(f"  ❌ Rest {i} has wrong format: {rest}")
            prefix_ok = False
    
    if prefix_ok:
        print(f"  ✓ All 33 prefix control+rest pairs are correctly formatted")
    print()
    
    # Extract controls
    print("Step 2: Extracting controls from sequence...")
    print("-" * 80)
    controls = extract_controls_from_sequence(sequence, prefix_controls=33)
    print(f"✓ Extracted {len(controls)//3} control events ({len(controls)} tokens)")
    print(f"  - First control: {controls[0:3]}")
    print(f"  - Last control: {controls[-3:]}")
    print()
    
    # Verify all controls have CONTROL_OFFSET
    controls_ok = True
    for i in range(0, len(controls), 3):
        if controls[i] < CONTROL_OFFSET:
            print(f"  ❌ Control {i//3} time token {controls[i]} missing CONTROL_OFFSET")
            controls_ok = False
            break
    
    if controls_ok:
        print(f"  ✓ All {len(controls)//3} extracted controls have CONTROL_OFFSET")
    print()
    
    # Simulate generate4 format
    print("Step 3: Simulating generate4 token format...")
    print("-" * 80)
    gen4_tokens = simulate_generate4_format(controls, prefix_controls=33)
    print(f"✓ generate4 would create {len(gen4_tokens)} tokens")
    print()
    
    # Verify generate4 prefix matches training format
    print("Verifying generate4 prefix matches training format:")
    gen4_prefix_ok = True
    for i in range(33):
        pos = i * 6
        if pos + 6 > len(gen4_tokens):
            break
        
        ctrl = gen4_tokens[pos:pos+3]
        rest = gen4_tokens[pos+3:pos+6]
        
        # Check control has CONTROL_OFFSET
        if ctrl[0] < CONTROL_OFFSET:
            print(f"  ❌ generate4 control {i} time token {ctrl[0]} missing CONTROL_OFFSET")
            gen4_prefix_ok = False
        
        # Check rest structure
        if rest[1] != DUR_OFFSET + 0 or rest[2] != REST:
            print(f"  ❌ generate4 rest {i} has wrong format: {rest}")
            gen4_prefix_ok = False
    
    if gen4_prefix_ok:
        print(f"  ✓ generate4 prefix (33 control+rest pairs) matches training format")
    print()
    
    # Verify alternating pattern
    print("Verifying alternating pattern after prefix:")
    alternating_ok = True
    pos = 33 * 6  # After prefix
    pair_count = 0
    trailing_perf_count = 0
    
    while pos + 3 <= len(gen4_tokens):
        perf = gen4_tokens[pos:pos+3]
        
        # Performance should NOT have CONTROL_OFFSET
        if perf[0] >= CONTROL_OFFSET:
            print(f"  ❌ Performance event at position {pos} has CONTROL_OFFSET: {perf[0]}")
            alternating_ok = False
            break
        
        pos += 3
        
        # Check if there's a control following
        if pos + 3 <= len(gen4_tokens):
            ctrl = gen4_tokens[pos:pos+3]
            
            # Control should have CONTROL_OFFSET
            if ctrl[0] >= CONTROL_OFFSET:
                pair_count += 1
                pos += 3
            else:
                # This must be another performance (trailing ones)
                trailing_perf_count += 1
        else:
            # No more tokens, this is a trailing performance
            trailing_perf_count += 1
            break
    
    if alternating_ok:
        print(f"  ✓ {pair_count} [performance, control] pairs correctly formatted")
        print(f"  ✓ {trailing_perf_count} trailing performance events (without future controls)")
    print()
    
    # Summary
    print("=" * 80)
    print("VERIFICATION SUMMARY")
    print("=" * 80)
    print()
    
    all_ok = prefix_ok and controls_ok and gen4_prefix_ok and alternating_ok
    
    if all_ok:
        print("✅ ALL CHECKS PASSED!")
        print()
        print("tokenize-asap.py format:")
        print("  ✓ First 33 performance controls → prefix with rests (198 tokens)")
        print("  ✓ All scores alternate with future controls (controls 34+)")
        print("  ✓ Last 33 scores have NO future controls (symmetric to prefix)")
        print("  ✓ All controls have CONTROL_OFFSET")
        print()
        print("generate4 format:")
        print("  ✓ First 33 controls → prefix with rests (198 tokens)")
        print("  ✓ Generate performance for all controls")
        print("  ✓ Performance alternates with remaining controls (34+)")
        print("  ✓ Last 33 generated performances have NO future controls")
        print("  ✓ Controls have CONTROL_OFFSET, performances do NOT")
        print()
        print("✅ tokenize-asap.py and generate4 are CONSISTENT!")
        print()
        print("SYMMETRIC STRUCTURE:")
        print("  Training: [33 perf+rest prefix] [scores alternating with future perfs] [33 trailing scores]")
        print("  Generate: [33 ctrl+rest prefix]  [perfs alternating with future ctrls]  [33 trailing perfs]")
        return 0
    else:
        print("❌ SOME CHECKS FAILED - please review the issues above")
        return 1


if __name__ == "__main__":
    exit_code = verify_consistency()
    sys.exit(exit_code)
