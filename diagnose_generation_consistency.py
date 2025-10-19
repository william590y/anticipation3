"""
Diagnostic script to verify generate4 is consistent with tokenization format.

This will:
1. Load a test sequence
2. Extract controls (as generate4 would receive them)
3. Simulate what generate4 builds (without model inference)
4. Compare the structure to training format
5. Identify any discrepancies
"""

from anticipation.config import *
from anticipation.vocab import *


def extract_controls_from_sequence(sequence_tokens, prefix_controls=33):
    """Extract controls exactly as test.py does."""
    if len(sequence_tokens) < 4:
        return []
    
    tokens = sequence_tokens[4:]  # Skip ANTICIPATE + 3 SEPs
    controls = []
    i = 0
    
    # Extract prefix controls
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


def simulate_generate4_structure(controls, prefix_controls=33):
    """
    Simulate the structure that generate4 creates (without actual model inference).
    Returns the prefix structure and info about what should follow.
    """
    # Shift controls to start from time 0
    first_arrival = controls[0] - CONTROL_OFFSET
    controls_shifted = controls.copy()
    for i in range(0, len(controls), 3):
        controls_shifted[i] = controls[i] - first_arrival
    
    tokens = []
    
    # Build prefix with control+rest pairs
    k = min(prefix_controls, len(controls) // 3)
    for i in range(k):
        ctrl = controls_shifted[i*3:i*3+3]
        tokens.extend(ctrl)
        cc_time = ctrl[0] - CONTROL_OFFSET
        tokens.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])
    
    # Remaining controls for alternating
    remaining_controls = controls_shifted[k*3:]
    
    return {
        'prefix_tokens': tokens,
        'prefix_size': len(tokens),
        'prefix_controls': k,
        'remaining_controls': remaining_controls,
        'remaining_count': len(remaining_controls) // 3,
        'controls_shifted': controls_shifted
    }


def analyze_training_sequence(sequence_tokens):
    """Analyze the structure of a training sequence."""
    if len(sequence_tokens) < 4:
        return None
    
    mode_token = sequence_tokens[0]
    seps = sequence_tokens[1:4]
    tokens = sequence_tokens[4:]
    
    # Analyze prefix
    prefix_controls = []
    prefix_rests = []
    i = 0
    
    for _ in range(33):
        if i + 6 <= len(tokens):
            ctrl = tokens[i:i+3]
            rest = tokens[i+3:i+6]
            
            if ctrl[0] >= CONTROL_OFFSET:
                prefix_controls.append(ctrl)
                prefix_rests.append(rest)
                i += 6
            else:
                break
        else:
            break
    
    # Analyze alternating section
    alternating_scores = []
    alternating_controls = []
    
    while i + 3 <= len(tokens):
        triplet = tokens[i:i+3]
        
        if triplet[0] >= CONTROL_OFFSET:
            alternating_controls.append(triplet)
        else:
            alternating_scores.append(triplet)
        
        i += 3
    
    return {
        'mode_token': mode_token,
        'seps': seps,
        'prefix_controls': prefix_controls,
        'prefix_rests': prefix_rests,
        'alternating_scores': alternating_scores,
        'alternating_controls': alternating_controls,
        'tokens_analyzed': i + 4
    }


def main():
    print("=" * 80)
    print("GENERATE4 CONSISTENCY DIAGNOSTIC")
    print("=" * 80)
    print()
    
    # Load first test sequence
    with open('data/test_output.txt', 'r') as f:
        line = f.readline()
    
    sequence_tokens = [int(tok) for tok in line.strip().split()]
    
    print(f"Loaded test sequence with {len(sequence_tokens)} tokens")
    print()
    
    # Analyze training sequence structure
    print("=" * 80)
    print("STEP 1: ANALYZE TRAINING SEQUENCE STRUCTURE")
    print("=" * 80)
    
    training_analysis = analyze_training_sequence(sequence_tokens)
    
    print(f"Mode token: {training_analysis['mode_token']}")
    print(f"  ANTICIPATE = {ANTICIPATE}")
    print(f"  Match: {training_analysis['mode_token'] == ANTICIPATE}")
    print()
    
    print(f"Prefix controls: {len(training_analysis['prefix_controls'])}")
    print(f"Prefix rests: {len(training_analysis['prefix_rests'])}")
    print()
    
    print(f"Alternating scores: {len(training_analysis['alternating_scores'])}")
    print(f"Alternating controls: {len(training_analysis['alternating_controls'])}")
    print()
    
    # Extract controls as generate4 would receive them
    print("=" * 80)
    print("STEP 2: EXTRACT CONTROLS (as generate4 receives them)")
    print("=" * 80)
    
    extracted_controls = extract_controls_from_sequence(sequence_tokens, prefix_controls=33)
    
    print(f"Extracted {len(extracted_controls) // 3} control events ({len(extracted_controls)} tokens)")
    print()
    
    # Check: should match prefix + alternating controls
    expected_controls = len(training_analysis['prefix_controls']) + len(training_analysis['alternating_controls'])
    print(f"Expected controls: {expected_controls}")
    print(f"Extracted controls: {len(extracted_controls) // 3}")
    print(f"Match: {expected_controls == len(extracted_controls) // 3}")
    print()
    
    # Simulate generate4 structure
    print("=" * 80)
    print("STEP 3: SIMULATE GENERATE4 STRUCTURE")
    print("=" * 80)
    
    gen4_structure = simulate_generate4_structure(extracted_controls, prefix_controls=33)
    
    print(f"Prefix size: {gen4_structure['prefix_size']} tokens")
    print(f"  ({gen4_structure['prefix_controls']} controls × 6 tokens each)")
    print()
    print(f"Remaining controls for alternating: {gen4_structure['remaining_count']}")
    print()
    
    # Compare structures
    print("=" * 80)
    print("STEP 4: COMPARE STRUCTURES")
    print("=" * 80)
    print()
    
    print("Training format:")
    print(f"  Prefix: {len(training_analysis['prefix_controls'])} controls + {len(training_analysis['prefix_rests'])} rests")
    print(f"  Body: {len(training_analysis['alternating_scores'])} scores alternating with {len(training_analysis['alternating_controls'])} controls")
    print()
    
    print("Generate4 format:")
    print(f"  Prefix: {gen4_structure['prefix_controls']} controls + rests")
    print(f"  Body: Should generate {len(extracted_controls) // 3} performances alternating with {gen4_structure['remaining_count']} controls")
    print()
    
    # Verify the controls are identical
    print("=" * 80)
    print("STEP 5: VERIFY CONTROL IDENTITY")
    print("=" * 80)
    print()
    
    # Check first 3 controls
    print("First 3 controls from training sequence:")
    for i in range(min(3, len(training_analysis['prefix_controls']))):
        ctrl = training_analysis['prefix_controls'][i]
        # Remove CONTROL_OFFSET to get raw values
        time_raw = ctrl[0] - CONTROL_OFFSET
        dur_raw = ctrl[1] - DUR_OFFSET - CONTROL_OFFSET  # CONTROL_OFFSET added to all elements
        note_raw = ctrl[2] - NOTE_OFFSET - CONTROL_OFFSET
        print(f"  Control {i}: time={time_raw}, dur={dur_raw}, note={note_raw}")
    print()
    
    print("First 3 extracted controls:")
    for i in range(min(3, len(extracted_controls) // 3)):
        ctrl_time = extracted_controls[i*3]
        ctrl_dur = extracted_controls[i*3 + 1]
        ctrl_note = extracted_controls[i*3 + 2]
        # Remove offsets
        time_raw = ctrl_time - CONTROL_OFFSET
        dur_raw = ctrl_dur - DUR_OFFSET - CONTROL_OFFSET
        note_raw = ctrl_note - NOTE_OFFSET - CONTROL_OFFSET
        print(f"  Control {i}: time={time_raw}, dur={dur_raw}, note={note_raw}")
    print()
    
    print("First 3 controls after generate4 shifts to time 0:")
    for i in range(min(3, len(gen4_structure['controls_shifted']) // 3)):
        ctrl_time = gen4_structure['controls_shifted'][i*3]
        ctrl_dur = gen4_structure['controls_shifted'][i*3 + 1]
        ctrl_note = gen4_structure['controls_shifted'][i*3 + 2]
        # Remove offsets
        time_raw = ctrl_time - CONTROL_OFFSET
        dur_raw = ctrl_dur - DUR_OFFSET - CONTROL_OFFSET
        note_raw = ctrl_note - NOTE_OFFSET - CONTROL_OFFSET
        print(f"  Control {i}: time={time_raw}, dur={dur_raw}, note={note_raw}")
    print()
    
    # Check what model should generate
    print("=" * 80)
    print("STEP 6: WHAT SHOULD THE MODEL GENERATE?")
    print("=" * 80)
    print()
    
    print("In training, the model sees PREFIX and generates:")
    print(f"  {len(training_analysis['alternating_scores'])} score events")
    print()
    
    print("In inference, the model sees PREFIX and should generate:")
    print(f"  {len(extracted_controls) // 3} performance events")
    print()
    
    print("Expected scores from training (first 3):")
    for i in range(min(3, len(training_analysis['alternating_scores']))):
        score = training_analysis['alternating_scores'][i]
        time_raw = score[0] - TIME_OFFSET
        dur_raw = score[1] - DUR_OFFSET
        note_raw = score[2] - NOTE_OFFSET
        print(f"  Score {i}: time={time_raw}, dur={dur_raw}, note={note_raw}")
    print()
    
    # Check if alternating controls match
    print("=" * 80)
    print("STEP 7: VERIFY ALTERNATING CONTROL CORRESPONDENCE")
    print("=" * 80)
    print()
    
    print("Training: Scores are matched with controls")
    print(f"  Number of scores: {len(training_analysis['alternating_scores'])}")
    print(f"  Number of alternating controls: {len(training_analysis['alternating_controls'])}")
    print()
    
    print("Checking if scores should match their corresponding controls...")
    # The controls in the alternating section are the "future" controls
    # They correspond to performances that come LATER in the sequence
    # So score_i should NOT necessarily match alternating_control_i
    
    print()
    print("CRITICAL QUESTION:")
    print("  In training: score_i appears at position i in alternating section")
    print("  In training: control_j appears at position i in alternating section")
    print("  Are score_i and control_j the SAME musical event?")
    print()
    
    # Check the correspondence
    if len(training_analysis['alternating_scores']) > 0 and len(training_analysis['alternating_controls']) > 0:
        score0 = training_analysis['alternating_scores'][0]
        ctrl0 = training_analysis['alternating_controls'][0]
        
        score_note = score0[2] - NOTE_OFFSET
        ctrl_note = ctrl0[2] - NOTE_OFFSET - CONTROL_OFFSET
        
        print(f"First alternating score note: {score_note}")
        print(f"First alternating control note: {ctrl_note}")
        print(f"Do they match? {score_note == ctrl_note}")
        print()
    
    print("=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print()
    print("If alternating score notes DON'T match alternating control notes:")
    print("  → This is BY DESIGN (controls are FUTURE performances)")
    print("  → Generate4 should generate based on PREFIX controls")
    print("  → Generated outputs won't necessarily match ALL controls")
    print()
    print("If we want generated notes to match controls:")
    print("  → We need to understand which control corresponds to which output")
    print("  → The prefix controls provide context")
    print("  → The remaining controls are revealed during generation")
    print()


if __name__ == "__main__":
    main()
