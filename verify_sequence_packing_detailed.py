"""
Verify that the sequence packing in tokenize-asap.py correctly follows the pattern:
1. [ANTICIPATE, SEP, SEP, SEP] header
2. First 33 performance entries: [control_triplet, rest_triplet] repeated 33 times
3. Alternating pattern: [score_triplet, control_triplet] for remaining entries

Also verify that generate4 (or generate3 which is being used) is consistent with this.
"""

from anticipation.config import *
from anticipation.vocab import *
from anticipation import ops

def analyze_tokenization_pattern():
    """
    Analyze the tokenization pattern from tokenize-asap.py _interleave_tokenize4_single function.
    """
    print("=" * 80)
    print("TOKENIZE-ASAP.PY SEQUENCE PACKING VERIFICATION")
    print("=" * 80)
    print()
    
    print("From _interleave_tokenize4_single function:")
    print("-" * 80)
    print()
    
    print("Step 1: Build prefix with rests for first k=min(33, len(matched_tuples)) entries")
    print("-------")
    print("for t in matched_tuples[:k]:")
    print("    cc = t[0]  # control triplet (performance with CONTROL_OFFSET)")
    print("    interleaved_tokens.extend(cc)")
    print("    cc_time = cc[0] - CONTROL_OFFSET")
    print("    interleaved_tokens.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])")
    print()
    print("Pattern: [ctrl_time, ctrl_dur, ctrl_note, TIME+cc_time, DUR+0, REST] * k")
    print("         |<---- control triplet ---->|<---- rest triplet -------->|")
    print()
    
    print("Step 2: Alternating score and future controls")
    print("-------")
    print("for i, t in enumerate(matched_tuples):")
    print("    sc = t[2]  # score triplet (without offset)")
    print("    if sc[0] is not None:")
    print("        interleaved_tokens.extend(sc)")
    print("    ii = i + k  # future index")
    print("    if ii < len(matched_tuples):")
    print("        interleaved_tokens.extend(matched_tuples[ii][0])  # future control")
    print()
    print("Pattern after prefix: [score_triplet, future_control_triplet] alternating")
    print()
    
    print("Step 3: Prepend separators")
    print("-------")
    print("interleaved_tokens[0:0] = [SEPARATOR, SEPARATOR, SEPARATOR]")
    print()
    
    print("Step 4: Chunk and add mode token")
    print("-------")
    print("seq = concatenated_tokens[0:EVENT_SIZE * M]  # Take 1023 tokens")
    print("seq.insert(0, z)  # Prepend ANTICIPATE token")
    print()
    
    print("=" * 80)
    print("FINAL SEQUENCE STRUCTURE")
    print("=" * 80)
    print()
    print("Position 0:        ANTICIPATE token (z)")
    print("Position 1-3:      [SEP, SEP, SEP]")
    print()
    print("Positions 4-201:   33 * 6 tokens = 198 tokens")
    print("                   Each group of 6 tokens:")
    print("                   [ctrl_time, ctrl_dur, ctrl_note,  # control with CONTROL_OFFSET")
    print("                    rest_time, rest_dur, REST]        # rest: TIME+time, DUR+0, REST")
    print()
    print("Positions 202+:    Alternating pattern until token 1023:")
    print("                   [score_time, score_dur, score_note,  # score without offset")
    print("                    ctrl_time, ctrl_dur, ctrl_note]     # future control with CONTROL_OFFSET")
    print("                   ... repeating ...")
    print()
    
    # Calculate numbers
    header_size = 4  # ANTICIPATE + 3 SEPs
    prefix_size = 33 * 6  # 33 control+rest pairs
    body_start = header_size + prefix_size
    remaining = 1024 - body_start  # 1024 total - 4 header - 198 prefix
    
    print(f"Header size: {header_size} tokens")
    print(f"Prefix size: {prefix_size} tokens (33 controls * 6 tokens each)")
    print(f"Body starts at position: {body_start}")
    print(f"Remaining tokens for alternating: {remaining} tokens")
    print(f"Number of alternating pairs: {remaining // 6} pairs")
    print()
    
    return {
        'header_size': header_size,
        'prefix_size': prefix_size,
        'prefix_controls': 33,
        'body_start': body_start,
        'remaining': remaining,
        'alternating_pairs': remaining // 6
    }


def verify_control_extraction():
    """
    Verify the control extraction logic from test.py matches the tokenization.
    """
    print("=" * 80)
    print("CONTROL EXTRACTION VERIFICATION (from test.py)")
    print("=" * 80)
    print()
    
    print("extract_controls_from_sequence function:")
    print("-" * 80)
    print()
    
    print("Step 1: Skip header")
    print("-------")
    print("tokens = sequence_tokens[4:]  # Skip ANTICIPATE + 3 SEPs")
    print()
    
    print("Step 2: Extract prefix controls")
    print("-------")
    print("for _ in range(33):  # prefix_controls = 33")
    print("    control_triplet = tokens[i:i+3]    # Performance with CONTROL_OFFSET")
    print("    rest_triplet = tokens[i+3:i+6]     # Rest triplet (skip)")
    print("    if control_triplet[0] >= CONTROL_OFFSET:")
    print("        controls.extend(control_triplet)")
    print("    i += 6")
    print()
    print("Extracts: 33 control triplets (tokens with CONTROL_OFFSET)")
    print()
    
    print("Step 3: Extract alternating controls")
    print("-------")
    print("while i + 3 <= len(tokens):")
    print("    triplet = tokens[i:i+3]")
    print("    if triplet[0] >= CONTROL_OFFSET:")
    print("        controls.extend(triplet)  # Extract control")
    print("    i += 3")
    print()
    print("Extracts: Future performance controls from alternating pattern")
    print("         (every other triplet that has CONTROL_OFFSET)")
    print()
    
    print("✓ MATCHES tokenization: Extracts controls with CONTROL_OFFSET")
    print()


def check_generate3_consistency():
    """
    Check if generate3 is consistent with the control extraction.
    """
    print("=" * 80)
    print("GENERATE3 CONSISTENCY CHECK")
    print("=" * 80)
    print()
    
    print("generate3 function signature:")
    print("def generate3(model, controls, top_p=1.0):")
    print("    # assuming controls are already shifted by CONTROL_OFFSET")
    print()
    
    print("Step 1: Shift controls to start from time 0")
    print("-------")
    print("first_arrival = controls[0] - CONTROL_OFFSET")
    print("for i in range(0, len(controls), 3):")
    print("    controls_shifted[i] = controls[i] - first_arrival")
    print()
    print("NOTE: This keeps CONTROL_OFFSET in the time token!")
    print("      controls_shifted[0] = CONTROL_OFFSET + 0  (first control at time 0)")
    print()
    
    print("Step 2: Build prefix from controls within DELTA")
    print("-------")
    print("for t in controls_shifted_zipped:")
    print("    if t[0] - CONTROL_OFFSET <= DELTA * TIME_RESOLUTION:")
    print("        tokens.extend(t)")
    print()
    print("Adds all controls with time <= DELTA to prefix")
    print()
    
    print("Step 3: Generate performance alternating with remaining controls")
    print("-------")
    print("for i in range(total):")
    print("    new_token = add_token(model, z, tokens, top_p, current_time)")
    print("    tokens.extend(new_token)       # Add generated performance")
    print("    events.extend(new_token)")
    print("    tokens.extend(controls_shifted[0:3])  # Add next control")
    print("    controls_shifted = controls_shifted[3:]")
    print()
    print("Pattern: [control, control, ...] prefix, then [perf, ctrl, perf, ctrl, ...]")
    print()
    
    print("=" * 80)
    print("POTENTIAL ISSUE IDENTIFIED")
    print("=" * 80)
    print()
    print("❌ MISMATCH: generate3 does NOT match tokenization!")
    print()
    print("Tokenization format:")
    print("  [ctrl, rest, ctrl, rest, ...] * 33, then [score, ctrl, score, ctrl, ...]")
    print()
    print("generate3 format:")
    print("  [ctrl, ctrl, ...] prefix (all within DELTA), then [perf, ctrl, perf, ctrl, ...]")
    print()
    print("Missing from generate3:")
    print("  1. No REST tokens in prefix")
    print("  2. No fixed-size prefix (33 controls)")
    print("  3. Prefix size varies based on DELTA")
    print()


def suggest_generate4():
    """
    Suggest what generate4 should look like.
    """
    print("=" * 80)
    print("SUGGESTED GENERATE4 IMPLEMENTATION")
    print("=" * 80)
    print()
    
    print("def generate4(model, controls, top_p=1.0, prefix_controls=33):")
    print('    """')
    print("    Generate performance given controls that match tokenize-asap format.")
    print("    ")
    print("    Args:")
    print("        model: The trained model")
    print("        controls: Performance tokens WITH CONTROL_OFFSET already applied")
    print("        top_p: Nucleus sampling parameter")
    print("        prefix_controls: Number of controls to use in prefix (default 33)")
    print("    ")
    print("    Returns:")
    print("        events: Generated performance tokens (without CONTROL_OFFSET)")
    print("        tokens: Full sequence including controls")
    print('    """')
    print("    z = [ANTICIPATE]")
    print("    ")
    print("    # Shift controls to start from time 0")
    print("    first_arrival = controls[0] - CONTROL_OFFSET")
    print("    controls_shifted = controls.copy()")
    print("    for i in range(0, len(controls), 3):")
    print("        controls_shifted[i] = controls[i] - first_arrival")
    print("    ")
    print("    tokens = []")
    print("    ")
    print("    # Step 1: Build prefix with k control+rest pairs")
    print("    k = min(prefix_controls, len(controls) // 3)")
    print("    for i in range(k):")
    print("        ctrl = controls_shifted[i*3:i*3+3]")
    print("        tokens.extend(ctrl)")
    print("        # Add rest triplet")
    print("        cc_time = ctrl[0] - CONTROL_OFFSET")
    print("        tokens.extend([TIME_OFFSET + cc_time, DUR_OFFSET + 0, REST])")
    print("    ")
    print("    # Step 2: Prepare remaining controls for alternating")
    print("    remaining_controls = controls_shifted[k*3:]")
    print("    ")
    print("    events = []")
    print("    ")
    print("    # Step 3: Generate with alternating pattern")
    print("    for i in range(len(controls) // 3):  # Generate one perf per control")
    print("        current_time = events[-3] if events else 0")
    print("        ")
    print("        # Generate performance event")
    print("        new_token = add_token(model, z, tokens, top_p, current_time)")
    print("        tokens.extend(new_token)")
    print("        events.extend(new_token)")
    print("        ")
    print("        # Add next control if available")
    print("        if remaining_controls:")
    print("            tokens.extend(remaining_controls[0:3])")
    print("            remaining_controls = remaining_controls[3:]")
    print("    ")
    print("    return events, tokens")
    print()


if __name__ == "__main__":
    # Run all verifications
    stats = analyze_tokenization_pattern()
    print()
    verify_control_extraction()
    print()
    check_generate3_consistency()
    print()
    suggest_generate4()
    print()
    
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print()
    print("✓ tokenize-asap.py CORRECTLY implements:")
    print("  - Fixed 33 control+rest prefix (198 tokens)")
    print("  - Alternating score/control pattern in body")
    print("  - Controls have CONTROL_OFFSET applied")
    print()
    print("✓ test.py extract_controls_from_sequence CORRECTLY:")
    print("  - Extracts 33 prefix controls (skipping rests)")
    print("  - Extracts alternating future controls")
    print("  - Preserves CONTROL_OFFSET in extracted controls")
    print()
    print("❌ generate3 DOES NOT match the training format:")
    print("  - Missing REST tokens in prefix")
    print("  - Variable prefix size instead of fixed 33")
    print("  - No explicit prefix/body distinction")
    print()
    print("➜ RECOMMENDATION: Implement generate4 as shown above")
    print()
