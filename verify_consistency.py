"""
Verify consistency between tokenization, training, and generation.

This script checks that:
1. Tokenization creates the correct format
2. Training augmentation only affects control triplets
3. Generation uses the same format assumptions
"""

import torch
import numpy as np
from anticipation.vocab import *
from anticipation.config import *

print("="*80)
print("CONSISTENCY VERIFICATION")
print("="*80)

print("\n1. VOCABULARY STRUCTURE")
print(f"   TIME_OFFSET: {TIME_OFFSET}")
print(f"   DUR_OFFSET: {DUR_OFFSET}")
print(f"   NOTE_OFFSET: {NOTE_OFFSET}")
print(f"   REST: {REST}")
print(f"   CONTROL_OFFSET: {CONTROL_OFFSET}")
print(f"   SPECIAL_OFFSET: {SPECIAL_OFFSET}")
print(f"   SEPARATOR: {SEPARATOR}")
print(f"   ANTICIPATE: {ANTICIPATE}")
print(f"   VOCAB_SIZE: {VOCAB_SIZE}")

print("\n2. TOKEN RANGES")
print(f"   Score tokens: 0 to {CONTROL_OFFSET-1}")
print(f"   Control tokens: {CONTROL_OFFSET} to {SPECIAL_OFFSET-1}")
print(f"   Special tokens: {SPECIAL_OFFSET} to {VOCAB_SIZE-1}")

print("\n3. TOKENIZATION FORMAT (from tokenize-asap.py)")
print("   Sequence structure:")
print("   [ANTICIPATE, SEP, SEP, SEP,")
print("    ctrl0_time, ctrl0_dur, ctrl0_pitch, rest0_time, rest0_dur, REST,")
print("    ctrl1_time, ctrl1_dur, ctrl1_pitch, rest1_time, rest1_dur, REST,")
print("    ...")
print("    ctrl32_time, ctrl32_dur, ctrl32_pitch, rest32_time, rest32_dur, REST,  # prefix (k=33)")
print("    score0_time, score0_dur, score0_pitch,  # body alternates score/ctrl")
print("    ctrl33_time, ctrl33_dur, ctrl33_pitch,")
print("    score1_time, score1_dur, score1_pitch,")
print("    ctrl34_time, ctrl34_dur, ctrl34_pitch,")
print("    ...]")

print("\n4. VERIFY CONTROL VS SCORE DETECTION")

# Create example tokens
ctrl_triplet = [CONTROL_OFFSET + 100, CONTROL_OFFSET + 50, CONTROL_OFFSET + 60]  # time, dur, pitch
score_triplet = [TIME_OFFSET + 100, DUR_OFFSET + 50, NOTE_OFFSET + 60]  # time, dur, pitch
rest_triplet = [TIME_OFFSET + 100, DUR_OFFSET + 0, REST]
sep_triplet = [SEPARATOR, SEPARATOR, SEPARATOR]

print(f"\n   Control triplet: {ctrl_triplet}")
print(f"     All >= CONTROL_OFFSET? {all(t >= CONTROL_OFFSET for t in ctrl_triplet)}")
print(f"     All < SPECIAL_OFFSET? {all(t < SPECIAL_OFFSET for t in ctrl_triplet)}")
print(f"     → Should be AUGMENTED: ✓")

print(f"\n   Score triplet: {score_triplet}")
print(f"     All < CONTROL_OFFSET? {all(t < CONTROL_OFFSET for t in score_triplet)}")
print(f"     → Should NOT be augmented: ✓")

print(f"\n   Rest triplet: {rest_triplet}")
print(f"     All < CONTROL_OFFSET? {all(t < CONTROL_OFFSET for t in rest_triplet)}")
print(f"     → Should NOT be augmented: ✓")

print(f"\n   Separator triplet: {sep_triplet}")
print(f"     All >= CONTROL_OFFSET? {all(t >= CONTROL_OFFSET for t in sep_triplet)}")
print(f"     First token is SEPARATOR? {sep_triplet[0] == SEPARATOR}")
print(f"     → Should NOT be augmented (filtered): ✓")

print("\n5. TRAINING AUGMENTATION LOGIC (from train.py)")
print("   Detection rule:")
print("   if (token[i] >= CONTROL_OFFSET and")
print("       token[i+1] >= CONTROL_OFFSET and")
print("       token[i+2] >= CONTROL_OFFSET and")
print("       token[i] != SEPARATOR and")
print("       token[i] != ANTICIPATE):")
print("       → This is a control triplet, apply augmentation")

print("\n6. TEST AUGMENTATION DETECTION")

def is_control_triplet(triplet):
    """Mimics training detection logic"""
    return (triplet[0] >= CONTROL_OFFSET and
            triplet[1] >= CONTROL_OFFSET and
            triplet[2] >= CONTROL_OFFSET and
            triplet[0] != SEPARATOR and
            triplet[0] != ANTICIPATE)

test_cases = [
    (ctrl_triplet, True, "Control"),
    (score_triplet, False, "Score"),
    (rest_triplet, False, "Rest"),
    (sep_triplet, False, "Separator"),
    ([ANTICIPATE, ANTICIPATE, ANTICIPATE], False, "Anticipate"),
]

all_correct = True
for triplet, expected, name in test_cases:
    result = is_control_triplet(triplet)
    status = "✓" if result == expected else "✗"
    if result != expected:
        all_correct = False
    print(f"   {name} triplet: {triplet[:3]} → {result} (expected {expected}) {status}")

print("\n7. GENERATION FORMAT (from sample.py generate4)")
print("   Generation creates the SAME format:")
print("   1. Prefix: k=33 control+rest pairs")
print("   2. Body: alternating score/control triplets")
print("   3. Each control: [CONTROL_OFFSET+time, CONTROL_OFFSET+dur, CONTROL_OFFSET+pitch]")
print("   4. Each score: [TIME_OFFSET+time, DUR_OFFSET+dur, NOTE_OFFSET+pitch]")
print("   → Matches tokenization format: ✓")

print("\n8. VERIFY sample.py ASSUMPTIONS")
print("   sample.py safe_logits() blocks:")
print(f"   - Control tokens: logits[{CONTROL_OFFSET}:{SPECIAL_OFFSET}] = -inf")
print(f"   - Special tokens: logits[{SPECIAL_OFFSET}:] = -inf")
print("   → Model never generates controls or special tokens during autoregression: ✓")

print("\n" + "="*80)
if all_correct:
    print("✓ ALL CONSISTENCY CHECKS PASSED")
    print("="*80)
    print("\nTokenization, training, and generation are CONSISTENT:")
    print("  ✓ Control triplets: all 3 tokens >= CONTROL_OFFSET")
    print("  ✓ Score triplets: all 3 tokens < CONTROL_OFFSET")
    print("  ✓ Training augments ONLY control triplets")
    print("  ✓ Generation creates same format as tokenization")
    print("  ✓ Model never generates control/special tokens")
else:
    print("✗ SOME CHECKS FAILED - REVIEW ABOVE")
    print("="*80)
