"""
Understand the EXACT tokenization format by carefully reading the code.
"""

print("=" * 80)
print("TOKENIZATION FORMAT (from tokenize-asap.py)")
print("=" * 80)
print()

print("Code:")
print("""
for i, t in enumerate(matched_tuples):
    sc = t[2]  # score triplet
    if sc[0] is not None:
        interleaved_tokens.extend(sc)
    ii = i + k
    if ii < len(matched_tuples):
        interleaved_tokens.extend(matched_tuples[ii][0])  # future control
""")
print()

print("What this does:")
print("For each matched tuple i from 0 to N-1:")
print("  1. Add score_i (if not None)")
print("  2. If i+k < N: add ctrl_(i+k)")
print()

print("Example with N=171, k=33:")
print("  i=0: score_0, ctrl_33")
print("  i=1: score_1, ctrl_34")
print("  ...")
print("  i=137: score_137, ctrl_170")
print("  i=138: score_138, (no ctrl since 138+33=171 >= 171)")
print("  ...")
print("  i=170: score_170, (no ctrl)")
print()

print("So the body IS:")
print("  [score_0, ctrl_33, score_1, ctrl_34, ..., score_137, ctrl_170, score_138, ..., score_170]")
print()

print("But our test data shows:")
print("  [ctrl_66, score_67, ctrl_68, ...]")
print()

print("This means position 66 is ctrl_33!")
print("Let me verify by checking if 136 scores in body matches...")
print()

N_controls_extracted = 171
k = 33

print(f"If we have {N_controls_extracted} controls total:")
print(f"  - {k} in prefix")
print(f"  - {N_controls_extracted - k} = {N_controls_extracted - k} in body (these are ctrl_{k} to ctrl_{N_controls_extracted-1})")
print()

print("According to tokenization logic:")
print(f"  - Scores i=0 to {N_controls_extracted-1} should be added")
print(f"  - Controls added when i+{k} < {N_controls_extracted}, i.e., i < {N_controls_extracted-k}")
print(f"  - So {N_controls_extracted - k} future controls are added (ctrl_{k} to ctrl_{N_controls_extracted-1})")
print()

print("Body structure:")
print(f"  - For i=0 to {N_controls_extracted-k-1}: [score_i, ctrl_(i+{k})]")
print(f"  - For i={N_controls_extracted-k} to {N_controls_extracted-1}: [score_i] (no future ctrl)")
print(f"  - Total: {N_controls_extracted-k} score+ctrl pairs + {k} trailing scores")
print(f"  - Total scores: {N_controls_extracted}")
print(f"  - Total controls in body: {N_controls_extracted-k}")
print()

print("But test data has 136 scores, not 171!")
print("This means not all matched tuples had valid scores (some were None)")
