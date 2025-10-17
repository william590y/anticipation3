"""
Verify generation logic matches training format.
"""

# Simulate the generation process
k = 33  # prefix controls
total = 171  # total control events

print("GENERATION PROCESS:")
print("==================\n")

print("PREFIX (first 33 controls):")
print("  Format: [ctrl0, rest, ctrl1, rest, ..., ctrl32, rest]")
print(f"  Uses controls 0 through {k-1}\n")

print("ALTERNATING SECTION:")
print("  Loop i from 0 to", total-1)
print("  Format built: [score_i, ctrl_(i+k), score_(i+1), ctrl_(i+k+1), ...]")
print()

print("First 10 iterations:")
for i in range(min(10, total)):
    ctrl_idx = i + k
    print(f"  i={i:3d}: Generate score_{i:3d}, then add ctrl_{ctrl_idx:3d}")

print("\n" + "="*60)
print("TRAINING DATA FORMAT (from check_format.py):")
print("="*60)
print("PREFIX: [ctrl0, rest, ctrl1, rest, ..., ctrl32, rest]")
print("ALTERNATING: [ctrl33, score0, ctrl34, score1, ctrl35, score2, ...]")
print()

print("COMPARISON:")
print("-" * 60)
print("Training alternating:  ctrl_33, score_0, ctrl_34, score_1, ...")
print("Generation alternating: score_0, ctrl_33, score_1, ctrl_34, ...")
print()
print("❌ MISMATCH! Generation order is [SCORE, CTRL] but training is [CTRL, SCORE]!")
print()
print("This means:")
print("  - During training, model sees ctrl_N and predicts score_N")
print("  - During generation, we generate score_N, THEN show ctrl_N")
print("  - This is BACKWARDS!")
