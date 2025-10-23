"""Verify training script compatibility with new tokenization"""
from anticipation.vocab import VOCAB_SIZE, ANTICIPATE, SEPARATOR, MASK
from transformers import AutoConfig

print("=" * 60)
print("TOKENIZATION & TRAINING COMPATIBILITY CHECK")
print("=" * 60)

# Check vocabulary
print("\n1. VOCABULARY:")
print(f"   Current VOCAB_SIZE: {VOCAB_SIZE}")
print(f"   ANTICIPATE token: {ANTICIPATE}")
print(f"   SEPARATOR token: {SEPARATOR}")
print(f"   MASK token: {MASK}")

# Check model config
print("\n2. PRE-TRAINED MODEL:")
config = AutoConfig.from_pretrained('stanford-crfm/music-medium-800k', trust_remote_code=True)
print(f"   Expected vocab_size: {config.vocab_size}")

# Compatibility check
print("\n3. COMPATIBILITY:")
if config.vocab_size == VOCAB_SIZE:
    print("   ✓ Vocabulary sizes match!")
elif config.vocab_size == VOCAB_SIZE - 1:
    print(f"   ⚠ Model expects {config.vocab_size}, we have {VOCAB_SIZE}")
    print(f"   → Need to resize model embeddings to add MASK token")
    print(f"   → This will add 1 new token embedding (MASK at position {MASK})")
else:
    print(f"   ✗ Vocabulary size mismatch!")
    print(f"   → Model: {config.vocab_size}, Ours: {VOCAB_SIZE}")

# Check tokenization format
print("\n4. TOKENIZATION FORMAT:")
print("   New format: [ANTICIPATE, control_tokens..., score_tokens..., PAD...]")
print("   Old format: [SEP, SEP, SEP, control_flag, tokens...]")
print("   → Training script validation needs update")

print("\n5. REQUIRED CHANGES:")
print("   A. Model resizing:")
print("      model.resize_token_embeddings(VOCAB_SIZE)")
print("   B. Update TokenizedDataset validation:")
print("      - Check sample[0] == ANTICIPATE (not SEP, SEP, SEP)")
print("   C. Ensure loss masking handles MASK tokens correctly")

print("\n" + "=" * 60)
