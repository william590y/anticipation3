# Diagnostic Summary: Why is Generation Quality Poor?

## Date: 2025-10-17

## Training Verification ✅

**Training did converge:**
- Training loss: 8.08 → 1.09 (good reduction)
- Validation loss: plateaued at ~1.26-1.28 (no overfitting)
- 3500 steps, effective batch size 1024

**Format consistency verified:**
- Tokenization: `[ANTICIPATE, SEP, SEP, SEP][prefix: ctrl+rest×33][body: score, future_ctrl alternating][trailing: scores]`
- Generation: Matches training format exactly ✅
- Testing: Properly extracts and compares ✅

## Generation Quality Results ❌

**Normal Generation:**
- Timing match: 0.69%
- Duration match: 3.91%
- Pitch match: 11.72%
- **Essentially random output**

**Rejection Sampling (force correct pitch):**
- Timing match: 0.73% (no improvement!)
- Duration match: 0.00% (got worse!)
- Pitch match: 64.0% (after avg 8.4 attempts, 36% failed even after 20 tries)
- **Proves model didn't learn timing/duration patterns**

## Root Cause Analysis

### What the model learned:
1. **Language modeling objective** - It minimized cross-entropy loss
2. **Token distribution** - It can generate plausible tokens
3. **Some pitch patterns** - 41.9% correct pitch on first try

### What the model DIDN'T learn:
1. **Conditional generation** - Given control tokens, generate matching score
2. **Timing patterns** - 0% timing match even with forced pitch
3. **Duration patterns** - 0-4% duration match
4. **Task structure** - Doesn't understand the anticipation task

## Hypothesis: Training Objective Mismatch

The model was trained with **causal language modeling loss** on the full sequence:
```
Loss = CrossEntropy(predicted_tokens, actual_tokens)
```

This treats ALL tokens equally:
- Loss on ANTICIPATE token
- Loss on SEP tokens  
- Loss on prefix controls (ctrl_0 to ctrl_32)
- Loss on REST tokens
- Loss on scores (what we care about!)
- Loss on future controls

**Problem**: The model is being asked to predict EVERYTHING, including:
- Controls (which are GIVEN as input in the real task)
- REST tokens (which are deterministic)
- SEP tokens (which are fixed)

Only ~33% of tokens are actually scores we want to generate!

## Proposed Solutions

### Option 1: Masked Training Loss ⭐ RECOMMENDED
Only compute loss on score tokens, mask out control/rest/sep tokens:

```python
# In training loop
labels = input_ids.clone()
for i in range(len(labels)):
    for j in range(len(labels[i])):
        token = labels[i][j]
        if token >= CONTROL_OFFSET or token == REST or token == SEPARATOR:
            labels[i][j] = -100  # Ignore in loss
```

This forces the model to learn: "given controls, predict scores"

### Option 2: Separate Encoder-Decoder Architecture
Use encoder for controls, decoder for scores. More complex, requires rewriting.

### Option 3: Better Prompt Engineering
During training, explicitly separate "input" (controls) from "output" (scores) with special tokens.

### Option 4: Increase Training
Current: 3500 steps. Maybe needs 10k-20k steps? But loss already plateaued, so unlikely to help.

## Recommendation

**RETRAIN with masked loss (Option 1)** because:
1. Simple to implement (just modify train.py)
2. Directly addresses the problem
3. Model architecture stays the same
4. Should significantly improve generation quality

Would you like me to:
1. Implement masked loss training?
2. Start retraining?
3. First verify this hypothesis with a smaller experiment?
