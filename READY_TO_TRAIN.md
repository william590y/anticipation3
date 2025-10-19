# Masked Loss Training - Ready to Start

## Date: 2025-10-17

## Problem Identified ✅

**Original training** used standard causal LM loss on ALL tokens:
- Loss computed on: headers, controls, rests, scores (100% of tokens)
- Model learned token distributions but not the conditional task
- Result: 0% timing/duration accuracy, random generation

## Solution: Masked Loss Training 🎯

**New training** uses masked loss ONLY on score tokens:
- Loss computed on: score tokens ONLY (~40% of tokens)
- Masked (ignored): headers, separators, controls, rests
- Forces model to learn: "Given controls → Generate scores"

## Verification Complete ✅

### Tokenization Structure
```
[ANTICIPATE, SEP, SEP, SEP]           # Positions 0-3 (MASKED)
[ctrl, rest] × 33                      # Positions 4-201 (MASKED)
[ctrl, score, ctrl, score, ...]       # Positions 202+ (MASK ctrl, PREDICT score)
```

### Masking Statistics
- Total tokens per sequence: 1024
- Masked tokens: 613 (59.9%)
  - Header: 4 tokens
  - Prefix controls + rests: 198 tokens
  - Body controls: 411 tokens (137 triplets)
- **Predicted tokens: 411 (40.1%)**
  - Body scores: 411 tokens (**137 triplets**)

### Implementation
Script: `train_masked.py`
- ✅ Loads sequences from `data/train_output.txt`
- ✅ Creates masked labels (triplet-based)
- ✅ Only computes loss on score triplets
- ✅ Uses same training hyperparameters as original
- ✅ Saves checkpoints every 500 steps
- ✅ Plots losses

## To Start Training

### Command
```bash
python train_masked.py \
    --data_file data/train_output.txt \
    --val_file data/test_output.txt \
    --model_name stanford-crfm/music-medium-800k \
    --output_dir masked_loss_training \
    --batch_size 8 \
    --gradient_accumulation_steps 32 \
    --learning_rate 3e-5 \
    --max_steps 3500 \
    --save_steps 500 \
    --eval_steps 100
```

### Expected Training Time
- Hardware: RTX 4090 (15.99 GB)
- Steps: 3500
- Effective batch size: 256 (8 × 32)
- Estimated: ~2-3 hours

### Expected Results
If masked loss solves the problem:
- ✅ Training loss should converge (as before)
- ✅ Model should learn timing patterns (>50% accuracy)
- ✅ Model should learn duration patterns (>50% accuracy)
- ✅ Pitch accuracy should improve (>30-40%)

## Comparison

### Old Training (Full Loss)
- Loss on all 1024 tokens
- Model learned: token distributions
- Generation quality: **0-4% timing/duration**

### New Training (Masked Loss)
- Loss on 411 score tokens only
- Model learns: conditional generation (control → score)
- Expected generation quality: **>50% timing/duration**

## Next Steps After Training

1. **Test generation quality**:
   ```bash
   python test_generation_quality_corrected.py \
       --checkpoint masked_loss_training/final \
       --num-sequences 20
   ```

2. **Compare with forced pitch**:
   ```bash
   python test_forced_generation.py \
       --checkpoint masked_loss_training/final \
       --num-sequences 10
   ```

3. **If results are good**:
   - Train for more steps (10k-20k) for better quality
   - Test on real generation tasks
   - Compare MIDI outputs

4. **If results are still poor**:
   - May need different architecture (encoder-decoder)
   - May need more context (current: 1017 tokens lookback)
   - May need to rethink the anticipation offset (k=33)

## Ready to Train! 🚀

The training script is verified and ready. Start with:
```bash
python train_masked.py
```

Good luck! 🎵
