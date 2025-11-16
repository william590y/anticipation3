# Model Comparison Results

## Overview

Generated comprehensive evaluation of three models:
- **50_model** (previously sliding_model, baseline)
- **100_model** (trained with 100ms time perturbation)
- **150_model** (trained with 150ms time perturbation)

## Autoregressive Validation Accuracy

Evaluated on 50 sequences from `test_sliding.txt`:

| Model      | Time Accuracy | Duration Accuracy | Pitch Accuracy | Overall |
|------------|---------------|-------------------|----------------|---------|
| **50_model**  | (previously tested ~4.31% on 15 seqs) | N/A | N/A | ~34% |
| **100_model** | **1.05%** (89/8500) | **36.11%** (3069/8500) | **72.95%** (6201/8500) | **36.70%** |
| **150_model** | **0.68%** (58/8500) | **30.68%** (2608/8500) | **63.68%** (5413/8500) | **31.68%** |

### Key Findings:

1. **150_model performs WORSE than 100_model** across all metrics
   - More perturbation hurt rather than helped
   - Time: 1.05% → 0.68% (worse)
   - Duration: 36.11% → 30.68% (worse)
   - Pitch: 72.95% → 63.68% (worse)

2. **Time prediction catastrophically bad** for both new models (<1%)
   - Same fundamental problem as baseline
   - Time perturbation augmentation did NOT solve exposure bias

3. **Relative performance ordering**: Pitch > Duration >>> Time
   - Pitch: 63-73% (moderate)
   - Duration: 31-36% (poor)
   - Time: 0.68-1.05% (catastrophic)

## Generated Outputs

### MIDI Examples (5 per model)

Location: `model_comparison_outputs/<model_name>/midi_examples/example_<N>/`

Each example contains three MIDI files:
- **ground_truth.mid** - Actual score from test set
- **generated_score.mid** - Model's autoregressive predictions
- **performance.mid** - Performance (control tokens) from test set

### Log Probability Plots

#### Individual Example Plots
- `model_comparison_outputs/<model_name>/midi_examples/example_<N>/<model_name>_example_<N>_log_probs.png`
- Shows log probability of each predicted token as generation progresses
- Separate plots for Time, Duration, and Pitch predictions

#### Aggregate Plots
- `model_comparison_outputs/<model_name>/<model_name>_aggregate_log_probs.png`
- Averages log probabilities across all 5 examples
- Shows model uncertainty trends across sequence positions

### Interpreting Log Probability Plots

- **High log prob (near 0)**: Model is very confident (e.g., -0.1 means ~90% probability)
- **Low log prob (-1 to -5)**: Model is uncertain (e.g., -3 means ~5% probability)
- **Very low log prob (< -5)**: Model is very confused/guessing

**Reference lines:**
- Green dashed (-1.0): p=0.37 (uncertain but reasonable)
- Red dashed (-5.0): p=0.007 (very uncertain)

## Analysis

### What the Data Shows:

1. **Time perturbation augmentation failed to improve autoregressive timing**
   - Both augmented models (100ms, 150ms) have <1% time accuracy
   - Original model (50_model) had ~4% - new models are even WORSE
   - Suggests core problem is NOT lack of time variation in training data

2. **More augmentation = worse performance**
   - 150ms perturbation hurt all metrics compared to 100ms
   - Possible over-regularization or domain shift

3. **Exposure bias remains the root cause**
   - Models likely still have good teacher forcing accuracy (not tested here)
   - Catastrophic autoregressive failure indicates training mismatch
   - Model never learns to handle its own prediction errors

### Why Time is So Bad:

From pitch-forced experiments (earlier analysis), we know:
- Pitch errors cascade and make timing worse (2% → 11% when pitch forced)
- But even with perfect pitch, timing is fundamentally broken (11%)
- Time prediction depends heavily on exact sequence history
- Small errors compound exponentially in autoregressive generation

## Recommendations

### Short Term (Use existing models):
1. **Avoid autoregressive generation for production**
   - Time accuracy < 1% is unusable
   - Teacher forcing accuracy likely much better - use that if possible

2. **Use 100_model over 150_model**
   - Better performance across all metrics
   - 150ms augmentation was counterproductive

### Medium Term (Training improvements):
1. **Scheduled sampling** during training
   - Mix teacher forcing with autoregressive rollouts
   - Gradually increase autoregressive ratio
   - Helps model learn to handle its own errors

2. **Separate prediction heads** for time vs note
   - Time prediction may need different architecture
   - Consider specialized temporal modeling

3. **Curriculum learning**
   - Start with short sequences
   - Gradually increase sequence length
   - Build up autoregressive capability

### Long Term (Architecture changes):
1. **Consider diffusion models** or other non-autoregressive approaches
2. **Reinforcement learning** with autoregressive rollouts
3. **Different tokenization** that makes timing easier to predict

## Files Generated

```
model_comparison_outputs/
├── 50_model/
│   ├── 50_model_aggregate_log_probs.png
│   └── midi_examples/
│       ├── example_1/
│       │   ├── 50_model_example_1_ground_truth.mid
│       │   ├── 50_model_example_1_generated_score.mid
│       │   ├── 50_model_example_1_performance.mid
│       │   └── 50_model_example_1_log_probs.png
│       ├── example_2/ ...
│       ├── example_3/ ...
│       ├── example_4/ ...
│       └── example_5/ ...
├── 100_model/
│   ├── 100_model_aggregate_log_probs.png
│   └── midi_examples/
│       ├── example_1/ ...
│       ├── example_2/ ...
│       ├── example_3/ ...
│       ├── example_4/ ...
│       └── example_5/ ...
└── 150_model/
    ├── 150_model_aggregate_log_probs.png
    └── midi_examples/
        ├── example_1/ ...
        ├── example_2/ ...
        ├── example_3/ ...
        ├── example_4/ ...
        └── example_5/ ...
```

**Total:** 3 models × (5 examples × 4 files + 1 aggregate plot) = 63 files
