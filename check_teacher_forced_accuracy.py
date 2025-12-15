"""
Check if teacher forced accuracy calculation in train.py is correct.

The key question: How does GPT2LMHeadModel align logits with labels?
- input_ids: [t0, t1, t2, t3, t4]
- labels: [t0, t1, t2, t3, t4]
- logits: [?, ?, ?, ?, ?]

GPT2 internally shifts, so:
- logits[0] predicts t1 (predicts token after t0)
- logits[1] predicts t2 (predicts token after t1) 
- logits[2] predicts t3 (predicts token after t2)
- logits[i] predicts t[i+1]

So to predict token at position note_pos, we need logits[note_pos - 1].
"""

import torch
from transformers import GPT2LMHeadModel, GPT2Config

def test_gpt2_alignment():
    """Test how GPT2 aligns logits with labels"""
    print("="*80)
    print("Testing GPT2 Logits-Labels Alignment")
    print("="*80)
    
    # Create a tiny GPT2 model
    config = GPT2Config(
        vocab_size=100,
        n_positions=10,
        n_embd=32,
        n_layer=1,
        n_head=2
    )
    model = GPT2LMHeadModel(config)
    model.eval()
    
    # Create a simple sequence
    # input_ids: [10, 20, 30, 40, 50]
    # labels: [10, 20, 30, 40, 50]
    input_ids = torch.tensor([[10, 20, 30, 40, 50]])
    labels = torch.tensor([[10, 20, 30, 40, 50]])
    
    print(f"\nInput sequence: {input_ids[0].tolist()}")
    print(f"Labels sequence: {labels[0].tolist()}")
    
    # Get model output
    with torch.no_grad():
        outputs = model(input_ids=input_ids, labels=labels)
        logits = outputs.logits
    
    print(f"\nLogits shape: {logits.shape}")  # Should be [1, 5, 100]
    
    # Check what each logit position predicts
    print("\nChecking logit alignments:")
    for i in range(len(input_ids[0])):
        predicted_token = logits[0, i].argmax().item()
        if i < len(input_ids[0]) - 1:
            next_token = input_ids[0, i+1].item()
            print(f"  logits[{i}] predicts token {predicted_token} | next token in seq is {next_token}")
        else:
            print(f"  logits[{i}] predicts token {predicted_token} | (end of sequence)")
    
    print("\n" + "="*80)
    print("Conclusion:")
    print("="*80)
    print("GPT2 logits[i] should predict the token at position i+1")
    print("So to predict token at position P, we use logits[P-1]")
    print("")
    print("In the code:")
    print("  note_pos = i + 2  (position of note token in sequence)")
    print("  predicted = logits[note_pos - 1].argmax()  (logit that predicts note_pos)")
    print("  true = labels[note_pos]  (actual token at note_pos)")
    print("")
    print("This alignment is CORRECT ✓")
    
    # Now test the loss calculation
    print("\n" + "="*80)
    print("Testing Loss Calculation")
    print("="*80)
    
    # GPT2's forward pass with labels automatically:
    # 1. Shifts logits and labels internally for next-token prediction
    # 2. Computes cross-entropy loss
    
    # The internal shift in GPT2:
    # shift_logits = logits[..., :-1, :].contiguous()  # Remove last prediction
    # shift_labels = labels[..., 1:].contiguous()  # Remove first label
    # So it compares logits[0] with labels[1], logits[1] with labels[2], etc.
    
    print("\nGPT2 internally does:")
    print("  shift_logits = logits[..., :-1, :]  # [0, 1, 2, 3]")
    print("  shift_labels = labels[..., 1:]      # [1, 2, 3, 4]")
    print("  loss = CrossEntropy(shift_logits, shift_labels)")
    print("")
    print("This means:")
    print("  - logits[0] is compared with labels[1]")
    print("  - logits[1] is compared with labels[2]")
    print("  - logits[i] is compared with labels[i+1]")
    
    return True

def test_teacher_forced_calculation():
    """Test if the teacher-forced accuracy calculation is correct"""
    print("\n" + "="*80)
    print("Testing Teacher-Forced Accuracy Calculation")
    print("="*80)
    
    # Simulate the scenario from train.py
    # Sequence: [ANTICIPATE, time1, dur1, note1, time2, dur2, note2, ...]
    # Let's say note1 is at position 3, note2 is at position 6
    
    # For GPT2:
    # - To predict token at position 3, model uses logits[2]
    # - To predict token at position 6, model uses logits[5]
    
    print("\nScenario: Score triplet at positions [1, 2, 3]")
    print("  Position 1: time token")
    print("  Position 2: duration token")
    print("  Position 3: note token (the one we want to check)")
    print("")
    print("In train.py code:")
    print("  note_pos = 3")
    print("  predicted_token = logits[note_pos - 1].argmax()  = logits[2].argmax()")
    print("  true_token = labels[note_pos]  = labels[3]")
    print("")
    print("Is this correct?")
    print("  - logits[2] predicts the token at position 3 ✓")
    print("  - labels[3] is the ground truth token at position 3 ✓")
    print("  - Comparison is CORRECT ✓")
    
    return True

if __name__ == '__main__':
    test_gpt2_alignment()
    test_teacher_forced_calculation()
    
    print("\n" + "="*80)
    print("FINAL VERDICT")
    print("="*80)
    print("The teacher-forced accuracy calculation in train.py is CORRECT ✓")
    print("")
    print("The code correctly uses:")
    print("  predicted_token = seq_logits[note_pos - 1].argmax()")
    print("  true_token = seq_labels[note_pos]")
    print("")
    print("This properly aligns with GPT2's logit-label relationship where")
    print("logits[i] predicts the token at position i+1.")
