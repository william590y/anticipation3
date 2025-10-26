"""
Smoketest for pitch accuracy calculation in train.py
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import constants from train.py
import sys
sys.path.insert(0, '.')

# Constants
CONTROL_OFFSET = 27513

class DummyModel(nn.Module):
    """Model that we can control predictions for"""
    def __init__(self, vocab_size, mode='perfect'):
        super().__init__()
        self.vocab_size = vocab_size
        self.mode = mode
        
    def forward(self, input_ids, labels=None):
        batch_size, seq_len = input_ids.shape
        
        if self.mode == 'perfect':
            # Perfect predictions: logits[t] predicts labels[t+1]
            logits = torch.zeros(batch_size, seq_len, self.vocab_size)
            if labels is not None:
                for b in range(batch_size):
                    for t in range(seq_len - 1):  # logits[t] predicts labels[t+1]
                        if labels[b, t + 1] != -100:
                            logits[b, t, labels[b, t + 1]] = 100.0  # High confidence on correct token
        
        elif self.mode == 'random':
            # Random predictions
            logits = torch.randn(batch_size, seq_len, self.vocab_size)
        
        elif self.mode == 'off_by_one':
            # Always predict token+1 (always wrong for notes)
            logits = torch.zeros(batch_size, seq_len, self.vocab_size)
            if labels is not None:
                for b in range(batch_size):
                    for t in range(seq_len - 1):
                        if labels[b, t + 1] != -100:
                            wrong_token = min(labels[b, t + 1].item() + 1, self.vocab_size - 1)
                            logits[b, t, wrong_token] = 100.0
        
        return type('Output', (), {'logits': logits})()

def evaluate_model(model, dataloader, device):
    """Copy of evaluate_model from train.py (simplified)"""
    model.eval()
    total_loss = 0
    correct_pitches = 0
    total_pitches = 0
    
    from tqdm import tqdm
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)
            
            outputs = model(input_ids, labels=labels)
            logits = outputs.logits
            
            # Calculate pitch accuracy on score note tokens
            for b in range(input_ids.shape[0]):
                seq_input = input_ids[b]
                seq_labels = labels[b]
                seq_logits = logits[b]
                
                # Start from position 1 (skip first token)
                i = 1
                while i < len(seq_input) - 2:
                    # Check if this is a score triplet (all 3 tokens < CONTROL_OFFSET)
                    if (seq_input[i] < CONTROL_OFFSET and 
                        seq_input[i+1] < CONTROL_OFFSET and 
                        seq_input[i+2] < CONTROL_OFFSET):
                        # This is a score triplet
                        # Position i+2 is the note token
                        note_pos = i + 2
                        
                        # Only count if not masked in labels
                        if seq_labels[note_pos] != -100:
                            predicted_token = seq_logits[note_pos - 1].argmax().item()
                            true_token = seq_labels[note_pos].item()
                            
                            if predicted_token == true_token:
                                correct_pitches += 1
                            total_pitches += 1
                        
                        i += 3  # Move to next triplet
                    else:
                        i += 1  # Not a score triplet, move forward
    
    pitch_accuracy = correct_pitches / total_pitches if total_pitches > 0 else 0.0
    print(f"  Total pitches evaluated: {total_pitches}, Correct: {correct_pitches}")
    return pitch_accuracy

def create_test_data():
    """Create synthetic test data with score and control triplets"""
    sequences = []
    
    for _ in range(4):  # 4 sequences
        seq = []
        # Add 10 score triplets (time, duration, note)
        for j in range(10):
            time = 100 + j * 10  # < CONTROL_OFFSET
            duration = 50 + j * 5  # < CONTROL_OFFSET
            note = 60 + (j % 12)  # MIDI notes 60-71 (< CONTROL_OFFSET)
            seq.extend([time, duration, note])
        
        # Add 10 control triplets (time, duration, note)
        for j in range(10):
            time = CONTROL_OFFSET + 100 + j * 10  # >= CONTROL_OFFSET
            duration = CONTROL_OFFSET + 50 + j * 5  # >= CONTROL_OFFSET
            note = CONTROL_OFFSET + 60 + (j % 12)  # >= CONTROL_OFFSET
            seq.extend([time, duration, note])
        
        sequences.append(seq)
    
    # Pad sequences to same length
    max_len = max(len(s) for s in sequences)
    input_ids = []
    labels = []
    
    for seq in sequences:
        # For next-token prediction:
        # input_ids are the sequence
        # labels are the same sequence (model predicts next token)
        # The model internally handles the shift
        input_seq = seq.copy()
        label_seq = seq.copy()
        
        # Pad
        input_seq += [0] * (max_len - len(input_seq))
        label_seq += [-100] * (max_len - len(label_seq))  # Pad with -100 for masking
        
        input_ids.append(input_seq)
        labels.append(label_seq)
    
    return torch.tensor(input_ids), torch.tensor(labels)

def run_smoketest():
    """Run smoketest with different model behaviors"""
    print("="*80)
    print("PITCH ACCURACY SMOKETEST")
    print("="*80)
    
    device = torch.device('cpu')
    vocab_size = 55028
    
    # Create test data
    input_ids, labels = create_test_data()
    print(f"\nTest data created:")
    print(f"  - Batch size: {input_ids.shape[0]}")
    print(f"  - Sequence length: {input_ids.shape[1]}")
    print(f"  - Score triplets per sequence: 10 (30 tokens)")
    print(f"  - Control triplets per sequence: 10 (30 tokens)")
    print(f"  - Total score note tokens to predict: {input_ids.shape[0] * 10}")
    
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=2)
    
    print("\n" + "-"*80)
    print("TEST 1: Perfect predictions (should get 100% accuracy)")
    print("-"*80)
    model = DummyModel(vocab_size, mode='perfect')
    accuracy = evaluate_model(model, dataloader, device)
    print(f"Pitch Accuracy: {accuracy*100:.2f}%")
    expected = 100.0
    actual = accuracy * 100
    if abs(actual - expected) < 0.01:
        print("✅ PASS: Got expected 100% accuracy")
    else:
        print(f"❌ FAIL: Expected {expected}%, got {actual}%")
    
    print("\n" + "-"*80)
    print("TEST 2: Off-by-one predictions (should get 0% accuracy)")
    print("-"*80)
    model = DummyModel(vocab_size, mode='off_by_one')
    accuracy = evaluate_model(model, dataloader, device)
    print(f"Pitch Accuracy: {accuracy*100:.2f}%")
    expected = 0.0
    actual = accuracy * 100
    if abs(actual - expected) < 0.01:
        print("✅ PASS: Got expected 0% accuracy")
    else:
        print(f"❌ FAIL: Expected {expected}%, got {actual}%")
    
    print("\n" + "-"*80)
    print("TEST 3: Random predictions (should get ~0.04% accuracy)")
    print("-"*80)
    model = DummyModel(vocab_size, mode='random')
    accuracy = evaluate_model(model, dataloader, device)
    print(f"Pitch Accuracy: {accuracy*100:.2f}%")
    print(f"Note: Random chance is 1/{vocab_size} ≈ {100.0/vocab_size:.4f}%")
    if accuracy * 100 < 5.0:  # Should be very low
        print("✅ PASS: Random accuracy is appropriately low")
    else:
        print(f"❌ FAIL: Random accuracy too high: {accuracy*100:.2f}%")
    
    print("\n" + "="*80)
    print("ADDITIONAL CHECKS")
    print("="*80)
    
    # Verify that control triplets are NOT counted
    print("\nVerifying that only SCORE triplets are counted (not control)...")
    
    # Create data with ONLY control triplets (no score triplets)
    control_only_sequences = []
    for _ in range(2):
        seq = []
        for j in range(20):  # 20 control triplets
            time = CONTROL_OFFSET + 100 + j * 10
            duration = CONTROL_OFFSET + 50 + j * 5
            note = CONTROL_OFFSET + 60 + (j % 12)
            seq.extend([time, duration, note])
        control_only_sequences.append(seq)
    
    max_len = max(len(s) for s in control_only_sequences)
    control_input_ids = []
    control_labels = []
    
    for seq in control_only_sequences:
        input_seq = [0] + seq[:-1]
        label_seq = seq
        input_seq += [0] * (max_len - len(input_seq))
        label_seq += [-100] * (max_len - len(label_seq))
        control_input_ids.append(input_seq)
        control_labels.append(label_seq)
    
    control_input_ids = torch.tensor(control_input_ids)
    control_labels = torch.tensor(control_labels)
    control_dataset = TensorDataset(control_input_ids, control_labels)
    control_dataloader = DataLoader(control_dataset, batch_size=2)
    
    model = DummyModel(vocab_size, mode='perfect')
    accuracy = evaluate_model(model, control_dataloader, device)
    print(f"Pitch Accuracy on CONTROL-ONLY data: {accuracy*100:.2f}%")
    
    if accuracy == 0.0:
        print("✅ PASS: Control triplets are NOT counted (accuracy is undefined/0)")
    else:
        print(f"❌ WARNING: Got {accuracy*100:.2f}% accuracy on control-only data")
    
    print("\n" + "="*80)
    print("SMOKETEST COMPLETE")
    print("="*80)

def test_real_model():
    """Test the actual trained model at new_model/"""
    print("\n" + "="*80)
    print("TESTING REAL MODEL: new_model/")
    print("="*80)
    
    from transformers import GPT2LMHeadModel
    from torch.utils.data import DataLoader
    
    # Load the model
    print("\nLoading model from new_model/...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = GPT2LMHeadModel.from_pretrained('new_model/')
    model = model.to(device)
    model.eval()
    print(f"Model loaded on {device}")
    
    # Load validation data
    print("\nLoading validation data from data/test_output.txt...")
    try:
        with open('data/test_output.txt', 'r') as f:
            lines = f.readlines()
    except FileNotFoundError:
        print("❌ data/test_output.txt not found! Please run tokenize-asap.py first.")
        return
    
    # Parse data
    val_data = []
    for line in lines[:30]:  # Use first 30 sequences
        parts = line.strip().split(' | ')
        if len(parts) >= 1:
            tokens = [int(t) for t in parts[0].split()]
            val_data.append(tokens)
    
    print(f"Loaded {len(val_data)} validation sequences")
    
    # Pad sequences
    max_len = max(len(seq) for seq in val_data)
    input_ids = []
    labels = []
    
    for seq in val_data:
        input_seq = seq.copy()
        label_seq = seq.copy()
        
        # Pad
        input_seq += [0] * (max_len - len(input_seq))
        label_seq += [-100] * (max_len - len(label_seq))
        
        input_ids.append(input_seq)
        labels.append(label_seq)
    
    input_ids = torch.tensor(input_ids)
    labels = torch.tensor(labels)
    
    from torch.utils.data import TensorDataset
    dataset = TensorDataset(input_ids, labels)
    dataloader = DataLoader(dataset, batch_size=2)  # Smaller batch size for speed
    
    # Evaluate
    print("\nEvaluating pitch accuracy on validation set...")
    print(f"Note: Batched evaluation, {len(dataloader)} batches")
    accuracy = evaluate_model(model, dataloader, device)
    
    print("\n" + "-"*80)
    print(f"REAL MODEL PITCH ACCURACY: {accuracy*100:.2f}%")
    print("-"*80)
    
    return accuracy

if __name__ == '__main__':
    run_smoketest()
    test_real_model()
