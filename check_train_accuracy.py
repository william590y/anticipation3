"""
Check pitch accuracy on training set (train_clean.txt)
"""
import torch
from transformers import GPT2LMHeadModel
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

CONTROL_OFFSET = 27513

def evaluate_pitch_accuracy(model, dataloader, device):
    """Evaluate pitch accuracy on score note tokens"""
    model.eval()
    correct_pitches = 0
    total_pitches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch[0].to(device)
            labels = batch[1].to(device)
            
            outputs = model(input_ids, labels=labels)
            logits = outputs.logits
            
            for b in range(input_ids.shape[0]):
                seq_input = input_ids[b]
                seq_labels = labels[b]
                seq_logits = logits[b]
                
                i = 1
                while i < len(seq_input) - 2:
                    if (seq_input[i] < CONTROL_OFFSET and 
                        seq_input[i+1] < CONTROL_OFFSET and 
                        seq_input[i+2] < CONTROL_OFFSET):
                        
                        note_pos = i + 2
                        if seq_labels[note_pos] != -100:
                            predicted_token = seq_logits[note_pos - 1].argmax().item()
                            true_token = seq_labels[note_pos].item()
                            
                            if predicted_token == true_token:
                                correct_pitches += 1
                            total_pitches += 1
                        
                        i += 3
                    else:
                        i += 1
    
    accuracy = correct_pitches / total_pitches if total_pitches > 0 else 0.0
    return accuracy, total_pitches, correct_pitches

# Load model
print("Loading model from new_model/...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GPT2LMHeadModel.from_pretrained('new_model/')
model = model.to(device)
model.eval()
print(f"Model loaded on {device}")

# Load TRAINING data
print("\nLoading TRAINING data from data/train_clean.txt...")
try:
    with open('data/train_clean.txt', 'r') as f:
        lines = f.readlines()[:100]  # Sample 100 sequences from training set
except FileNotFoundError:
    print("train_clean.txt not found, trying train_output.txt...")
    with open('data/train_output.txt', 'r') as f:
        lines = f.readlines()[:100]

val_data = []
for line in lines:
    line = line.strip()
    if '|' in line:
        token_str, _ = line.split('|')
        tokens = [int(t) for t in token_str.strip().split()]
    else:
        tokens = [int(t) for t in line.split()]
    val_data.append(tokens)

print(f"Loaded {len(val_data)} training sequences")

# Pad sequences
max_len = max(len(seq) for seq in val_data)
input_ids = []
labels = []

for seq in val_data:
    input_seq = seq.copy()
    label_seq = seq.copy()
    input_seq += [0] * (max_len - len(input_seq))
    label_seq += [-100] * (max_len - len(label_seq))
    input_ids.append(input_seq)
    labels.append(label_seq)

dataset = TensorDataset(torch.tensor(input_ids), torch.tensor(labels))
dataloader = DataLoader(dataset, batch_size=4)

# Evaluate
print("Evaluating pitch accuracy on TRAINING set...")
accuracy, total, correct = evaluate_pitch_accuracy(model, dataloader, device)

print("\n" + "="*80)
print(f"TRAINING SET Pitch Accuracy: {accuracy*100:.2f}%")
print(f"Correct: {correct}/{total}")
print("="*80)
print("\nComparison:")
print(f"  Validation set (test_clean.txt): ~91.29%")
print(f"  Training set (train_clean.txt):  {accuracy*100:.2f}%")
print("\nNote: If training accuracy is also ~92%, this suggests:")
print("  - The model has reached its ceiling on this task")
print("  - ~8% of notes may be inherently ambiguous or noisy in the data")
