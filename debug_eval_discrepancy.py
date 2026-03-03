"""
Debug script to find discrepancies between train.py and evaluate_checkpoints.py
autoregressive evaluation methods.
"""
import torch
from transformers import AutoModelForCausalLM
from anticipation.vocab import CONTROL_OFFSET, REST

# Configuration
CHECKPOINT = 'checkpoint-1750'
TEST_FILE = 'data/test_combined.txt'
NUM_SEQUENCES = 3  # Test with small number first

def load_test_sequences(filepath, num_sequences):
    """Load sequences matching train.py's preprocessing."""
    sequences = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if '|' in line:
                token_str, _ = line.split('|')
                tokens = list(map(int, token_str.strip().split()))
            else:
                tokens = list(map(int, line.split()))
            # Match train.py: replace invalid tokens
            tokens = [max(0, t) for t in tokens]
            sequences.append(torch.tensor(tokens, dtype=torch.long))
            if len(sequences) >= num_sequences:
                break
    return sequences

def train_py_style_eval(model, seq, device):
    """Exactly replicate train.py's autoregressive evaluation."""
    correct = 0
    total = 0
    
    alternating_start = 202
    if len(seq) <= alternating_start:
        return 0, 0
    
    context = seq[:alternating_start].tolist()
    pos = alternating_start
    
    while pos + 5 < len(seq):
        if (seq[pos] < CONTROL_OFFSET and 
            seq[pos+1] < CONTROL_OFFSET and 
            seq[pos+2] < CONTROL_OFFSET and
            seq[pos+2] != REST):
            
            # Generate TIME token
            input_tensor = torch.tensor([context]).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs.logits[0, -1, :]
                pred_time = logits.argmax().item()
            context.append(pred_time)
            
            # Generate DURATION token
            input_tensor = torch.tensor([context]).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs.logits[0, -1, :]
                pred_dur = logits.argmax().item()
            context.append(pred_dur)
            
            # Generate PITCH token
            input_tensor = torch.tensor([context]).to(device)
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs.logits[0, -1, :]
                pred_pitch = logits.argmax().item()
            context.append(pred_pitch)
            
            # Check pitch
            true_pitch = seq[pos + 2].item()
            if pred_pitch == true_pitch:
                correct += 1
            total += 1
            
            pos += 3
            
            if pos + 2 < len(seq):
                context.extend([seq[pos].item(), seq[pos+1].item(), seq[pos+2].item()])
                pos += 3
        else:
            context.append(seq[pos].item())
            pos += 1
    
    return correct, total

def eval_checkpoints_style_eval(model, tokens, device):
    """Replicate evaluate_checkpoints.py's approach."""
    # tokens is a list
    score_start_idx = 202
    
    # Generate
    context = list(tokens[:score_start_idx])
    pos = score_start_idx
    
    while pos + 5 < len(tokens):
        if (tokens[pos] < CONTROL_OFFSET and 
            tokens[pos+1] < CONTROL_OFFSET and 
            tokens[pos+2] < CONTROL_OFFSET and
            tokens[pos+2] != REST):
            
            with torch.no_grad():
                input_tensor = torch.tensor([context], device=device)
                outputs = model(input_tensor)
                pred_time = outputs.logits[0, -1, :].argmax().item()
                context.append(pred_time)
                
                input_tensor = torch.tensor([context], device=device)
                outputs = model(input_tensor)
                pred_dur = outputs.logits[0, -1, :].argmax().item()
                context.append(pred_dur)
                
                input_tensor = torch.tensor([context], device=device)
                outputs = model(input_tensor)
                pred_pitch = outputs.logits[0, -1, :].argmax().item()
                context.append(pred_pitch)
            
            pos += 3
            
            if pos + 2 < len(tokens):
                context.extend([tokens[pos], tokens[pos+1], tokens[pos+2]])
                pos += 3
        else:
            context.append(tokens[pos])
            pos += 1
    
    # Now compute stats by position (matching compute_statistics_by_position)
    correct = 0
    total = 0
    
    pos = score_start_idx
    while pos + 5 < len(tokens) and pos + 5 < len(context):
        gt_t0, gt_t1, gt_t2 = tokens[pos], tokens[pos+1], tokens[pos+2]
        
        if (gt_t0 < CONTROL_OFFSET and gt_t1 < CONTROL_OFFSET and 
            gt_t2 < CONTROL_OFFSET and gt_t2 != REST):
            
            pred_t2 = context[pos+2]
            
            total += 1
            if gt_t2 == pred_t2:
                correct += 1
            
            pos += 3
            if pos + 2 < len(tokens):
                pos += 3
        else:
            pos += 1
    
    return correct, total

def main():
    print(f"Loading model from {CHECKPOINT}...")
    model = AutoModelForCausalLM.from_pretrained(CHECKPOINT)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    model.eval()
    
    print(f"Loading sequences from {TEST_FILE}...")
    sequences = load_test_sequences(TEST_FILE, NUM_SEQUENCES)
    print(f"  Loaded {len(sequences)} sequences")
    
    train_correct = 0
    train_total = 0
    eval_correct = 0
    eval_total = 0
    
    for i, seq in enumerate(sequences):
        print(f"\nSequence {i}:")
        
        # Train.py style
        tc, tt = train_py_style_eval(model, seq, device)
        train_correct += tc
        train_total += tt
        
        # Eval checkpoints style
        tokens = seq.tolist()
        ec, et = eval_checkpoints_style_eval(model, tokens, device)
        eval_correct += ec
        eval_total += et
        
        train_acc = 100.0 * tc / tt if tt > 0 else 0.0
        eval_acc = 100.0 * ec / et if et > 0 else 0.0
        
        print(f"  train.py style:       {tc}/{tt} = {train_acc:.2f}%")
        print(f"  eval_checkpoints style: {ec}/{et} = {eval_acc:.2f}%")
        
        if tc != ec or tt != et:
            print(f"  *** MISMATCH! ***")
    
    print("\n" + "="*60)
    print("AGGREGATE RESULTS")
    print("="*60)
    train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0
    eval_acc = 100.0 * eval_correct / eval_total if eval_total > 0 else 0.0
    print(f"train.py style:       {train_correct}/{train_total} = {train_acc:.2f}%")
    print(f"eval_checkpoints style: {eval_correct}/{eval_total} = {eval_acc:.2f}%")
    
    if train_correct == eval_correct and train_total == eval_total:
        print("\n✓ Results match perfectly!")
    else:
        print("\n✗ DISCREPANCY FOUND!")

if __name__ == '__main__':
    main()
