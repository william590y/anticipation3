"""
Evaluate autoregressive pitch accuracy with train-style protocol only:
score tokens are predicted one triplet at a time with ground-truth control (performance)
tokens interleaved, matching train.py validation. Use --checkpoints to evaluate
multiple checkpoints (e.g. checkpoint-1000 checkpoint-1750).
"""
import argparse
import os
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM
from anticipation.vocab import *
from anticipation.config import *
from anticipation.convert import events_to_midi

# #region agent log
DEBUG_LOG_PATH = os.path.join(os.path.abspath(os.path.dirname(__file__)), "debug-e30de5.log")
def _dbg(payload):
    with open(DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps({"sessionId": "e30de5", "timestamp": __import__("time").time() * 1000, **payload}) + "\n")
# #endregion

def greedy_decode_sequence(model, input_ids, max_new_tokens=1024):
    """Greedy decoding with KV caching."""
    device = model.device
    input_ids = input_ids.to(device)
    
    generated = input_ids.clone()
    past_key_values = None
    
    with torch.no_grad():
        for _ in range(max_new_tokens):
            if past_key_values is None:
                model_inputs = generated
            else:
                model_inputs = generated[:, -1:]
            
            outputs = model(model_inputs, past_key_values=past_key_values, use_cache=True)
            next_token_logits = outputs.logits[:, -1, :]
            past_key_values = outputs.past_key_values
            
            # Greedy: argmax
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated = torch.cat([generated, next_token], dim=1)
            
            if generated.shape[1] >= CONTEXT_SIZE:
                break
    
    return generated

def extract_aligned_pairs(tokens):
    """
    Extract aligned performance-score pairs from the interleaved sequence.
    
    Sequence format:
    - Position 0: ANTICIPATE
    - Positions 1-3: SEP SEP SEP
    - Positions 4-201: 33 control+rest pairs (k=33)
    - Positions 202+: Alternating score/control triplets
    
    Alignment relationship:
    - Performance note i from control+rest pairs → Score note i in alternating section
    - Performance note 33+j from alternating → Score note 33+j in alternating
    
    Returns:
        performance_triplets: List of [time, dur, pitch] (no CONTROL_OFFSET)
        score_triplets: List of [time, dur, pitch] (with offsets)
        Both lists have same length and are aligned by index
    """
    # Skip ANTICIPATE (position 0) and SEP SEP SEP (positions 1-3)
    # Start at position 4
    body = tokens[4:]
    
    k = 33  # Number of control+rest pairs
    control_rest_section_length = k * 6  # 33 pairs × 6 tokens = 198 tokens
    
    # Extract performance from control+rest pairs (positions 4-201)
    perf_from_pairs = []
    for i in range(k):
        base = i * 6
        # Control triplet (first 3 tokens of each pair)
        ctrl_time = body[base] - CONTROL_OFFSET
        ctrl_dur = body[base + 1] - CONTROL_OFFSET
        ctrl_pitch = body[base + 2] - CONTROL_OFFSET
        perf_from_pairs.append([ctrl_time, ctrl_dur, ctrl_pitch])
        # Rest triplet is ignored (positions base+3 to base+5)
    
    # Extract from alternating section (positions 202+, which is index 198 in body)
    alternating = body[control_rest_section_length:]
    
    score_from_alternating = []
    perf_from_alternating = []
    
    pos = 0
    while pos + 5 < len(alternating):
        # Score triplet (first 3 tokens)
        score_time = alternating[pos]
        score_dur = alternating[pos + 1]
        score_pitch = alternating[pos + 2]
        
        # Verify it's a score triplet (all < CONTROL_OFFSET, not REST)
        if (score_time < CONTROL_OFFSET and 
            score_dur < CONTROL_OFFSET and 
            score_pitch < CONTROL_OFFSET and 
            score_pitch != REST):
            score_from_alternating.append([score_time, score_dur, score_pitch])
        else:
            # Not a valid score triplet, stop extraction
            break
        
        pos += 3
        
        # Control triplet (next 3 tokens)
        if pos + 2 < len(alternating):
            ctrl_time = alternating[pos] - CONTROL_OFFSET
            ctrl_dur = alternating[pos + 1] - CONTROL_OFFSET
            ctrl_pitch = alternating[pos + 2] - CONTROL_OFFSET
            
            # Verify it's a control triplet
            if (alternating[pos] >= CONTROL_OFFSET and 
                alternating[pos + 1] >= CONTROL_OFFSET and 
                alternating[pos + 2] >= CONTROL_OFFSET):
                perf_from_alternating.append([ctrl_time, ctrl_dur, ctrl_pitch])
            else:
                # Not a valid control triplet, stop
                break
        
        pos += 3
    
    # Combine performance: all from control+rest pairs + those from alternating
    all_performance = perf_from_pairs + perf_from_alternating
    
    # Score only comes from alternating section
    all_score = score_from_alternating
    
    # The alignment is:
    # - Performance note 0-32 (from control+rest) → Score note 0-32 (from alternating)
    # - Performance note 33+ (from alternating) → Score note 33+ (from alternating)
    # So both lists should have the same length
    
    return all_performance, all_score


def run_autoregressive_eval(model, lines, num_examples, device):
    """
    Run train-style autoregressive eval only: predict score triplets with GT control
    (performance) tokens interleaved. Returns (train_style_correct, train_style_total).
    """
    train_style_correct = 0
    train_style_total = 0
    for example_idx in tqdm(range(num_examples), desc='Train-style (GT control interleaved)', unit='ex', leave=True):
        line = lines[example_idx]
        if '|' in line:
            token_str, _ = line.split('|')
            tokens = [int(t) for t in token_str.strip().split()]
        else:
            tokens = [int(t) for t in line.strip().split()]
        # Match train.py / evaluate_checkpoints: clamp invalid tokens
        tokens = [max(0, t) for t in tokens]
        alternating_start = 202
        if len(tokens) <= alternating_start:
            continue
        context = list(tokens[:alternating_start])
        pos = alternating_start
        while pos + 5 < len(tokens):
            if (tokens[pos] < CONTROL_OFFSET and tokens[pos+1] < CONTROL_OFFSET and tokens[pos+2] < CONTROL_OFFSET and tokens[pos+2] != REST):
                with torch.no_grad():
                    inp = torch.tensor([context], device=device)
                    out = model(inp)
                    pred_time = out.logits[0, -1, :].argmax().item()
                    context.append(pred_time)
                    inp = torch.tensor([context], device=device)
                    out = model(inp)
                    pred_dur = out.logits[0, -1, :].argmax().item()
                    context.append(pred_dur)
                    inp = torch.tensor([context], device=device)
                    out = model(inp)
                    pred_pitch = out.logits[0, -1, :].argmax().item()
                    context.append(pred_pitch)
                true_pitch = tokens[pos + 2]
                if pred_pitch == true_pitch:
                    train_style_correct += 1
                train_style_total += 1
                pos += 3
                if pos + 2 < len(tokens):
                    context.extend([tokens[pos], tokens[pos+1], tokens[pos+2]])
                    pos += 3
            else:
                context.append(tokens[pos])
                pos += 1
    return train_style_correct, train_style_total


def main():
    parser = argparse.ArgumentParser(description='Evaluate autoregressive pitch accuracy on checkpoints.')
    parser.add_argument('--checkpoints', nargs='+', default=['newest_model'],
                        help='Checkpoint dirs to evaluate (e.g. checkpoint-1000 checkpoint-1750). Default: newest_model')
    parser.add_argument('--data', default='data/test_combined.txt', help='Test data file (default matches train.py --val_file and evaluate_checkpoints.py)')
    parser.add_argument('--num_examples', type=int, default=30, help='Number of examples (first N lines). For comparison with evaluate_checkpoints use same file and similar N.')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    with open(args.data, 'r') as f:
        lines = f.readlines()
    num_examples = min(args.num_examples, len(lines))

    results = []
    for ckpt in tqdm(args.checkpoints, desc='Checkpoints'):
        tqdm.write(f"Loading {ckpt}...")
        model = AutoModelForCausalLM.from_pretrained(ckpt)
        model = model.to(device)
        model.eval()
        train_style_correct, train_style_total = run_autoregressive_eval(model, lines, num_examples, device)
        results.append((ckpt, train_style_correct, train_style_total))
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print()
    print("=" * 70)
    print("Autoregressive pitch accuracy (train-style: GT control interleaved)")
    print("=" * 70)
    for ckpt, train_style_correct, train_style_total in results:
        acc = 100.0 * train_style_correct / train_style_total if train_style_total else 0.0
        print(f"  {ckpt}: {acc:.2f}% ({train_style_correct}/{train_style_total})")
    print("=" * 70)


if __name__ == '__main__':
    main()