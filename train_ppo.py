"""
PPO fine-tuning for music anticipation model.

Uses PPO to fine-tune a pretrained model by maximizing autoregressive accuracy.
Addresses exposure bias by training on model's own predictions.
"""
import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import numpy as np
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import matplotlib.pyplot as plt
from anticipation.vocab import ANTICIPATE, CONTROL_OFFSET, REST, NOTE_OFFSET

class TokenizedDataset(Dataset):
    """Dataset for PPO training - loads sequences without augmentation."""
    def __init__(self, file_path):
        self.sequences = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '|' in line:
                    token_str, _ = line.split('|')
                    tokens = list(map(int, token_str.strip().split()))
                else:
                    tokens = list(map(int, line.split()))
                
                self.sequences.append(torch.tensor(tokens, dtype=torch.long))
        
        print(f"Loaded {len(self.sequences)} sequences")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


def find_score_positions(tokens):
    """Find all score triplet positions (excluding REST)."""
    positions = []
    i = 1  # Skip mode token
    while i < len(tokens) - 2:
        if (tokens[i] < CONTROL_OFFSET and 
            tokens[i+1] < CONTROL_OFFSET and 
            tokens[i+2] < CONTROL_OFFSET and
            tokens[i+2] != REST):
            positions.append((i, i+1, i+2))
            i += 3
        else:
            i += 1
    return positions


def autoregressive_generate_with_log_probs(model, tokens, device):
    """
    Generate score tokens autoregressively and collect log probabilities.
    
    Returns:
        generated_tokens: list of generated tokens
        log_probs: log probabilities of generated tokens
        rewards: reward for each token (1 if matches GT, 0 otherwise)
        score_positions: positions of score triplets in original sequence
    """
    tokens_list = tokens.tolist()
    score_positions = find_score_positions(tokens_list)
    
    if len(score_positions) == 0:
        return [], [], [], []
    
    first_score_pos = score_positions[0][0]
    context = tokens_list[:first_score_pos]
    
    generated_tokens = []
    log_probs = []
    rewards = []
    
    model.eval()
    with torch.no_grad():
        last_pos = first_score_pos
        
        for time_pos, dur_pos, pitch_pos in score_positions:
            # Add ground truth control tokens
            if time_pos > last_pos:
                context.extend(tokens_list[last_pos:time_pos])
            
            # Predict TIME (without KV cache to save memory)
            input_tensor = torch.tensor([context]).to(device)
            outputs = model(input_tensor, use_cache=False)
            logits = outputs.logits[0, -1]
            probs = F.softmax(logits, dim=-1)
            pred_time = torch.multinomial(probs, 1).item()
            log_prob_time = torch.log(probs[pred_time] + 1e-10).item()
            
            generated_tokens.append(pred_time)
            log_probs.append(log_prob_time)
            rewards.append(1.0 if pred_time == tokens_list[time_pos] else 0.0)
            context.append(pred_time)
            
            # Predict DURATION
            input_tensor = torch.tensor([context]).to(device)
            outputs = model(input_tensor, use_cache=False)
            logits = outputs.logits[0, -1]
            probs = F.softmax(logits, dim=-1)
            pred_dur = torch.multinomial(probs, 1).item()
            log_prob_dur = torch.log(probs[pred_dur] + 1e-10).item()
            
            generated_tokens.append(pred_dur)
            log_probs.append(log_prob_dur)
            rewards.append(1.0 if pred_dur == tokens_list[dur_pos] else 0.0)
            context.append(pred_dur)
            
            # Predict PITCH
            input_tensor = torch.tensor([context]).to(device)
            outputs = model(input_tensor, use_cache=False)
            logits = outputs.logits[0, -1]
            probs = F.softmax(logits, dim=-1)
            pred_pitch = torch.multinomial(probs, 1).item()
            log_prob_pitch = torch.log(probs[pred_pitch] + 1e-10).item()
            
            generated_tokens.append(pred_pitch)
            log_probs.append(log_prob_pitch)
            # Higher reward for correct pitch
            rewards.append(2.0 if pred_pitch == tokens_list[pitch_pos] else 0.0)
            context.append(pred_pitch)
            
            last_pos = pitch_pos + 1
    
    return generated_tokens, log_probs, rewards, score_positions


def compute_advantages(rewards, gamma=0.99, lam=0.95):
    """Compute GAE (Generalized Advantage Estimation)."""
    advantages = []
    returns = []
    
    gae = 0
    running_return = 0
    
    for r in reversed(rewards):
        running_return = r + gamma * running_return
        returns.insert(0, running_return)
        
        delta = r + gamma * 0 - 0  # Simplified TD error (no value function)
        gae = delta + gamma * lam * gae
        advantages.insert(0, gae)
    
    advantages = torch.tensor(advantages, dtype=torch.float32)
    returns = torch.tensor(returns, dtype=torch.float32)
    
    # Normalize advantages
    if len(advantages) > 1:
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    
    return advantages, returns


def ppo_update(model, optimizer, tokens, generated_tokens, old_log_probs, advantages, 
               score_positions, device, clip_epsilon=0.2, ppo_epochs=4):
    """Perform PPO update on the model."""
    
    total_loss = 0
    num_updates = 0
    
    for _ in range(ppo_epochs):
        # Rebuild context and compute new log probs (without KV cache to save memory)
        tokens_list = tokens.tolist()
        first_score_pos = score_positions[0][0]
        context = tokens_list[:first_score_pos]
        
        model.train()
        
        new_log_probs = []
        gen_idx = 0
        
        for time_pos, dur_pos, pitch_pos in score_positions:
            # Add ground truth control tokens
            if time_pos > context[-1] if context else time_pos > 0:
                context.extend(tokens_list[len(context):time_pos])
            
            # Time token
            input_tensor = torch.tensor([context + [generated_tokens[gen_idx]]]).to(device)
            outputs = model(input_tensor, use_cache=False)
            logits = outputs.logits[0, -2]  # Get logits before the appended token
            probs = F.softmax(logits, dim=-1)
            new_log_probs.append(torch.log(probs[generated_tokens[gen_idx]] + 1e-10))
            context.append(generated_tokens[gen_idx])
            gen_idx += 1
            
            # Duration token
            input_tensor = torch.tensor([context + [generated_tokens[gen_idx]]]).to(device)
            outputs = model(input_tensor, use_cache=False)
            logits = outputs.logits[0, -2]
            probs = F.softmax(logits, dim=-1)
            new_log_probs.append(torch.log(probs[generated_tokens[gen_idx]] + 1e-10))
            context.append(generated_tokens[gen_idx])
            gen_idx += 1
            
            # Pitch token
            input_tensor = torch.tensor([context + [generated_tokens[gen_idx]]]).to(device)
            outputs = model(input_tensor, use_cache=False)
            logits = outputs.logits[0, -2]
            probs = F.softmax(logits, dim=-1)
            new_log_probs.append(torch.log(probs[generated_tokens[gen_idx]] + 1e-10))
            context.append(generated_tokens[gen_idx])
            gen_idx += 1
        
        new_log_probs = torch.stack(new_log_probs)
        old_log_probs_tensor = torch.tensor(old_log_probs, device=device)
        
        # Compute ratio
        ratio = torch.exp(new_log_probs - old_log_probs_tensor)
        
        # PPO clipped objective
        advantages_tensor = advantages.to(device)
        surr1 = ratio * advantages_tensor
        surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages_tensor
        policy_loss = -torch.min(surr1, surr2).mean()
        
        # Entropy bonus (encourage exploration)
        entropy = -new_log_probs.mean()
        loss = policy_loss - 0.01 * entropy
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_updates += 1
    
    return total_loss / num_updates


def train_ppo(args):
    """Main PPO training loop."""
    
    # Setup
    accelerator = Accelerator(gradient_accumulation_steps=args.gradient_accumulation_steps)
    device = accelerator.device
    
    print(f"\n{'='*60}")
    print(f"PPO Training Configuration")
    print(f"{'='*60}")
    print(f"Model: {args.model}")
    print(f"Train file: {args.train_file}")
    print(f"Val file: {args.val_file}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"PPO epochs per batch: {args.ppo_epochs}")
    print(f"Clip epsilon: {args.clip_epsilon}")
    print(f"Total epochs: {args.epochs}")
    print(f"Device: {device}")
    print(f"{'='*60}\n")
    
    # Load datasets
    train_dataset = TokenizedDataset(args.train_file)
    val_dataset = TokenizedDataset(args.val_file)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)  # PPO processes one at a time
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    
    # Load model
    print(f"Loading model from {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model)
    
    # Enable gradient checkpointing to save memory
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        print("✓ Gradient checkpointing enabled")
    
    # Resize embeddings if needed
    from anticipation.vocab import VOCAB_SIZE
    if model.config.vocab_size != VOCAB_SIZE:
        print(f"Resizing embeddings from {model.config.vocab_size} to {VOCAB_SIZE}")
        model.resize_token_embeddings(VOCAB_SIZE)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Prepare with accelerator
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )
    
    # Training history
    history = {
        'epoch': [],
        'train_reward': [],
        'val_reward': [],
        'train_pitch_acc': [],
        'val_pitch_acc': []
    }
    
    best_val_reward = -float('inf')
    
    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")
        
        # Training
        model.train()
        total_reward = 0
        total_pitches = 0
        correct_pitches = 0
        num_sequences = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        for batch_idx, tokens in enumerate(progress_bar):
            tokens = tokens[0]  # Unwrap batch dimension
            
            # Generate with old policy and collect rollout
            gen_tokens, log_probs, rewards, score_pos = autoregressive_generate_with_log_probs(
                model, tokens, device
            )
            
            if len(gen_tokens) == 0:
                continue
            
            # Compute advantages
            advantages, returns = compute_advantages(rewards)
            
            # PPO update
            loss = ppo_update(
                model, optimizer, tokens, gen_tokens, log_probs, advantages,
                score_pos, device, args.clip_epsilon, args.ppo_epochs
            )
            
            # Track metrics
            total_reward += sum(rewards)
            num_sequences += 1
            
            # Count pitch accuracy (every 3rd token is pitch)
            for i in range(2, len(rewards), 3):
                total_pitches += 1
                if rewards[i] > 0:
                    correct_pitches += 1
            
            if (batch_idx + 1) % 10 == 0:
                avg_reward = total_reward / num_sequences if num_sequences > 0 else 0
                pitch_acc = 100 * correct_pitches / total_pitches if total_pitches > 0 else 0
                progress_bar.set_postfix({
                    'reward': f'{avg_reward:.2f}',
                    'pitch_acc': f'{pitch_acc:.1f}%'
                })
            
            if args.max_train_batches > 0 and batch_idx >= args.max_train_batches:
                break
        
        avg_train_reward = total_reward / num_sequences if num_sequences > 0 else 0
        train_pitch_acc = 100 * correct_pitches / total_pitches if total_pitches > 0 else 0
        
        print(f"\nTraining - Avg Reward: {avg_train_reward:.2f}, Pitch Acc: {train_pitch_acc:.2f}%")
        
        # Validation
        model.eval()
        val_reward = 0
        val_pitches = 0
        val_correct = 0
        val_sequences = 0
        
        with torch.no_grad():
            for batch_idx, tokens in enumerate(tqdm(val_loader, desc="Validation")):
                tokens = tokens[0]
                
                gen_tokens, log_probs, rewards, score_pos = autoregressive_generate_with_log_probs(
                    model, tokens, device
                )
                
                if len(gen_tokens) == 0:
                    continue
                
                val_reward += sum(rewards)
                val_sequences += 1
                
                for i in range(2, len(rewards), 3):
                    val_pitches += 1
                    if rewards[i] > 0:
                        val_correct += 1
                
                if args.max_val_batches > 0 and batch_idx >= args.max_val_batches:
                    break
        
        avg_val_reward = val_reward / val_sequences if val_sequences > 0 else 0
        val_pitch_acc = 100 * val_correct / val_pitches if val_pitches > 0 else 0
        
        print(f"Validation - Avg Reward: {avg_val_reward:.2f}, Pitch Acc: {val_pitch_acc:.2f}%")
        
        # Save history
        history['epoch'].append(epoch + 1)
        history['train_reward'].append(avg_train_reward)
        history['val_reward'].append(avg_val_reward)
        history['train_pitch_acc'].append(train_pitch_acc)
        history['val_pitch_acc'].append(val_pitch_acc)
        
        # Save best model
        if avg_val_reward > best_val_reward:
            best_val_reward = avg_val_reward
            output_dir = f"{args.output_dir}/best"
            os.makedirs(output_dir, exist_ok=True)
            
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir)
            print(f"✓ Saved best model to {output_dir} (reward: {avg_val_reward:.2f})")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            output_dir = f"{args.output_dir}/checkpoint-{epoch+1}"
            os.makedirs(output_dir, exist_ok=True)
            
            unwrapped_model = accelerator.unwrap_model(model)
            unwrapped_model.save_pretrained(output_dir)
            print(f"✓ Saved checkpoint to {output_dir}")
    
    # Plot results
    plot_training_curves(history, args.output_dir)
    
    print(f"\n{'='*60}")
    print("Training Complete!")
    print(f"Best validation reward: {best_val_reward:.2f}")
    print(f"Models saved to: {args.output_dir}")
    print(f"{'='*60}\n")


def plot_training_curves(history, output_dir):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Reward plot
    axes[0].plot(history['epoch'], history['train_reward'], 'b-', label='Train', linewidth=2)
    axes[0].plot(history['epoch'], history['val_reward'], 'r-', label='Val', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Average Reward')
    axes[0].set_title('PPO Training: Reward')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Pitch accuracy plot
    axes[1].plot(history['epoch'], history['train_pitch_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(history['epoch'], history['val_pitch_acc'], 'r-', label='Val', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Pitch Accuracy (%)')
    axes[1].set_title('PPO Training: Pitch Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_dir}/ppo_training_curves.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✓ Saved training curves to {output_dir}/ppo_training_curves.png")


def main():
    parser = argparse.ArgumentParser(description='PPO fine-tuning for music anticipation')
    
    # Data
    parser.add_argument('--train_file', type=str, default='data/train_normalized.txt',
                        help='Path to training data')
    parser.add_argument('--val_file', type=str, default='data/test_normalized.txt',
                        help='Path to validation data')
    
    # Model
    parser.add_argument('--model', type=str, default='stanford-crfm/music-medium-800k',
                        help='Pretrained model to fine-tune')
    parser.add_argument('--output_dir', type=str, default='model-ppo',
                        help='Directory to save fine-tuned model')
    
    # Training
    parser.add_argument('--epochs', type=int, default=3,
                        help='Number of PPO epochs')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size (PPO typically uses 1)')
    parser.add_argument('--lr', type=float, default=1e-5,
                        help='Learning rate')
    parser.add_argument('--ppo_epochs', type=int, default=4,
                        help='PPO optimization epochs per batch')
    parser.add_argument('--clip_epsilon', type=float, default=0.2,
                        help='PPO clip parameter')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help='Gradient accumulation steps')
    
    # Efficiency
    parser.add_argument('--max_train_batches', type=int, default=1000,
                        help='Max training batches per epoch (0 = all)')
    parser.add_argument('--max_val_batches', type=int, default=200,
                        help='Max validation batches (0 = all)')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    train_ppo(args)


if __name__ == "__main__":
    main()
