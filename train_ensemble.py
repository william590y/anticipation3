"""
Ensemble Training Script for Music Anticipation

Based on "Pre-training under infinite compute" (arXiv:2509.14786)
Key insight: Train K independent models with different random seeds,
then average their logits at inference time for better performance.

The paper shows that ensembling achieves a lower loss asymptote than
single model parameter scaling, especially in data-constrained settings.
"""

import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoConfig
from torch.optim import AdamW
from tqdm import tqdm
import gc
import traceback
import matplotlib.pyplot as plt
from anticipation.vocab import ANTICIPATE, AUTOREGRESS, VOCAB_SIZE, CONTROL_OFFSET, NOTE_OFFSET, REST
from anticipation.config import TIME_RESOLUTION, MAX_TIME, MAX_DUR
import random
import math
import json
from typing import List, Optional, Tuple

# ============================================================================
# Helper Functions (same as train.py)
# ============================================================================

def print_gpu_memory_stats():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"GPU {i} memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")

def check_model_for_nans(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN detected in parameter {name}")
            return True
    return False

# Force CUDA if available
if torch.cuda.is_available():
    device = torch.device("cuda")
    device_count = torch.cuda.device_count()
    print(f"CUDA is available with {device_count} device(s)")
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"  Device {i}: {device_name}")
else:
    device = torch.device("cpu")
    print("CUDA is not available! Training will be much slower on CPU.")

print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")

# ============================================================================
# Dataset (same as train.py)
# ============================================================================

class TokenizedDataset(Dataset):
    """Dataset that loads clean sequences and applies augmentation on-the-fly."""
    
    def __init__(self, file_path, perturb_std_ms=0.0, mask_prob=0.0, is_training=True):
        self.sequences = []
        self.perturb_std_ms = perturb_std_ms if is_training else 0.0
        self.mask_prob = mask_prob if is_training else 0.0
        self.is_training = is_training
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '|' in line:
                    token_str, _ = line.split('|')
                    tokens = list(map(int, token_str.strip().split()))
                else:
                    tokens = list(map(int, line.split()))
                
                tokens = [max(0, t) for t in tokens]
                self.sequences.append(torch.tensor(tokens, dtype=torch.long))
        
        self.sequence_length = len(self.sequences[0]) if self.sequences else 0
        print(f"Loaded {len(self.sequences)} sequences with length {self.sequence_length}")
        
        if self.sequences:
            max_token = max(max(seq.tolist()) for seq in self.sequences)
            min_token = min(min(seq.tolist()) for seq in self.sequences)
            if max_token >= VOCAB_SIZE or min_token < 0:
                raise ValueError(f"Invalid token range: [{min_token}, {max_token}], must be [0, {VOCAB_SIZE-1}]")
    
    def __len__(self):
        return len(self.sequences)
    
    def _augment_sequence(self, tokens):
        """Apply on-the-fly augmentation: time perturbation + masking."""
        from anticipation.vocab import (CONTROL_OFFSET, SEPARATOR, ANTICIPATE, REST,
                                        ATIME_OFFSET, ADUR_OFFSET, ANOTE_OFFSET)
        
        if not self.is_training or (self.perturb_std_ms == 0 and self.mask_prob == 0):
            return tokens.clone(), []
        
        augmented = tokens.clone()
        mask_indices = []
        perturb_std_units = (self.perturb_std_ms / 1000.0) * TIME_RESOLUTION if self.perturb_std_ms > 0 else 0
        
        i = 1
        while i < len(augmented) - 2:
            if (augmented[i] >= CONTROL_OFFSET and 
                augmented[i+1] >= CONTROL_OFFSET and 
                augmented[i+2] >= CONTROL_OFFSET and
                augmented[i] != SEPARATOR and 
                augmented[i] != ANTICIPATE):
                
                if self.mask_prob > 0 and torch.rand(1).item() < self.mask_prob:
                    mask_indices.extend([i, i+1, i+2])
                
                base_time = augmented[i].item() - ATIME_OFFSET
                time_perturbation = int(torch.randn(1).item() * perturb_std_units)
                perturbed_time = max(0, min(MAX_TIME - 1, base_time + time_perturbation))
                augmented[i] = ATIME_OFFSET + perturbed_time
                
                base_dur = augmented[i+1].item() - ADUR_OFFSET
                dur_perturbation = int(torch.randn(1).item() * perturb_std_units)
                perturbed_dur = max(0, min(MAX_DUR - 1, base_dur + dur_perturbation))
                augmented[i+1] = ADUR_OFFSET + perturbed_dur
                
                i += 3
            else:
                i += 1
        
        return augmented, mask_indices
    
    def __getitem__(self, idx):
        tokens = self.sequences[idx]
        augmented_tokens, mask_idxs = self._augment_sequence(tokens)
        augmented_tokens = torch.clamp(augmented_tokens, 0, VOCAB_SIZE - 1)
        
        attention_mask = torch.ones_like(augmented_tokens)
        if mask_idxs:
            attention_mask[mask_idxs] = 0
        
        labels = augmented_tokens.clone()
        if mask_idxs:
            labels[mask_idxs] = -100
        
        return {"input_ids": augmented_tokens, "attention_mask": attention_mask, "labels": labels}

# ============================================================================
# Ensemble Model Wrapper
# ============================================================================

class EnsembleModel(torch.nn.Module):
    """
    Wrapper that holds K independently trained models and averages their logits.
    
    From the paper (Section 4.1):
    LogitAvg(M_i) produces a model with likelihood for a sequence x given by:
    LogitAvg(M_i)(x) ∝ exp(1/K * sum(log(M(x))))
    
    This is equivalent to averaging the log-probabilities (logits) across models.
    """
    
    def __init__(self, models: List[torch.nn.Module]):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.num_members = len(models)
        print(f"Created ensemble with {self.num_members} members")
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """
        Forward pass that averages logits across all ensemble members.
        
        Returns:
            CausalLMOutputWithPast-like object with averaged logits and loss
        """
        all_logits = []
        
        # Get logits from each model
        for model in self.models:
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **kwargs
            )
            all_logits.append(outputs.logits)
        
        # Average logits across ensemble members
        stacked_logits = torch.stack(all_logits, dim=0)  # [K, batch, seq, vocab]
        averaged_logits = stacked_logits.mean(dim=0)  # [batch, seq, vocab]
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            from torch.nn import CrossEntropyLoss
            loss_fct = CrossEntropyLoss()
            # Shift for causal LM loss
            shift_logits = averaged_logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        # Return a simple namespace-like object
        class EnsembleOutput:
            def __init__(self, loss, logits):
                self.loss = loss
                self.logits = logits
        
        return EnsembleOutput(loss=loss, logits=averaged_logits)
    
    def save_pretrained(self, save_directory, **kwargs):
        """Save each ensemble member to a subdirectory."""
        os.makedirs(save_directory, exist_ok=True)
        
        # Save ensemble config
        config = {
            "num_members": self.num_members,
            "model_type": "ensemble"
        }
        with open(os.path.join(save_directory, "ensemble_config.json"), 'w') as f:
            json.dump(config, f)
        
        # Save each member
        for i, model in enumerate(self.models):
            member_dir = os.path.join(save_directory, f"member_{i}")
            os.makedirs(member_dir, exist_ok=True)
            # Unwrap if needed
            if hasattr(model, 'module'):
                model.module.save_pretrained(member_dir)
            else:
                model.save_pretrained(member_dir)
        
        print(f"Saved ensemble with {self.num_members} members to {save_directory}")
    
    @classmethod
    def from_pretrained(cls, load_directory, **kwargs):
        """Load ensemble from saved directory."""
        config_path = os.path.join(load_directory, "ensemble_config.json")
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        models = []
        for i in range(config["num_members"]):
            member_dir = os.path.join(load_directory, f"member_{i}")
            model = AutoModelForCausalLM.from_pretrained(member_dir, **kwargs)
            models.append(model)
        
        return cls(models)

# ============================================================================
# Training Functions
# ============================================================================

def create_ensemble_members(
    model_name: str,
    num_members: int,
    seeds: Optional[List[int]] = None,
    model_kwargs: dict = None
) -> Tuple[List[torch.nn.Module], List[int]]:
    """
    Create K independently initialized models for the ensemble.
    
    From the paper (Section 4.1):
    The ensembling algorithm trains K members that are identical except for
    random seed Z_i controlling the data order and model initialization.
    """
    if seeds is None:
        seeds = [42 + i * 1000 for i in range(num_members)]
    
    if model_kwargs is None:
        model_kwargs = {}
    
    models = []
    for i, seed in enumerate(seeds):
        print(f"\nInitializing ensemble member {i+1}/{num_members} with seed {seed}...")
        
        # Set seed for reproducible initialization
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        
        # Load model with fresh initialization influenced by seed
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            trust_remote_code=True,
            use_cache=False,
            **model_kwargs
        )
        
        # Resize embeddings if needed
        current_vocab_size = model.config.vocab_size
        if current_vocab_size != VOCAB_SIZE:
            print(f"  Resizing embeddings from {current_vocab_size} to {VOCAB_SIZE}")
            model.resize_token_embeddings(VOCAB_SIZE)
        
        models.append(model)
        print(f"  ✓ Member {i+1} initialized")
    
    return models, seeds


def evaluate_single_model(model, dataloader, accelerator, max_batches=50, autoregressive_samples=10):
    """
    Evaluate a single model on validation data.
    
    Returns:
        tuple: (avg_loss, teacher_forced_accuracy, autoregressive_accuracy)
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    correct_pitches = 0
    total_pitches = 0
    
    # Collect sequences for autoregressive evaluation
    all_sequences = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
            
            outputs = model(**batch)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            logits = outputs.logits
            
            batch_size = input_ids.size(0)
            total_loss += outputs.loss.item() * batch_size
            total_samples += batch_size
            
            # Collect sequences for autoregressive eval
            if len(all_sequences) < autoregressive_samples:
                for seq in input_ids:
                    all_sequences.append(seq)
                    if len(all_sequences) >= autoregressive_samples:
                        break
            
            # Calculate teacher-forced pitch accuracy on score triplets
            for b in range(batch_size):
                seq_input = input_ids[b]
                seq_labels = labels[b]
                seq_logits = logits[b]
                
                i = 1
                while i < len(seq_input) - 2:
                    if (seq_input[i] < CONTROL_OFFSET and 
                        seq_input[i+1] < CONTROL_OFFSET and 
                        seq_input[i+2] < CONTROL_OFFSET and
                        seq_input[i+2] != REST):
                        
                        note_pos = i + 2
                        if seq_labels[note_pos] != -100:
                            predicted_token = seq_logits[note_pos - 1].argmax().item()
                            true_token = seq_labels[note_pos].item()
                            if predicted_token == true_token:
                                correct_pitches += 1
                            total_pitches += 1
                    i += 3
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    teacher_forced_accuracy = correct_pitches / total_pitches if total_pitches > 0 else 0
    
    # Autoregressive evaluation
    autoregressive_correct = 0
    autoregressive_total = 0
    
    if autoregressive_samples > 0 and all_sequences:
        for seq in all_sequences[:autoregressive_samples]:
            # Sequence format: [ANTICIPATE, SEP, SEP, SEP, control+rest pairs (positions 4-201), alternating score/control (202+)]
            alternating_start = 202
            if len(seq) <= alternating_start:
                continue
            
            # Start context with: ANTICIPATE + SEP SEP SEP + all control+rest pairs (positions 0-201)
            context = seq[:alternating_start].tolist()
            
            # Autoregressively generate the alternating score/control section
            pos = alternating_start
            while pos + 5 < len(seq):
                # Check if this is a score triplet
                if (seq[pos] < CONTROL_OFFSET and 
                    seq[pos+1] < CONTROL_OFFSET and 
                    seq[pos+2] < CONTROL_OFFSET and
                    seq[pos+2] != REST):
                    
                    # Generate TIME token
                    input_tensor = torch.tensor([context]).to(accelerator.device)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        logits = outputs.logits[0, -1, :]
                        pred_time = logits.argmax().item()
                    context.append(pred_time)
                    
                    # Generate DURATION token
                    input_tensor = torch.tensor([context]).to(accelerator.device)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        logits = outputs.logits[0, -1, :]
                        pred_dur = logits.argmax().item()
                    context.append(pred_dur)
                    
                    # Generate PITCH token
                    input_tensor = torch.tensor([context]).to(accelerator.device)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        logits = outputs.logits[0, -1, :]
                        pred_pitch = logits.argmax().item()
                    context.append(pred_pitch)
                    
                    # Check if predicted pitch matches ground truth
                    true_pitch = seq[pos + 2].item()
                    if pred_pitch == true_pitch:
                        autoregressive_correct += 1
                    autoregressive_total += 1
                    
                    pos += 3
                    
                    # After score triplet, add ground truth control triplet to context
                    if pos + 2 < len(seq):
                        context.extend([seq[pos].item(), seq[pos+1].item(), seq[pos+2].item()])
                        pos += 3
                else:
                    context.append(seq[pos].item())
                    pos += 1
    
    autoregressive_accuracy = autoregressive_correct / autoregressive_total if autoregressive_total > 0 else 0
    
    return avg_loss, teacher_forced_accuracy, autoregressive_accuracy


def plot_member_metrics(train_losses, val_losses, val_tf_accuracies, val_ar_accuracies, validation_steps, member_idx, output_dir):
    """Plot training metrics for a single ensemble member."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Training loss (linear)
    ax1 = axes[0, 0]
    steps = list(range(10, 10 * len(train_losses) + 1, 10))
    ax1.plot(steps, train_losses, alpha=0.7, color='blue')
    if validation_steps and val_losses:
        ax1.scatter(validation_steps, val_losses, color='red', s=30, zorder=5, label='Val Loss')
        ax1.plot(validation_steps, val_losses, color='red', alpha=0.3, linestyle='--')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'Member {member_idx+1}: Training Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Training loss (log)
    ax2 = axes[0, 1]
    ax2.semilogy(steps, train_losses, alpha=0.7, color='blue')
    if validation_steps and val_losses:
        ax2.scatter(validation_steps, val_losses, color='red', s=30, zorder=5, label='Val Loss')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Loss (log)')
    ax2.set_title(f'Member {member_idx+1}: Training Loss (Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Teacher-forced pitch accuracy
    ax3 = axes[1, 0]
    if validation_steps and val_tf_accuracies:
        ax3.plot(validation_steps, val_tf_accuracies, color='green', marker='o', label='Teacher-Forced')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Pitch Accuracy (%)')
    ax3.set_title(f'Member {member_idx+1}: Teacher-Forced Accuracy')
    ax3.set_ylim([0, 100])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Autoregressive pitch accuracy
    ax4 = axes[1, 1]
    if validation_steps and val_ar_accuracies:
        ax4.plot(validation_steps, val_ar_accuracies, color='purple', marker='s', label='Autoregressive')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Pitch Accuracy (%)')
    ax4.set_title(f'Member {member_idx+1}: Autoregressive Accuracy')
    ax4.set_ylim([0, 100])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()


def train_single_member(
    member_idx: int,
    model: torch.nn.Module,
    train_dataset: Dataset,
    val_dataset: Dataset,
    args,
    seed: int,
    output_dir: Path
) -> Tuple[torch.nn.Module, dict]:
    """
    Train a single ensemble member independently.
    
    From the paper (Section 4.2):
    Each member of an ensemble happens to learn different features when
    independently trained, which is why ensembling helps.
    """
    print(f"\n{'='*60}")
    print(f"Training Ensemble Member {member_idx + 1}")
    print(f"Seed: {seed}")
    print(f"{'='*60}")
    
    # Set seed for this member's training
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
    # Create member-specific output directory
    member_dir = output_dir / f"member_{member_idx}"
    os.makedirs(member_dir, exist_ok=True)
    
    # Initialize accelerator
    mixed_precision = 'bf16' if torch.cuda.is_available() else 'no'
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=mixed_precision,
    )
    
    # Create data loaders with different shuffling based on seed
    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    
    # Create train dataloader with seed-specific shuffling
    g = torch.Generator()
    g.manual_seed(seed)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        generator=g,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )
    
    # Setup optimizer (same as train.py)
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=1e-6,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )
    
    # Prepare for training
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    val_dataloader = accelerator.prepare_data_loader(val_dataloader)
    
    # Learning rate scheduler - cosine decay
    initial_lr = args.learning_rate
    final_lr = initial_lr / 10
    
    def lr_lambda(current_step):
        progress = float(current_step) / float(max(1, args.max_steps_per_member))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return (final_lr / initial_lr) + (1.0 - final_lr / initial_lr) * cosine_decay
    
    from torch.optim.lr_scheduler import LambdaLR
    scheduler = LambdaLR(optimizer, lr_lambda)
    
    # Training loop
    model.train()
    completed_steps = 0
    train_losses = []
    val_losses = []
    val_tf_accuracies = []  # Teacher-forced
    val_ar_accuracies = []  # Autoregressive
    validation_steps = []
    
    progress_bar = tqdm(total=args.max_steps_per_member, desc=f"Member {member_idx+1}", leave=True)
    
    while completed_steps < args.max_steps_per_member:
        for batch in train_dataloader:
            with accelerator.accumulate(model):
                outputs = model(**batch)
                loss = outputs.loss
                
                if torch.isnan(loss).any() or torch.isinf(loss).any():
                    optimizer.zero_grad()
                    continue
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), max_norm=2.0)
                    
                    # Check for NaN in gradients (same as train.py)
                    has_nan_grads = False
                    for name, param in model.named_parameters():
                        if param.grad is not None and torch.isnan(param.grad).any():
                            print(f"NaN gradient detected in {name}")
                            has_nan_grads = True
                            break
                    
                    if has_nan_grads:
                        print("Skipping update due to NaN gradients")
                        optimizer.zero_grad()
                        continue
                    
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    completed_steps += 1
                    progress_bar.update(1)
                    
                    if completed_steps % 10 == 0:
                        train_losses.append(loss.item())
                        
                        # Check for NaN parameters periodically (same as train.py)
                        if check_model_for_nans(model):
                            print("NaN parameters detected in model! Training may be unstable.")
                    
                    if completed_steps % 100 == 0:
                        print(f"  Step {completed_steps}, Loss: {loss.item():.4f}, LR: {scheduler.get_last_lr()[0]:.2e}")
                        print_gpu_memory_stats()
                    
                    # Periodic validation and plotting (every eval_steps)
                    # Skip if this is also a checkpoint step to avoid double validation
                    is_checkpoint_step = (completed_steps % args.save_steps == 0)
                    if completed_steps % args.eval_steps == 0 and not is_checkpoint_step:
                        val_loss, val_tf_acc, val_ar_acc = evaluate_single_model(model, val_dataloader, accelerator)
                        val_losses.append(val_loss)
                        val_tf_accuracies.append(val_tf_acc * 100)
                        val_ar_accuracies.append(val_ar_acc * 100)
                        validation_steps.append(completed_steps)
                        print(f"  [Validation] Step {completed_steps}: Loss={val_loss:.4f}, TF-Acc={val_tf_acc*100:.2f}%, AR-Acc={val_ar_acc*100:.2f}%")
                        model.train()
                    
                    # Save checkpoint with plot (every save_steps)
                    if is_checkpoint_step:
                        # Run validation before saving checkpoint
                        val_loss, val_tf_acc, val_ar_acc = evaluate_single_model(model, val_dataloader, accelerator)
                        val_losses.append(val_loss)
                        val_tf_accuracies.append(val_tf_acc * 100)
                        val_ar_accuracies.append(val_ar_acc * 100)
                        validation_steps.append(completed_steps)
                        print(f"  [Validation] Step {completed_steps}: Loss={val_loss:.4f}, TF-Acc={val_tf_acc*100:.2f}%, AR-Acc={val_ar_acc*100:.2f}%")
                        model.train()
                        checkpoint_dir = member_dir / f"checkpoint-{completed_steps}"
                        os.makedirs(checkpoint_dir, exist_ok=True)
                        unwrapped = accelerator.unwrap_model(model)
                        unwrapped.save_pretrained(checkpoint_dir)
                        
                        # Save metrics and plot
                        np.savez(
                            checkpoint_dir / "training_history.npz",
                            train_losses=np.array(train_losses),
                            val_losses=np.array(val_losses),
                            val_tf_accuracies=np.array(val_tf_accuracies),
                            val_ar_accuracies=np.array(val_ar_accuracies),
                            validation_steps=np.array(validation_steps)
                        )
                        plot_member_metrics(train_losses, val_losses, val_tf_accuracies, 
                                          val_ar_accuracies, validation_steps, member_idx, checkpoint_dir)
                        print(f"  [Checkpoint] Saved to {checkpoint_dir}")
                        
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                else:
                    optimizer.zero_grad()
                
                if completed_steps >= args.max_steps_per_member:
                    break
    
    progress_bar.close()
    
    # Final validation
    final_val_loss, final_tf_acc, final_ar_acc = evaluate_single_model(model, val_dataloader, accelerator)
    val_losses.append(final_val_loss)
    val_tf_accuracies.append(final_tf_acc * 100)
    val_ar_accuracies.append(final_ar_acc * 100)
    validation_steps.append(completed_steps)
    
    print(f"  Member {member_idx+1} final: Loss={final_val_loss:.4f}, TF-Acc={final_tf_acc*100:.2f}%, AR-Acc={final_ar_acc*100:.2f}%")
    
    # Save member checkpoint
    unwrapped_model = accelerator.unwrap_model(model)
    unwrapped_model.save_pretrained(member_dir)
    
    # Save training history
    np.savez(
        member_dir / "training_history.npz",
        train_losses=np.array(train_losses),
        val_losses=np.array(val_losses),
        val_tf_accuracies=np.array(val_tf_accuracies),
        val_ar_accuracies=np.array(val_ar_accuracies),
        validation_steps=np.array(validation_steps),
        seed=seed
    )
    
    # Final plot for this member
    plot_member_metrics(train_losses, val_losses, val_tf_accuracies, 
                       val_ar_accuracies, validation_steps, member_idx, member_dir)
    
    metrics = {
        "final_val_loss": final_val_loss,
        "final_tf_acc": final_tf_acc,
        "final_ar_acc": final_ar_acc,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_tf_accuracies": val_tf_accuracies,
        "val_ar_accuracies": val_ar_accuracies,
        "validation_steps": validation_steps,
        "seed": seed
    }
    
    return unwrapped_model, metrics


def evaluate_ensemble(
    ensemble: EnsembleModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int = 50,
    autoregressive_samples: int = 10
) -> Tuple[float, float, float]:
    """
    Evaluate the ensemble model.
    
    Returns:
        tuple: (avg_loss, teacher_forced_accuracy, autoregressive_accuracy)
    """
    ensemble.eval()
    total_loss = 0
    total_samples = 0
    correct_pitches = 0
    total_pitches = 0
    
    # Collect sequences for autoregressive evaluation
    all_sequences = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating ensemble", leave=False)):
            if batch_idx >= max_batches:
                break
            
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = ensemble(**batch)
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            logits = outputs.logits
            
            batch_size = input_ids.size(0)
            total_loss += outputs.loss.item() * batch_size
            total_samples += batch_size
            
            # Collect sequences for autoregressive eval
            if len(all_sequences) < autoregressive_samples:
                for seq in input_ids:
                    all_sequences.append(seq)
                    if len(all_sequences) >= autoregressive_samples:
                        break
            
            # Calculate teacher-forced pitch accuracy
            for b in range(batch_size):
                seq_input = input_ids[b]
                seq_labels = labels[b]
                seq_logits = logits[b]
                
                i = 1
                while i < len(seq_input) - 2:
                    if (seq_input[i] < CONTROL_OFFSET and 
                        seq_input[i+1] < CONTROL_OFFSET and 
                        seq_input[i+2] < CONTROL_OFFSET and
                        seq_input[i+2] != REST):
                        
                        note_pos = i + 2
                        if seq_labels[note_pos] != -100:
                            predicted_token = seq_logits[note_pos - 1].argmax().item()
                            true_token = seq_labels[note_pos].item()
                            if predicted_token == true_token:
                                correct_pitches += 1
                            total_pitches += 1
                    i += 3
    
    avg_loss = total_loss / total_samples if total_samples > 0 else 0
    teacher_forced_accuracy = correct_pitches / total_pitches if total_pitches > 0 else 0
    
    # Autoregressive evaluation
    autoregressive_correct = 0
    autoregressive_total = 0
    
    if autoregressive_samples > 0 and all_sequences:
        for seq in tqdm(all_sequences[:autoregressive_samples], desc="Autoregressive eval", leave=False):
            alternating_start = 202
            if len(seq) <= alternating_start:
                continue
            
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
                        outputs = ensemble(input_tensor)
                        logits = outputs.logits[0, -1, :]
                        pred_time = logits.argmax().item()
                    context.append(pred_time)
                    
                    # Generate DURATION token
                    input_tensor = torch.tensor([context]).to(device)
                    with torch.no_grad():
                        outputs = ensemble(input_tensor)
                        logits = outputs.logits[0, -1, :]
                        pred_dur = logits.argmax().item()
                    context.append(pred_dur)
                    
                    # Generate PITCH token
                    input_tensor = torch.tensor([context]).to(device)
                    with torch.no_grad():
                        outputs = ensemble(input_tensor)
                        logits = outputs.logits[0, -1, :]
                        pred_pitch = logits.argmax().item()
                    context.append(pred_pitch)
                    
                    true_pitch = seq[pos + 2].item()
                    if pred_pitch == true_pitch:
                        autoregressive_correct += 1
                    autoregressive_total += 1
                    
                    pos += 3
                    
                    if pos + 2 < len(seq):
                        context.extend([seq[pos].item(), seq[pos+1].item(), seq[pos+2].item()])
                        pos += 3
                else:
                    context.append(seq[pos].item())
                    pos += 1
    
    autoregressive_accuracy = autoregressive_correct / autoregressive_total if autoregressive_total > 0 else 0
    
    return avg_loss, teacher_forced_accuracy, autoregressive_accuracy


def plot_ensemble_metrics(all_member_metrics: List[dict], ensemble_metrics: dict, output_dir: Path):
    """Plot training metrics for all ensemble members and the combined ensemble."""
    num_members = len(all_member_metrics)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Plot 1: Individual member training losses
    ax1 = axes[0, 0]
    for i, metrics in enumerate(all_member_metrics):
        losses = metrics["train_losses"]
        steps = list(range(10, 10 * len(losses) + 1, 10))
        ax1.plot(steps, losses, label=f'Member {i+1}', alpha=0.7)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Training Loss')
    ax1.set_title('Individual Member Training Losses')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Final validation losses comparison
    ax2 = axes[0, 1]
    member_losses = [m["final_val_loss"] for m in all_member_metrics]
    x = list(range(1, num_members + 1))
    ax2.bar(x, member_losses, color='steelblue', alpha=0.7, label='Individual Members')
    ax2.axhline(y=ensemble_metrics["val_loss"], color='red', linestyle='--', 
                linewidth=2, label=f'Ensemble: {ensemble_metrics["val_loss"]:.4f}')
    ax2.axhline(y=np.mean(member_losses), color='green', linestyle=':', 
                linewidth=2, label=f'Mean: {np.mean(member_losses):.4f}')
    ax2.set_xlabel('Member')
    ax2.set_ylabel('Validation Loss')
    ax2.set_title('Validation Loss: Members vs Ensemble')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Loss improvement from ensembling
    ax3 = axes[0, 2]
    ensemble_sizes = list(range(1, num_members + 1))
    cumulative_losses = []
    for k in ensemble_sizes:
        avg_loss = np.mean(member_losses[:k])
        cumulative_losses.append(avg_loss)
    ax3.plot(ensemble_sizes, cumulative_losses, 'o-', color='purple', 
             label='Avg of first K members')
    ax3.axhline(y=ensemble_metrics["val_loss"], color='red', linestyle='--',
                label=f'Full ensemble: {ensemble_metrics["val_loss"]:.4f}')
    ax3.set_xlabel('Number of Ensemble Members (K)')
    ax3.set_ylabel('Validation Loss')
    ax3.set_title('Loss vs Ensemble Size')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Teacher-forced accuracy comparison
    ax4 = axes[1, 0]
    member_tf_accs = [m["final_tf_acc"] * 100 for m in all_member_metrics]
    bars = ax4.bar(x, member_tf_accs, color='green', alpha=0.7, label='Individual Members')
    ax4.axhline(y=ensemble_metrics["tf_accuracy"] * 100, color='red', linestyle='--', 
                linewidth=2, label=f'Ensemble: {ensemble_metrics["tf_accuracy"]*100:.2f}%')
    ax4.set_xlabel('Member')
    ax4.set_ylabel('Pitch Accuracy (%)')
    ax4.set_title('Teacher-Forced Accuracy: Members vs Ensemble')
    ax4.set_ylim([0, 100])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Autoregressive accuracy comparison
    ax5 = axes[1, 1]
    member_ar_accs = [m["final_ar_acc"] * 100 for m in all_member_metrics]
    bars = ax5.bar(x, member_ar_accs, color='purple', alpha=0.7, label='Individual Members')
    ax5.axhline(y=ensemble_metrics["ar_accuracy"] * 100, color='red', linestyle='--', 
                linewidth=2, label=f'Ensemble: {ensemble_metrics["ar_accuracy"]*100:.2f}%')
    ax5.set_xlabel('Member')
    ax5.set_ylabel('Pitch Accuracy (%)')
    ax5.set_title('Autoregressive Accuracy: Members vs Ensemble')
    ax5.set_ylim([0, 100])
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Summary comparison
    ax6 = axes[1, 2]
    categories = ['TF Accuracy', 'AR Accuracy']
    avg_member_tf = np.mean(member_tf_accs)
    avg_member_ar = np.mean(member_ar_accs)
    ensemble_tf = ensemble_metrics["tf_accuracy"] * 100
    ensemble_ar = ensemble_metrics["ar_accuracy"] * 100
    
    x_pos = np.arange(len(categories))
    width = 0.35
    
    ax6.bar(x_pos - width/2, [avg_member_tf, avg_member_ar], width, label='Avg Member', color='steelblue', alpha=0.7)
    ax6.bar(x_pos + width/2, [ensemble_tf, ensemble_ar], width, label='Ensemble', color='red', alpha=0.7)
    ax6.set_ylabel('Pitch Accuracy (%)')
    ax6.set_title('Ensemble vs Average Member')
    ax6.set_xticks(x_pos)
    ax6.set_xticklabels(categories)
    ax6.set_ylim([0, 100])
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'ensemble_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Ensemble metrics plot saved to {output_dir / 'ensemble_metrics.png'}")


def main():
    parser = argparse.ArgumentParser(description="Ensemble Training for Music Anticipation")
    
    # Data arguments
    parser.add_argument('--data_file', type=Path, default=Path('./data/train_normalized.txt'))
    parser.add_argument('--val_file', type=Path, default=Path('./data/test_normalized.txt'))
    
    # Model arguments
    parser.add_argument('--model_name', type=str, default='stanford-crfm/music-medium-800k')
    parser.add_argument('--output_dir', type=Path, default=Path('./ensemble_model_new'))
    
    # Ensemble arguments
    parser.add_argument('--num_members', type=int, default=4,
                       help='Number of ensemble members (K). Paper uses 2-8.')
    parser.add_argument('--base_seed', type=int, default=42,
                       help='Base random seed. Each member gets base_seed + i*1000')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=64)
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01,
                       help='Base weight decay (will be halved for ensemble members per paper)')
    parser.add_argument('--max_steps_per_member', type=int, default=1250,
                       help='Training steps per ensemble member')
    parser.add_argument('--save_steps', type=int, default=200,
                       help='Save checkpoint every N steps')
    parser.add_argument('--eval_steps', type=int, default=100,
                       help='Run validation every N steps')
    
    # Augmentation arguments
    parser.add_argument('--perturb_std_ms', type=float, default=50.0)
    parser.add_argument('--mask_prob', type=float, default=0)
    
    # Other arguments
    parser.add_argument('--force_cpu', action='store_true')
    parser.add_argument('--reduce_memory', action='store_true')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("ENSEMBLE TRAINING")
    print("Based on 'Pre-training under infinite compute' (arXiv:2509.14786)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Number of ensemble members (K): {args.num_members}")
    print(f"  Base model: {args.model_name}")
    print(f"  Steps per member: {args.max_steps_per_member}")
    print(f"  Total training steps: {args.num_members * args.max_steps_per_member}")
    print(f"  Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"  Output directory: {args.output_dir}")
    print()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save configuration
    config = vars(args).copy()
    config['data_file'] = str(config['data_file'])
    config['val_file'] = str(config['val_file'])
    config['output_dir'] = str(config['output_dir'])
    with open(args.output_dir / "ensemble_config.json", 'w') as f:
        json.dump(config, f, indent=2)
    
    # Load datasets
    print(f"Loading training dataset from {args.data_file}...")
    train_dataset = TokenizedDataset(
        args.data_file,
        perturb_std_ms=args.perturb_std_ms,
        mask_prob=args.mask_prob,
        is_training=True
    )
    
    print(f"Loading validation dataset from {args.val_file}...")
    val_dataset = TokenizedDataset(
        args.val_file,
        perturb_std_ms=0.0,
        mask_prob=0.0,
        is_training=False
    )
    
    # Generate seeds for each member
    seeds = [args.base_seed + i * 1000 for i in range(args.num_members)]
    print(f"\nEnsemble member seeds: {seeds}")
    
    # Model kwargs
    model_kwargs = {}
    if args.reduce_memory and torch.cuda.is_available():
        model_kwargs.update({
            "torch_dtype": torch.bfloat16,
            "low_cpu_mem_usage": True,
        })
    
    # Train each member independently
    trained_models = []
    all_member_metrics = []
    
    for i in range(args.num_members):
        print(f"\n{'#'*70}")
        print(f"# TRAINING MEMBER {i+1}/{args.num_members}")
        print(f"{'#'*70}")
        
        # Create fresh model for this member
        torch.manual_seed(seeds[i])
        np.random.seed(seeds[i])
        random.seed(seeds[i])
        
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name,
            trust_remote_code=True,
            use_cache=False,
            **model_kwargs
        )
        
        if model.config.vocab_size != VOCAB_SIZE:
            model.resize_token_embeddings(VOCAB_SIZE)
        
        # Train this member
        trained_model, metrics = train_single_member(
            member_idx=i,
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            args=args,
            seed=seeds[i],
            output_dir=args.output_dir
        )
        
        trained_models.append(trained_model)
        all_member_metrics.append(metrics)
        
        # Clear memory between members
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
    
    # Create and evaluate ensemble
    print(f"\n{'='*70}")
    print("CREATING ENSEMBLE")
    print(f"{'='*70}")
    
    # Load all trained members
    ensemble_models = []
    for i in range(args.num_members):
        member_dir = args.output_dir / f"member_{i}"
        model = AutoModelForCausalLM.from_pretrained(
            member_dir,
            trust_remote_code=True,
            use_cache=False,
            **model_kwargs
        )
        ensemble_models.append(model)
    
    # Create ensemble
    ensemble = EnsembleModel(ensemble_models)
    
    # Move ensemble to device
    if torch.cuda.is_available() and not args.force_cpu:
        ensemble = ensemble.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    # Create validation dataloader for ensemble evaluation
    def collate_fn(batch):
        input_ids = torch.stack([item["input_ids"] for item in batch])
        attention_mask = torch.stack([item["attention_mask"] for item in batch])
        labels = torch.stack([item["labels"] for item in batch])
        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
    )
    
    # Evaluate ensemble
    print("\nEvaluating ensemble...")
    ensemble_val_loss, ensemble_tf_acc, ensemble_ar_acc = evaluate_ensemble(
        ensemble, val_dataloader, device
    )
    
    ensemble_metrics = {
        "val_loss": ensemble_val_loss,
        "tf_accuracy": ensemble_tf_acc,
        "ar_accuracy": ensemble_ar_acc,
        "num_members": args.num_members
    }
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}")
    print(f"\nIndividual member results:")
    for i, metrics in enumerate(all_member_metrics):
        print(f"  Member {i+1}: Loss={metrics['final_val_loss']:.4f}, TF-Acc={metrics['final_tf_acc']*100:.2f}%, AR-Acc={metrics['final_ar_acc']*100:.2f}%")
    
    mean_member_loss = np.mean([m["final_val_loss"] for m in all_member_metrics])
    mean_member_tf_acc = np.mean([m["final_tf_acc"] for m in all_member_metrics])
    mean_member_ar_acc = np.mean([m["final_ar_acc"] for m in all_member_metrics])
    
    print(f"\nMean of individual members:")
    print(f"  Loss: {mean_member_loss:.4f}")
    print(f"  Teacher-Forced Accuracy: {mean_member_tf_acc*100:.2f}%")
    print(f"  Autoregressive Accuracy: {mean_member_ar_acc*100:.2f}%")
    
    print(f"\nEnsemble results:")
    print(f"  Loss: {ensemble_val_loss:.4f}")
    print(f"  Teacher-Forced Accuracy: {ensemble_tf_acc*100:.2f}%")
    print(f"  Autoregressive Accuracy: {ensemble_ar_acc*100:.2f}%")
    
    improvement = (mean_member_loss - ensemble_val_loss) / mean_member_loss * 100
    tf_improvement = (ensemble_tf_acc - mean_member_tf_acc) / mean_member_tf_acc * 100 if mean_member_tf_acc > 0 else 0
    ar_improvement = (ensemble_ar_acc - mean_member_ar_acc) / mean_member_ar_acc * 100 if mean_member_ar_acc > 0 else 0
    
    print(f"\nImprovement from ensembling:")
    print(f"  Loss reduction: {improvement:.2f}%")
    print(f"  Teacher-Forced Accuracy gain: {tf_improvement:.2f}%")
    print(f"  Autoregressive Accuracy gain: {ar_improvement:.2f}%")
    
    # Save ensemble
    ensemble_dir = args.output_dir / "ensemble_final"
    ensemble.save_pretrained(ensemble_dir)
    
    # Save final metrics
    with open(args.output_dir / "final_metrics.json", 'w') as f:
        json.dump({
            "ensemble_metrics": ensemble_metrics,
            "member_metrics": [
                {
                    "member": i, 
                    "val_loss": m["final_val_loss"], 
                    "tf_accuracy": m["final_tf_acc"],
                    "ar_accuracy": m["final_ar_acc"],
                    "seed": m["seed"]
                }
                for i, m in enumerate(all_member_metrics)
            ],
            "mean_member_loss": float(mean_member_loss),
            "mean_member_tf_acc": float(mean_member_tf_acc),
            "mean_member_ar_acc": float(mean_member_ar_acc),
            "loss_improvement_percent": float(improvement),
            "tf_accuracy_improvement_percent": float(tf_improvement),
            "ar_accuracy_improvement_percent": float(ar_improvement)
        }, f, indent=2)
    
    # Plot metrics
    plot_ensemble_metrics(all_member_metrics, ensemble_metrics, args.output_dir)
    
    print(f"\n✓ Ensemble training complete!")
    print(f"  Models saved to: {args.output_dir}")
    print(f"  Ensemble saved to: {ensemble_dir}")


if __name__ == "__main__":
    main()
