import argparse
import os
from pathlib import Path
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, get_cosine_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import gc
import traceback
import matplotlib.pyplot as plt
from anticipation.vocab import ANTICIPATE, AUTOREGRESS  # Import the flag token constants

# Helper function to monitor GPU memory usage
def print_gpu_memory_stats():
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i} memory allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"GPU {i} memory reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
            print(f"GPU {i} max memory allocated: {torch.cuda.max_memory_allocated(i) / 1024**2:.2f} MB")

# Check for NaN values in model parameters
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
    print(f"✓ CUDA is available with {device_count} device(s)")
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"  Device {i}: {device_name}")
        props = torch.cuda.get_device_properties(i)
        print(f"    - Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    - CUDA capability: {props.major}.{props.minor}")
else:
    device = torch.device("cpu")
    print("✗ CUDA is not available! Training will be much slower on CPU.")

# Explicitly print which device we're using
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

class TokenizedDataset(Dataset):
    """Dataset that loads clean sequences and applies augmentation on-the-fly.
    
    Sequences are packed and formatted by tokenize-asap.py:
    - Each sequence is exactly 1024 tokens
    - Format: [ANTICIPATE, control_tokens..., score_tokens..., PAD...]
    - Augmentation (perturbation + masking) applied during training, not tokenization
    """
    def __init__(self, file_path, perturb_std_ms=0.0, mask_prob=0.0, is_training=True):
        self.sequences = []
        self.perturb_std_ms = perturb_std_ms if is_training else 0.0
        self.mask_prob = mask_prob if is_training else 0.0
        self.is_training = is_training
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if '|' in line:
                    # New format: "token1 token2 ... | mask_idx1 mask_idx2 ..." (ignored, we augment on-the-fly)
                    token_str, _ = line.split('|')
                    tokens = list(map(int, token_str.strip().split()))
                else:
                    # Old format: just tokens
                    tokens = list(map(int, line.split()))
                
                self.sequences.append(torch.tensor(tokens, dtype=torch.long))
        
        self.sequence_length = len(self.sequences[0]) if self.sequences else 0
        print(f"Loaded {len(self.sequences)} sequences with length {self.sequence_length}")
        if self.is_training:
            print(f"  Training mode: perturb_std={self.perturb_std_ms}ms, mask_prob={self.mask_prob}")
        else:
            print(f"  Validation mode: no augmentation")
        
        # Validate format
        if self.sequences:
            from anticipation.vocab import ANTICIPATE
            sample = self.sequences[0].tolist()
            if len(sample) >= 1:
                if sample[0] == ANTICIPATE:
                    print(f"✓ Tokenization format validated (starts with ANTICIPATE token)")
                else:
                    print(f"⚠ Warning: First token is {sample[0]}, expected ANTICIPATE ({ANTICIPATE})")
    
    def __len__(self):
        return len(self.sequences)
    
    def _augment_sequence(self, tokens):
        """Apply on-the-fly augmentation: time perturbation + masking.
        
        Only augments CONTROL triplets, not score triplets or special tokens.
        Control triplets have all 3 tokens >= CONTROL_OFFSET.
        Score triplets have all 3 tokens < CONTROL_OFFSET.
        
        Returns:
            augmented_tokens: Perturbed tokens
            mask_indices: Indices to mask in loss (list of ints)
        """
        from anticipation.vocab import CONTROL_OFFSET, SEPARATOR, ANTICIPATE, REST
        from anticipation.config import TIME_RESOLUTION
        
        if not self.is_training or (self.perturb_std_ms == 0 and self.mask_prob == 0):
            # No augmentation for validation or if disabled
            return tokens.clone(), []
        
        augmented = tokens.clone()
        mask_indices = []
        
        # Convert perturbation std from ms to time resolution units
        perturb_std_units = (self.perturb_std_ms / 1000.0) * TIME_RESOLUTION if self.perturb_std_ms > 0 else 0
        
        # Iterate through sequence in triplets
        # Skip first token (ANTICIPATE mode token)
        i = 1
        while i < len(augmented) - 2:
            # Check if this is a control triplet:
            # - All 3 tokens must be >= CONTROL_OFFSET
            # - First token must not be SEPARATOR or ANTICIPATE (these are also >= CONTROL_OFFSET)
            if (augmented[i] >= CONTROL_OFFSET and 
                augmented[i+1] >= CONTROL_OFFSET and 
                augmented[i+2] >= CONTROL_OFFSET and
                augmented[i] != SEPARATOR and 
                augmented[i] != ANTICIPATE):
                
                # This is a control triplet (time, dur, pitch) with CONTROL_OFFSET added
                
                # Decide whether to mask this triplet
                if self.mask_prob > 0 and torch.rand(1).item() < self.mask_prob:
                    # Mark these 3 positions for masking in loss
                    mask_indices.extend([i, i+1, i+2])
                
                # Apply time perturbation to time and duration (NOT pitch)
                if perturb_std_units > 0:
                    # Perturb time (first token)
                    base_time = augmented[i].item() - CONTROL_OFFSET
                    time_perturbation = int(torch.randn(1).item() * perturb_std_units)
                    perturbed_time = max(0, base_time + time_perturbation)
                    augmented[i] = CONTROL_OFFSET + perturbed_time
                    
                    # Perturb duration (second token)
                    base_dur = augmented[i+1].item() - CONTROL_OFFSET
                    dur_perturbation = int(torch.randn(1).item() * perturb_std_units)
                    perturbed_dur = max(0, base_dur + dur_perturbation)
                    augmented[i+1] = CONTROL_OFFSET + perturbed_dur
                    
                    # Leave pitch (third token) unchanged
                
                i += 3  # Skip to next triplet
            else:
                # Not a control triplet - could be score, rest, separator, etc.
                # Don't augment, just move to next token
                i += 1
        
        return augmented, mask_indices
    
    def __getitem__(self, idx):
        tokens = self.sequences[idx]
        
        # Apply on-the-fly augmentation
        augmented_tokens, mask_idxs = self._augment_sequence(tokens)
        
        # Create labels (same as input, but with masked positions set to -100)
        labels = augmented_tokens.clone()
        if mask_idxs:
            labels[mask_idxs] = -100
        
        return {"input_ids": augmented_tokens, "labels": labels}

def evaluate_model(model, dataloader, accelerator, max_samples=500, autoregressive_samples=100):
    """Calculate validation loss and pitch accuracy on a dataset
    
    Args:
        model: The model to evaluate
        dataloader: DataLoader with validation data
        accelerator: Accelerator instance
        max_samples: Maximum number of sequences to evaluate for teacher-forced metrics (default: 500)
        autoregressive_samples: Number of sequences to evaluate autoregressively (default: 20)
    
    Returns:
        tuple: (avg_loss, teacher_forced_pitch_accuracy, autoregressive_pitch_accuracy)
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    # For teacher-forced pitch accuracy: track predictions on score note tokens
    correct_pitches = 0
    total_pitches = 0
    
    from anticipation.vocab import CONTROL_OFFSET, NOTE_OFFSET, REST
    import random
    
    # Randomly sample indices, then take only those batches from the dataloader
    # We need to iterate through dataloader (not convert to list) to preserve device placement
    total_batches = len(dataloader)
    if total_batches > 0:
        # Calculate how many batches we need for max_samples (estimate batch_size as 8)
        estimated_batch_size = 8
        num_batches_needed = min(total_batches, (max_samples + estimated_batch_size - 1) // estimated_batch_size)
        
        # Randomly select batch indices
        selected_indices = set(random.sample(range(total_batches), num_batches_needed))
    else:
        selected_indices = set()
    
    batches_processed = 0
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="Evaluating", leave=False)):
            # Only process selected batches
            if batch_idx not in selected_indices:
                continue
            
            batches_processed += 1
            if batches_processed > len(selected_indices):
                break
            outputs = model(**batch)
            loss = outputs.loss
            logits = outputs.logits
            
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            
            # Get batch size from the input shape
            batch_size = input_ids.size(0)
            
            # Accumulate loss (weighted by batch size)
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Calculate pitch accuracy on score note tokens (position 2 of score triplets)
            # Score triplets have all tokens < CONTROL_OFFSET
            # We need to find triplets where all 3 tokens < CONTROL_OFFSET and position is %3==2
            for b in range(batch_size):
                seq_input = input_ids[b]
                seq_labels = labels[b]
                seq_logits = logits[b]
                
                # Skip first token (mode token), start from position 1
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
                            predicted_token = seq_logits[note_pos - 1].argmax().item()  # Predict next token
                            true_token = seq_labels[note_pos].item()
                            
                            # Check if prediction matches ground truth
                            if predicted_token == true_token:
                                correct_pitches += 1
                            total_pitches += 1
                        
                        i += 3  # Move to next triplet
                    else:
                        i += 1  # Not a score triplet, move forward
    
    avg_loss = total_loss / total_samples
    teacher_forced_accuracy = correct_pitches / total_pitches if total_pitches > 0 else 0.0
    
    # Autoregressive evaluation: greedy decoding on a small subset
    autoregressive_correct = 0
    autoregressive_total = 0
    
    if autoregressive_samples > 0:
        # Collect a few sequences for autoregressive evaluation
        all_sequences = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"]
                for seq in input_ids:
                    all_sequences.append(seq)
                    if len(all_sequences) >= autoregressive_samples:
                        break
                if len(all_sequences) >= autoregressive_samples:
                    break
        
        # Run autoregressive generation on each sequence
        for seq in tqdm(all_sequences[:autoregressive_samples], desc="Autoregressive eval", leave=False):
            # Find all score triplet positions (for ground truth)
            score_positions = []
            i = 1  # Skip mode token
            while i < len(seq) - 2:
                if (seq[i] < CONTROL_OFFSET and 
                    seq[i+1] < CONTROL_OFFSET and 
                    seq[i+2] < CONTROL_OFFSET and
                    seq[i+2] != REST):  # Exclude REST tokens
                    score_positions.append((i, i+1, i+2))
                    i += 3
                else:
                    i += 1
            
            if len(score_positions) == 0:
                continue
            
            # Find the first score triplet position
            first_score_pos = score_positions[0][0]
            
            # Start with context up to first score triplet
            context = seq[:first_score_pos].tolist()
            
            # Autoregressively generate all score triplets
            for time_pos, dur_pos, pitch_pos in score_positions:
                # Generate up to the pitch position
                while len(context) <= pitch_pos:
                    # Get model prediction
                    input_tensor = torch.tensor([context]).to(accelerator.device)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        logits = outputs.logits[0, -1, :]
                        next_token = logits.argmax().item()
                    
                    context.append(next_token)
                
                # Check if the predicted pitch matches ground truth
                predicted_pitch = context[pitch_pos]
                true_pitch = seq[pitch_pos].item()
                
                if predicted_pitch == true_pitch:
                    autoregressive_correct += 1
                autoregressive_total += 1
    
    autoregressive_accuracy = autoregressive_correct / autoregressive_total if autoregressive_total > 0 else 0.0
    
    return avg_loss, teacher_forced_accuracy, autoregressive_accuracy

def plot_losses(train_losses, val_losses, val_accuracies, val_autoregressive_accuracies, validation_steps, output_dir):
    """
    Plot training/validation losses and validation pitch accuracy, save figures
    
    Args:
        train_losses (list): Training loss history
        val_losses (list): Validation loss history
        val_accuracies (list): Validation teacher-forced pitch accuracy history
        val_autoregressive_accuracies (list): Validation autoregressive pitch accuracy history
        validation_steps (list): Steps at which validation was performed
        output_dir (Path): Directory to save the plots
    """
    steps = list(range(1, len(train_losses) + 1))
    
    # Create figure with 4 subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # Plot 1: Linear loss plot
    ax1.plot(steps, train_losses, label='Training Loss', alpha=0.7, color='blue')
    ax1.scatter(validation_steps, val_losses, label='Validation Loss', color='red', s=30, zorder=5)
    ax1.plot(validation_steps, val_losses, alpha=0.3, color='red', linestyle='--')
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss (Linear Scale)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Log-log loss plot
    ax2.loglog(steps, train_losses, label='Training Loss', alpha=0.7, color='blue')
    ax2.scatter(validation_steps, val_losses, label='Validation Loss', color='red', s=30, zorder=5)
    ax2.plot(validation_steps, val_losses, alpha=0.3, color='red', linestyle='--')
    ax2.set_xlabel('Step (log scale)')
    ax2.set_ylabel('Loss (log scale)')
    ax2.set_title('Training and Validation Loss (Log-Log Scale)')
    ax2.legend()
    ax2.grid(True, alpha=0.3, which='both')
    
    # Plot 3: Validation teacher-forced pitch accuracy
    ax3.plot(validation_steps, val_accuracies, label='Teacher-Forced Pitch Accuracy', color='green', marker='o')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Pitch Accuracy (%)')
    ax3.set_title('Validation Teacher-Forced Pitch Accuracy')
    ax3.set_ylim([0, 100])
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Validation autoregressive pitch accuracy
    ax4.plot(validation_steps, val_autoregressive_accuracies, label='Autoregressive Pitch Accuracy', color='purple', marker='s')
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Pitch Accuracy (%)')
    ax4.set_title('Validation Autoregressive Pitch Accuracy')
    ax4.set_ylim([0, 100])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'training_metrics.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training metrics plot saved to {output_dir / 'training_metrics.png'}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=Path, default=Path('./data/train_normalized.txt'))
    parser.add_argument('--val_file', type=Path, default=Path('./data/test_normalized.txt'))
    parser.add_argument('--model_name', type=str, default='stanford-crfm/music-medium-800k')
    parser.add_argument('--output_dir', type=Path, default=Path('./fine_tuned_normalized'))
    parser.add_argument('--batch_size', type=int, default=8) 
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=64) 
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--max_steps', type=int, default=3500)
    parser.add_argument('--save_steps', type=int, default=250)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--warmup_steps', type=int, default=0)  # No warmup
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('--reduce_memory', action='store_true', help='Use memory-saving techniques')
    parser.add_argument('--perturb_std_ms', type=float, default=50.0, help='Standard deviation of time perturbation in milliseconds (training only)')
    parser.add_argument('--mask_prob', type=float, default=0.5, help='Probability of masking each control triplet (training only)')
    args = parser.parse_args()
    
    # Override device if requested
    global device
    if args.force_cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage as requested")
    
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Final device confirmation: {device}")
    
    try:
        # Initialize accelerator with memory optimization if requested
        # Use bf16 instead of fp16 for better numerical stability
        mixed_precision = 'bf16' if torch.cuda.is_available() and not args.force_cpu else 'no'
        print(f"Mixed precision mode: {mixed_precision}")
        
        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            cpu=args.force_cpu,
            mixed_precision=mixed_precision,
        )
        
        # Create output directory
        os.makedirs(args.output_dir, exist_ok=True)
        
        # Monitor initial GPU memory
        print("Initial GPU memory stats:")
        print_gpu_memory_stats()
        
        # Load training dataset
        print(f"Loading training dataset from {args.data_file}...")
        train_dataset = TokenizedDataset(
            args.data_file, 
            perturb_std_ms=args.perturb_std_ms,
            mask_prob=args.mask_prob,
            is_training=True
        )
        
        def collate_fn(batch):
            input_ids = torch.stack([item["input_ids"] for item in batch])
            labels = torch.stack([item["labels"] for item in batch])
            return {"input_ids": input_ids, "labels": labels}
            
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=args.batch_size, 
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available() and not args.force_cpu,
            num_workers=0,  # Avoid multiprocessing issues
        )
        
        # Load validation dataset (NO augmentation)
        print(f"Loading validation dataset from {args.val_file}...")
        val_dataset = TokenizedDataset(
            args.val_file,
            perturb_std_ms=0.0,
            mask_prob=0.0,
            is_training=False
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.val_batch_size,
            shuffle=False,  # No need to shuffle validation data
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available() and not args.force_cpu,
            num_workers=0,
        )
        
        # Load model with memory optimizations
        print(f"Loading model {args.model_name}...")
        model_kwargs = {
            "trust_remote_code": True,
            "use_cache": False,  # Important for training
        }
        
        if args.reduce_memory and torch.cuda.is_available():
            print("Using memory reduction techniques...")
            # BF16 is more stable than FP16
            model_kwargs.update({
                "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                "low_cpu_mem_usage": True,
            })
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                **model_kwargs
            )
        except Exception as e:
            print(f"Error loading model with advanced options: {e}")
            print("Trying with basic options...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                trust_remote_code=True,
                use_cache=False
            )
        
        # Resize model embeddings to match our vocabulary (VOCAB_SIZE=55028)
        from anticipation.vocab import VOCAB_SIZE
        current_vocab_size = model.config.vocab_size
        if current_vocab_size != VOCAB_SIZE:
            print(f"Resizing model embeddings from {current_vocab_size} to {VOCAB_SIZE}")
            model.resize_token_embeddings(VOCAB_SIZE)
            print(f"✓ Model embeddings resized successfully")
        else:
            print(f"✓ Model vocabulary size matches tokenization ({VOCAB_SIZE})")
        
        # Check memory after loading model
        print("GPU memory after loading model:")
        print_gpu_memory_stats()
        
        # DON'T manually move model to device - let accelerator handle it!
        # This is critical for multi-GPU training
        
        # Setup optimizer with gradient clipping to prevent exploding gradients
        # Using a lower learning rate and better epsilon value for numerical stability
        optimizer = AdamW(
            model.parameters(), 
            lr=args.learning_rate,
            eps=1e-6,  # More stable epsilon
            weight_decay=0.01,
            betas=(0.9, 0.999),  # Stable default betas
        )
        
        # Prepare for training with accelerate - this handles device placement
        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
        val_dataloader = accelerator.prepare_data_loader(val_dataloader)
        print(f"After accelerator preparation, model device: {next(model.parameters()).device}")
        
        # Learning rate scheduler - cosine decay from 3e-5 to 3e-6 (no warmup)
        initial_lr = args.learning_rate  # 3e-5
        final_lr = 3e-6
        
        # Custom cosine decay without warmup
        from torch.optim.lr_scheduler import LambdaLR
        import math
        
        def lr_lambda(current_step):
            # Pure cosine decay from start to finish
            progress = float(current_step) / float(max(1, args.max_steps))
            # Cosine annealing from 1.0 to (final_lr / initial_lr)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (final_lr / initial_lr) + (1.0 - final_lr / initial_lr) * cosine_decay
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        
        # Check memory before training
        print("GPU memory before training:")
        print_gpu_memory_stats()
        
        # Disable anomaly detection which can cause overhead
        torch.autograd.set_detect_anomaly(False)
        
        # Set deterministic algorithms for reproducibility
        torch.backends.cudnn.deterministic = False  # Better performance
        torch.backends.cudnn.benchmark = True  # Better performance
        
        if torch.cuda.is_available():
            print("Clearing CUDA cache before training")
            torch.cuda.empty_cache()
            torch.cuda.set_device(0)
        
        # Training loop
        print("Starting training...")
        model.train()
        completed_steps = 0
        step = 0
        
        # Lists to track losses and metrics
        train_losses = []
        val_losses = []
        val_accuracies = []
        val_autoregressive_accuracies = []
        validation_steps = []
        
        # Use standard tqdm with disable=False to ensure it always displays
        progress_bar = tqdm(total=args.max_steps, desc="Training", disable=False)
        
        try:
            while completed_steps < args.max_steps:
                for batch in train_dataloader:
                    try:
                        with accelerator.accumulate(model):
                            # Forward pass with gradient scaling
                            outputs = model(**batch)
                            loss = outputs.loss
                            
                            # Check for NaN loss
                            if torch.isnan(loss).any() or torch.isinf(loss).any():
                                print(f"WARNING: NaN or Inf loss detected: {loss.item()}")
                                # Skip this backward pass
                                optimizer.zero_grad()
                                continue
                                
                            # Backward pass
                            accelerator.backward(loss)
                            
                            # Only update optimizer and scheduler when gradients are synchronized
                            if accelerator.sync_gradients:
                                # Gradient clipping - industry standard value
                                accelerator.clip_grad_norm_(model.parameters(), max_norm=2.0)
                                
                                # Check for NaN in gradients
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
                                
                                # Only update optimizer and scheduler here
                                optimizer.step()
                                scheduler.step()
                                optimizer.zero_grad()
                                
                                # Only update step counters when we actually update weights
                                completed_steps += 1
                                progress_bar.update(1)
                                
                                # Log progress
                                if completed_steps % 10 == 0:
                                    # Store the training loss every 10 steps
                                    train_losses.append(loss.item())
                                    
                                    # Print more precise learning rate
                                    print(f"Step: {completed_steps}/{args.max_steps}, Loss: {loss.item():.4f}, "
                                          f"LR: {scheduler.get_last_lr()[0]:.8e}")
                                    
                                    # Check for NaN parameters periodically
                                    if check_model_for_nans(model):
                                        print("NaN parameters detected in model! Training may be unstable.")
                                    
                                    # Check memory periodically
                                    if completed_steps % 100 == 0:
                                        print_gpu_memory_stats()
                                
                                # Run validation periodically (but skip if we're about to checkpoint, which also validates)
                                is_checkpoint_step = (completed_steps % args.save_steps == 0)
                                if completed_steps % args.eval_steps == 0 and not is_checkpoint_step:
                                    print(f"\nRunning validation at step {completed_steps}...")
                                    val_loss, val_acc, val_auto_acc = evaluate_model(model, val_dataloader, accelerator)
                                    validation_steps.append(completed_steps // 10)  # Store step number (divided by 10 for plotting)
                                    val_losses.append(val_loss)
                                    val_accuracies.append(val_acc * 100)  # Store as percentage
                                    val_autoregressive_accuracies.append(val_auto_acc * 100)  # Store as percentage
                                    print(f"Validation Loss: {val_loss:.4f}, Teacher-Forced Accuracy: {val_acc*100:.2f}%, Autoregressive Accuracy: {val_auto_acc*100:.2f}%")
                                    
                                    # Return to training mode
                                    model.train()
                                    
                                    # Free up memory after validation
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                        gc.collect()
                                
                                # Save checkpoint (with validation)
                                if is_checkpoint_step:
                                    # Run validation before saving checkpoint
                                    print(f"\nRunning validation at checkpoint step {completed_steps}...")
                                    val_loss, val_acc, val_auto_acc = evaluate_model(model, val_dataloader, accelerator)
                                    validation_steps.append(completed_steps // 10)
                                    val_losses.append(val_loss)
                                    val_accuracies.append(val_acc * 100)
                                    val_autoregressive_accuracies.append(val_auto_acc * 100)
                                    print(f"Validation Loss: {val_loss:.4f}, Teacher-Forced Accuracy: {val_acc*100:.2f}%, Autoregressive Accuracy: {val_auto_acc*100:.2f}%")
                                    
                                    # Return to training mode
                                    model.train()
                                    
                                    checkpoint_dir = args.output_dir / f"checkpoint-{completed_steps}"
                                    os.makedirs(checkpoint_dir, exist_ok=True)
                                    
                                    # Unwrap model before saving
                                    unwrapped_model = accelerator.unwrap_model(model)
                                    unwrapped_model.save_pretrained(
                                        checkpoint_dir,
                                        is_main_process=accelerator.is_main_process,
                                        save_function=accelerator.save,
                                    )
                                    print(f"Saved checkpoint to {checkpoint_dir}")
                                    
                                    # Save the losses and metrics so far
                                    np.savez(
                                        checkpoint_dir / "losses.npz",
                                        train_losses=np.array(train_losses),
                                        val_losses=np.array(val_losses),
                                        val_accuracies=np.array(val_accuracies),
                                        val_autoregressive_accuracies=np.array(val_autoregressive_accuracies),
                                        validation_steps=np.array(validation_steps)
                                    )
                                    
                                    # Create and save loss plot
                                    plot_losses(train_losses, val_losses, val_accuracies, val_autoregressive_accuracies, validation_steps, checkpoint_dir)
                                    
                                    # Free up memory
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                        gc.collect()
                            
                            # Zero gradients even if we don't sync (needed for some accelerator configurations)
                            if not accelerator.sync_gradients:
                                optimizer.zero_grad()
                                
                            # Check if we've reached max steps
                            if completed_steps >= args.max_steps:
                                break
                            
                    except RuntimeError as e:
                        if "CUDA out of memory" in str(e):
                            print(f"CUDA OOM error! Current batch size: {args.batch_size}")
                            print(f"Current memory usage:")
                            print_gpu_memory_stats()
                            print("Consider reducing batch size or model size.")
                            print(f"Error details: {str(e)}")
                            raise
                        elif "nan" in str(e).lower() or "inf" in str(e).lower():
                            print(f"NaN/Inf error: {str(e)}")
                            print("Trying to recover by skipping this batch...")
                            optimizer.zero_grad()
                            continue
                        else:
                            print(f"Runtime error: {str(e)}")
                            print(traceback.format_exc())
                            raise
            
        except Exception as e:
            print(f"Error during training: {e}")
            print(traceback.format_exc())
            raise
        finally:
            # Make sure we always close the progress bar
            progress_bar.close()
            
            # Always try to save whatever we have and generate the final plot
            try:
                # Final validation run
                print("\nRunning final validation...")
                final_val_loss, final_val_acc, final_auto_acc = evaluate_model(model, val_dataloader, accelerator)
                validation_steps.append(completed_steps // 10)
                val_losses.append(final_val_loss)
                val_accuracies.append(final_val_acc * 100)
                val_autoregressive_accuracies.append(final_auto_acc * 100)
                print(f"Final validation Loss: {final_val_loss:.4f}, Teacher-Forced Accuracy: {final_val_acc*100:.2f}%, Autoregressive Accuracy: {final_auto_acc*100:.2f}%")
                
                # Final save
                final_dir = args.output_dir / "final"
                os.makedirs(final_dir, exist_ok=True)
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    final_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )
                print(f"Saved final model to {final_dir}")
                
                # Save the final losses
                np.savez(
                    final_dir / "losses.npz",
                    train_losses=np.array(train_losses),
                    val_losses=np.array(val_losses),
                    val_accuracies=np.array(val_accuracies),
                    val_autoregressive_accuracies=np.array(val_autoregressive_accuracies),
                    validation_steps=np.array(validation_steps)
                )
                
                # Create and save final loss plot
                plot_losses(train_losses, val_losses, val_accuracies, val_autoregressive_accuracies, validation_steps, final_dir)
                
            except Exception as save_error:
                print(f"Error saving final model or generating plot: {save_error}")
            
    except Exception as setup_error:
        print(f"Error in setup: {setup_error}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
