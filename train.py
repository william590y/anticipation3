import argparse
import json
import os
from pathlib import Path
import time
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
from anticipation.config import CONTEXT_SIZE, EVENT_SIZE

DEFAULT_NUM_WORKERS = 32

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
    print(f"CUDA is available with {device_count} device(s)")
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"  Device {i}: {device_name}")
        props = torch.cuda.get_device_properties(i)
        print(f"    - Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    - CUDA capability: {props.major}.{props.minor}")
else:
    device = torch.device("cpu")
    print("CUDA is not available! Training will be much slower on CPU.")

# Explicitly print which device we're using
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

PACKED_SEQUENCE_LENGTH = CONTEXT_SIZE - 4
PREFIX_CONTROLS = 33
ALTERNATING_START = PREFIX_CONTROLS * 2 * EVENT_SIZE

class TokenizedDataset(Dataset):
    """Dataset that loads clean sequences and applies augmentation on-the-fly.
    
    Sequences are packed and formatted by tokenize-combined.py:
    - Each sequence is exactly 1020 tokens
    - Format: [control/rest prefix..., alternating score/control...]
    - Augmentation (perturbation + masking) applied during training, not tokenization
    """
    def __init__(self, file_path, onset_jitter_std=0.0, dur_jitter_range=0.0,
                 mask_prob=0.0, transpose_range_semitones=0, tempo_scale_range=0.0,
                 is_training=True):
        self.onset_jitter_std = onset_jitter_std if is_training else 0.0
        self.dur_jitter_range = dur_jitter_range if is_training else 0.0
        self.mask_prob = mask_prob if is_training else 0.0
        self.transpose_range_semitones = transpose_range_semitones if is_training else 0
        self.tempo_scale_range = tempo_scale_range if is_training else 0.0
        self.is_training = is_training
        

        # Lazy loading: store byte offsets only — sequences are read on demand in __getitem__.
        # This keeps startup fast and RAM usage minimal regardless of file size.
        self.file_path = str(file_path)
        self.offsets = []          # byte offset of each non-empty line
        self.sequence_length = 0   # filled from first line

        print(f"Scanning {file_path} for line offsets...")
        with open(self.file_path, 'rb') as f:
            offset = 0
            for raw_line in f:
                stripped = raw_line.strip()
                if stripped:
                    self.offsets.append(offset)
                offset += len(raw_line)

        print(f"Found {len(self.offsets)} sequences")

        # Read a single line to determine sequence length and validate format
        if self.offsets:
            tokens = self._read_tokens(0)
            self.sequence_length = len(tokens)
            if self.sequence_length != PACKED_SEQUENCE_LENGTH:
                print(
                    f"Warning: Sequence length is {self.sequence_length}, "
                    f"expected {PACKED_SEQUENCE_LENGTH}"
                )
            elif self.sequence_length % 3 != 0:
                print(f"Warning: Sequence length {self.sequence_length} is not triplet-aligned")
            else:
                print("Tokenization format validated (triplet-aligned, no header tokens)")
            print(f"Sequence length: {self.sequence_length}")

        if self.is_training:
            print(f"  Training mode: onset_jitter_std={self.onset_jitter_std} (N(1,std²) IOI scaling), "
                  f"dur_jitter_range={self.dur_jitter_range} (U(1±range)), "
                  f"mask_prob={self.mask_prob} (score-history triplet dropout), "
                  f"transpose_range={self.transpose_range_semitones} semitones, "
                  f"tempo_scale_range=U(1±{self.tempo_scale_range})")
        else:
            print(f"  Validation mode: no augmentation")
    
    def __len__(self):
        return len(self.offsets)
    
    def _read_tokens(self, idx):
        """Read and parse tokens for sequence at index idx from disk."""
        with open(self.file_path, 'rb') as f:
            f.seek(self.offsets[idx])
            raw_line = f.readline().decode('utf-8').strip()
        if '|' in raw_line:
            token_str, _ = raw_line.split('|', 1)
            tokens = list(map(int, token_str.strip().split()))
        else:
            tokens = list(map(int, raw_line.split()))
        # Clamp negatives that can arise from tokenization bugs
        return [max(0, t) for t in tokens]
    
    def _augment_sequence(self, tokens):
        """Apply on-the-fly augmentation to a single training sequence.
        
        Global augmentations (one value sampled per sequence):
          - Transposition: uniform ±transpose_range semitones applied to all pitch tokens;
            pitches outside MIDI [0,127] are folded inward by octave steps.
          - Tempo scaling: λ ~ U(1-range, 1+range) scales all time/duration tokens.
        
        Local augmentations on control (performance) triplets:
          - Onset jitter:  ô_{i+1} - ô_i = (o_{i+1} - o_i) · N(1, std²)
            Each inter-onset interval is scaled by an independent Gaussian factor.
            Requires a two-pass approach: collect all onsets first, then reconstruct.
          - Duration jitter: each note duration scaled by U(1-range, 1+range).
        Local augmentation on score/output tokens:
          - Score-history dropout: replace prior score triplets with dedicated
            slot-specific mask tokens while preserving their labels.
        
        Returns:
            augmented_inputs: Tensor of augmented token ids for model input
            augmented_targets: Tensor of augmented token ids before history dropout
            concealed_indices: List of score-token positions replaced for history dropout
        """
        from anticipation.vocab import (CONTROL_OFFSET, SEPARATOR, REST,
                                        TIME_MASK, DUR_MASK, NOTE_MASK,
                                        ATIME_OFFSET, ADUR_OFFSET, ANOTE_OFFSET,
                                        TIME_OFFSET, DUR_OFFSET, NOTE_OFFSET)
        from anticipation.config import TIME_RESOLUTION, MAX_TIME, MAX_DUR, MAX_PITCH
        
        no_augmentation = (
            self.onset_jitter_std == 0 and
            self.dur_jitter_range == 0 and
            self.mask_prob == 0 and
            self.transpose_range_semitones == 0 and
            self.tempo_scale_range == 0.0
        )
        if not self.is_training or no_augmentation:
            cloned = tokens.clone()
            return cloned, cloned.clone(), []
        
        augmented = tokens.clone()
        concealed_indices = []
        
        # ── Global augmentation parameters (sampled once per sequence) ──────────
        
        # Transposition: uniform integer in [-range, +range] semitones
        transpose_shift = 0
        if self.transpose_range_semitones > 0:
            transpose_shift = torch.randint(
                -self.transpose_range_semitones,
                self.transpose_range_semitones + 1,
                (1,)
            ).item()
        
        # Tempo scaling: λ ~ U(1 - range, 1 + range)
        tempo_factor = 1.0
        if self.tempo_scale_range > 0.0:
            tempo_factor = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.tempo_scale_range
        
        MIDI_MIN, MIDI_MAX = 0, MAX_PITCH - 1  # 0..127
        
        def _transpose_note(raw_tok, note_base):
            """Transpose a note token; fold out-of-range pitches inward by octave steps."""
            raw_note = raw_tok - note_base
            instr = raw_note // MAX_PITCH
            pitch = raw_note % MAX_PITCH
            new_pitch = pitch + transpose_shift
            while new_pitch > MIDI_MAX:
                new_pitch -= 12
            while new_pitch < MIDI_MIN:
                new_pitch += 12
            new_pitch = max(MIDI_MIN, min(MIDI_MAX, new_pitch))  # safety clamp
            return note_base + instr * MAX_PITCH + new_pitch
        
        def _scale_time(raw_tok, time_base):
            """Apply global tempo scaling to a time token."""
            t = int(round((raw_tok - time_base) * tempo_factor))
            return time_base + max(0, min(MAX_TIME - 1, t))
        
        def _scale_dur(raw_tok, dur_base):
            """Apply global tempo scaling to a duration token."""
            d = int(round((raw_tok - dur_base) * tempo_factor))
            return dur_base + max(0, min(MAX_DUR - 1, d))
        
        # ── Pass 1: handle event triplets immediately; collect control triplets ──
        # IOI-based onset jitter requires all control onsets before writing back,
        # so control triplets are deferred to a second pass.
        ctrl_positions = []  # list of (seq_pos, tok0, tok1, tok2)
        
        i = 0
        while i < len(augmented) - 2:
            tok0 = augmented[i].item()
            tok1 = augmented[i + 1].item()
            tok2 = augmented[i + 2].item()
            
            is_event_triplet = (tok0 < CONTROL_OFFSET and
                                tok1 < CONTROL_OFFSET and
                                tok2 < CONTROL_OFFSET)
            is_control_triplet = (tok0 >= CONTROL_OFFSET and
                                  tok1 >= CONTROL_OFFSET and
                                  tok2 >= CONTROL_OFFSET and
                                  tok0 != SEPARATOR)
            
            if is_event_triplet:
                # Apply global augmentations and write back immediately
                if tempo_factor != 1.0:
                    tok0 = _scale_time(tok0, TIME_OFFSET)
                    tok1 = _scale_dur(tok1, DUR_OFFSET)
                if transpose_shift != 0 and tok2 != REST:
                    tok2 = _transpose_note(tok2, NOTE_OFFSET)
                augmented[i] = tok0
                augmented[i + 1] = tok1
                augmented[i + 2] = tok2
                i += 3
            elif is_control_triplet:
                ctrl_positions.append((i, tok0, tok1, tok2))
                i += 3
            else:
                # SEPARATOR, padding, or other special token – skip
                i += 1
        
        # ── Pass 2: compute IOI-jittered onset times for control triplets ────────
        # ô_{i+1} - ô_i = (o_{i+1} - o_i) · N(1, std²)
        new_ctrl_times = None
        if self.onset_jitter_std > 0 and len(ctrl_positions) >= 2:
            raw_times = [pos[1] - ATIME_OFFSET for pos in ctrl_positions]
            new_t = float(raw_times[0])  # first onset unchanged
            jittered = [new_t]
            for k in range(1, len(raw_times)):
                ioi = raw_times[k] - raw_times[k - 1]
                scale = 1.0 + torch.randn(1).item() * self.onset_jitter_std
                new_t = new_t + ioi * scale
                jittered.append(new_t)
            new_ctrl_times = [max(0, min(MAX_TIME - 1, int(round(t)))) for t in jittered]
        
        # ── Pass 3: write back all control triplet modifications ──────────────────
        for k, (pos_i, tok0, tok1, tok2) in enumerate(ctrl_positions):
            # IOI-based onset jitter
            if new_ctrl_times is not None:
                tok0 = ATIME_OFFSET + new_ctrl_times[k]
            
            # Duration jitter: scale by U(1 - range, 1 + range) per note
            if self.dur_jitter_range > 0:
                d_factor = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.dur_jitter_range
                base_dur = tok1 - ADUR_OFFSET
                tok1 = ADUR_OFFSET + max(0, min(MAX_DUR - 1, int(round(base_dur * d_factor))))
            
            # Global tempo scaling (applied after local jitter)
            if tempo_factor != 1.0:
                tok0 = _scale_time(tok0, ATIME_OFFSET)
                tok1 = _scale_dur(tok1, ADUR_OFFSET)
            
            # Global transposition
            if transpose_shift != 0:
                tok2 = _transpose_note(tok2, ANOTE_OFFSET)
            
            augmented[pos_i] = tok0
            augmented[pos_i + 1] = tok1
            augmented[pos_i + 2] = tok2
        
        augmented_targets = augmented.clone()

        if self.mask_prob > 0:
            i = ALTERNATING_START
            while i < len(augmented) - 2:
                tok0 = augmented[i].item()
                tok1 = augmented[i + 1].item()
                tok2 = augmented[i + 2].item()

                is_score_triplet = (
                    tok0 < CONTROL_OFFSET and
                    tok1 < CONTROL_OFFSET and
                    tok2 < CONTROL_OFFSET and
                    tok2 != REST
                )

                if is_score_triplet and torch.rand(1).item() < self.mask_prob:
                    augmented[i] = TIME_MASK
                    augmented[i + 1] = DUR_MASK
                    augmented[i + 2] = NOTE_MASK
                    concealed_indices.extend((i, i + 1, i + 2))
                i += 3
        
        return augmented, augmented_targets, concealed_indices
    
    def __getitem__(self, idx):
        tokens = torch.tensor(self._read_tokens(idx), dtype=torch.long)
        
        # Apply on-the-fly augmentation (time perturbation + masking)
        augmented_tokens, augmented_labels, _concealed_idxs = self._augment_sequence(tokens)
        
        # Safety check: clamp all tokens to valid range [0, VOCAB_SIZE-1]
        from anticipation.vocab import VOCAB_SIZE
        augmented_tokens = torch.clamp(augmented_tokens, 0, VOCAB_SIZE - 1)
        
        # Keep all positions active; concealed history uses a placeholder token.
        attention_mask = torch.ones_like(augmented_tokens)
        
        # Supervise the augmented sequence; only score-history dropout corrupts inputs.
        labels = augmented_labels
        
        return {"input_ids": augmented_tokens, "attention_mask": attention_mask, "labels": labels}


class CurriculumDataset(Dataset):
    """Curriculum dataset that linearly transitions from ATEPP (low quality) to ASAP (high quality).

    The training sequence is pre-planned: at position i out of total_samples,
    a sample is drawn from ASAP with probability p = i / (total_samples - 1),
    rising linearly from 0.0 to 1.0.  The DataLoader MUST use shuffle=False so
    that earlier batches are ATEPP-heavy and later batches are ASAP-heavy.
    """

    def __init__(self, asap_dataset, atepp_dataset, total_samples, seed=42):
        self.asap = asap_dataset
        self.atepp = atepp_dataset

        n_asap = len(asap_dataset)
        n_atepp = len(atepp_dataset)
        if n_asap == 0:
            raise ValueError("ASAP dataset is empty")
        if n_atepp == 0:
            raise ValueError("ATEPP dataset is empty")


        rng = np.random.default_rng(seed)

        # p_asap rises linearly from 0 → 1 over total_samples positions
        probs = np.linspace(0.0, 1.0, total_samples)
        draws = rng.random(total_samples)
        self._use_asap = draws < probs  # bool array, True → draw from ASAP

        # Pre-sample local indices (with replacement) for each dataset
        self._asap_idx = rng.integers(0, n_asap, size=total_samples).astype(np.int32)
        self._atepp_idx = rng.integers(0, n_atepp, size=total_samples).astype(np.int32)

        n_from_asap = int(self._use_asap.sum())
        n_from_atepp = total_samples - n_from_asap
        print(f"Curriculum plan: {total_samples:,} total samples")
        print(f"  ASAP  (high quality): {n_from_asap:,} ({100 * n_from_asap / total_samples:.1f}%) — used more at end")
        print(f"  ATEPP (low  quality): {n_from_atepp:,} ({100 * n_from_atepp / total_samples:.1f}%) — used more at start")

    def __len__(self):
        return len(self._use_asap)

    def __getitem__(self, idx):
        if self._use_asap[idx]:
            return self.asap[int(self._asap_idx[idx])]
        else:
            return self.atepp[int(self._atepp_idx[idx])]


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
    local_total_loss = 0.0
    local_total_samples = 0
    
    # For teacher-forced pitch accuracy: track predictions on score note tokens
    local_correct_pitches = 0
    local_total_pitches = 0
    
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
        for batch_idx, batch in enumerate(
            tqdm(
                dataloader,
                desc="Evaluating",
                leave=False,
                disable=not accelerator.is_local_main_process,
            )
        ):
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
            local_total_loss += loss.item() * batch_size
            local_total_samples += batch_size
            
            # Calculate pitch accuracy on score note tokens (position 2 of score triplets)
            # Score triplets have all tokens < CONTROL_OFFSET
            # IMPORTANT: Must iterate in triplet-aligned steps to avoid mis-identifying triplets
            for b in range(batch_size):
                seq_input = input_ids[b]
                seq_labels = labels[b]
                seq_logits = logits[b]
                
                # Sequence is triplet-aligned from token 0
                i = 0
                while i < len(seq_input) - 2:
                    # Check if this is a score triplet (all 3 tokens < CONTROL_OFFSET, but not REST)
                    if (seq_input[i] < CONTROL_OFFSET and 
                        seq_input[i+1] < CONTROL_OFFSET and 
                        seq_input[i+2] < CONTROL_OFFSET and
                        seq_input[i+2] != REST):  # Exclude REST triplets
                        # This is a score triplet
                        # Position i+2 is the note token
                        note_pos = i + 2
                        
                        # Only count if not masked in labels
                        if seq_labels[note_pos] != -100:
                            predicted_token = seq_logits[note_pos - 1].argmax().item()  # Predict next token
                            true_token = seq_labels[note_pos].item()
                            
                            # Check if prediction matches ground truth
                            if predicted_token == true_token:
                                local_correct_pitches += 1
                            local_total_pitches += 1
                    
                    # Always move in steps of 3 to maintain triplet alignment
                    i += 3
    
    local_teacher_forced_metrics = torch.tensor(
        [
            local_total_loss,
            float(local_total_samples),
            float(local_correct_pitches),
            float(local_total_pitches),
        ],
        dtype=torch.float64,
        device=accelerator.device,
    )
    gathered_teacher_forced_metrics = accelerator.gather_for_metrics(local_teacher_forced_metrics)
    if gathered_teacher_forced_metrics.ndim == 1:
        global_teacher_forced_metrics = gathered_teacher_forced_metrics
    else:
        global_teacher_forced_metrics = gathered_teacher_forced_metrics.sum(dim=0)

    global_total_loss = global_teacher_forced_metrics[0].item()
    global_total_samples = int(global_teacher_forced_metrics[1].item())
    global_correct_pitches = int(global_teacher_forced_metrics[2].item())
    global_total_pitches = int(global_teacher_forced_metrics[3].item())

    avg_loss = global_total_loss / global_total_samples if global_total_samples > 0 else 0.0
    teacher_forced_accuracy = (
        global_correct_pitches / global_total_pitches if global_total_pitches > 0 else 0.0
    )
    
    # Autoregressive evaluation: use performance (control) to generate score
    # Format: control+rest pairs + alternating score/control
    # Goal: Given the performance context, autoregressively generate score triplets
    autoregressive_correct = 0
    autoregressive_total = 0
    
    if autoregressive_samples > 0:
        # Sample validation sequences across the whole dataset instead of taking the
        # file head, which can be biased by tokenization/write order.
        all_sequences = []
        dataset = getattr(dataloader, "dataset", None)
        if dataset is not None and hasattr(dataset, "__len__") and hasattr(dataset, "__getitem__"):
            seq_count = len(dataset)
            if seq_count > 0:
                sample_count = min(autoregressive_samples, seq_count)
                rng = random.Random(0)
                sampled_indices = rng.sample(range(seq_count), sample_count)
                # Shard sampled indices across processes so autoregressive eval runs in parallel.
                local_sampled_indices = sampled_indices[
                    accelerator.process_index::accelerator.num_processes
                ]
                for idx in local_sampled_indices:
                    all_sequences.append(dataset[idx]["input_ids"])
        else:
            # Fallback for unusual dataloaders without a readable dataset.
            seen_sequences = 0
            with torch.no_grad():
                for batch in dataloader:
                    input_ids = batch["input_ids"]
                    for seq in input_ids:
                        if seen_sequences % accelerator.num_processes == accelerator.process_index:
                            all_sequences.append(seq)
                        seen_sequences += 1
                        if len(all_sequences) >= autoregressive_samples:
                            break
                    if len(all_sequences) >= autoregressive_samples:
                        break
        
        # Run autoregressive generation on each sampled sequence
        for seq in tqdm(
            all_sequences,
            desc="Autoregressive eval",
            leave=False,
            disable=not accelerator.is_local_main_process,
        ):
            # Sequence format: [control+rest pairs (positions 0-197), alternating score/control (198+)]
            # We want to use the control tokens as context and generate the score tokens
            
            # Find where alternating section starts (position 198)
            alternating_start = ALTERNATING_START
            if len(seq) <= alternating_start:
                continue
            
            # Start context with all control+rest pairs
            # This gives the model all the performance information
            context = seq[:alternating_start].tolist()
            
            # Now autoregressively generate the alternating score/control section
            # Pattern: score_triplet, control_triplet, score_triplet, control_triplet, ...
            pos = alternating_start
            while pos + 5 < len(seq):
                # Check if this is a score triplet (all 3 tokens < CONTROL_OFFSET)
                if (seq[pos] < CONTROL_OFFSET and 
                    seq[pos+1] < CONTROL_OFFSET and 
                    seq[pos+2] < CONTROL_OFFSET and
                    seq[pos+2] != REST):
                    
                    # This is a score triplet - generate it autoregressively
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
                    # (We're only testing score generation, not control generation)
                    if pos + 2 < len(seq):
                        context.extend([seq[pos].item(), seq[pos+1].item(), seq[pos+2].item()])
                        pos += 3
                else:
                    # Not a score triplet, add to context and continue
                    context.append(seq[pos].item())
                    pos += 1
    
    local_autoregressive_metrics = torch.tensor(
        [float(autoregressive_correct), float(autoregressive_total)],
        dtype=torch.float64,
        device=accelerator.device,
    )
    gathered_autoregressive_metrics = accelerator.gather_for_metrics(local_autoregressive_metrics)
    if gathered_autoregressive_metrics.ndim == 1:
        global_autoregressive_metrics = gathered_autoregressive_metrics
    else:
        global_autoregressive_metrics = gathered_autoregressive_metrics.sum(dim=0)

    global_autoregressive_correct = int(global_autoregressive_metrics[0].item())
    global_autoregressive_total = int(global_autoregressive_metrics[1].item())
    autoregressive_accuracy = (
        global_autoregressive_correct / global_autoregressive_total
        if global_autoregressive_total > 0
        else 0.0
    )

    # #region agent log
    if accelerator.is_main_process:
        _debug_log_path = os.path.join(os.path.dirname(__file__), "debug-e30de5.log")
        try:
            with open(_debug_log_path, "a", encoding="utf-8") as _f:
                _f.write(json.dumps({"sessionId": "e30de5", "timestamp": time.time() * 1000, "location": "train.py:evaluate_model", "message": "train_ar_metrics", "data": {"protocol": "train_gt_control", "uses_gt_control": True, "autoregressive_correct": global_autoregressive_correct, "autoregressive_total": global_autoregressive_total, "autoregressive_acc_pct": round(autoregressive_accuracy * 100, 2)}, "hypothesisId": "A"}) + "\n")
        except Exception:
            pass
    # #endregion

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
    parser.add_argument('--data_file', type=Path, default=Path('./data/train_combined_01b10e4_hybrid.txt'))
    parser.add_argument('--val_file', type=Path, default=Path('./data/test_combined_01b10e4_hybrid.txt'))
    parser.add_argument('--asap_file', type=Path, default=Path('./data/train_asap_01b10e4_hybrid.txt'),
                        help='ASAP-only training sequences (high quality, used at end of curriculum)')
    parser.add_argument('--atepp_file', type=Path, default=Path('./data/train_atepp_01b10e4_hybrid.txt'),
                        help='ATEPP-only training sequences (low quality, used at start of curriculum)')
    parser.add_argument('--curriculum', action='store_true',
                        help='Enable curriculum learning: linear transition from ATEPP to ASAP over training')
    parser.add_argument('--model_name', type=str, default='stanford-crfm/music-medium-800k')
    parser.add_argument('--output_dir', type=Path, default=Path('./model_01b10e4_hybrid'))
    parser.add_argument('--batch_size', type=int, default=8) 
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=64) 
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--max_steps', type=int, default=4000)
    parser.add_argument('--save_steps', type=int, default=250)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument(
        '--log_every_steps',
        type=int,
        default=1,
        help='Print training loss every N optimizer steps (default: 1)',
    )
    parser.add_argument('--warmup_steps', type=int, default=0)  # No warmup
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('--reduce_memory', action='store_true', help='Use memory-saving techniques')
    parser.add_argument('--onset_jitter_std', type=float, default=0.1,
                        help='Std of N(1, std²) multiplier applied to each inter-onset interval of control tokens (training only)')
    parser.add_argument('--dur_jitter_range', type=float, default=0.1,
                        help='Half-range of U(1-r, 1+r) duration rescaling per control note, e.g. 0.05 gives U(0.95, 1.05) (training only)')
    parser.add_argument('--mask_prob', type=float, default=0.0, help='Probability of concealing prior score/output triplets during training (0.0 to 1.0)')
    parser.add_argument('--transpose_range_semitones', type=int, default=12,
                        help='Max transposition shift in semitones, uniform in [-range, +range] (training only)')
    parser.add_argument('--tempo_scale_range', type=float, default=0.2,
                        help='Tempo scale half-range: lambda ~ U(1-range, 1+range), e.g. 0.2 gives U(0.8,1.2) (training only)')
    parser.add_argument('--num_workers', type=int, default=DEFAULT_NUM_WORKERS,
                        help='DataLoader worker processes')
    args = parser.parse_args()
    
    # Override device if requested
    global device
    if args.force_cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage as requested")
    
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Final device confirmation: {device}")
    print(
        f"Training loss will be printed every {args.log_every_steps} optimizer step(s) "
        f"({args.log_every_steps * args.gradient_accumulation_steps} micro-batch(es) "
        f"with gradient accumulation)."
    )
    
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
        def collate_fn(batch):
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            labels = torch.stack([item["labels"] for item in batch])
            return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        dataset_kwargs = dict(
            onset_jitter_std=args.onset_jitter_std,
            dur_jitter_range=args.dur_jitter_range,
            mask_prob=args.mask_prob,
            transpose_range_semitones=args.transpose_range_semitones,
            tempo_scale_range=args.tempo_scale_range,
            is_training=True,
        )

        if args.curriculum:
            if not args.asap_file.exists():
                raise FileNotFoundError(
                    f"Curriculum mode requires {args.asap_file}. "
                    "Re-run tokenize-combined.py to generate per-source files."
                )
            if not args.atepp_file.exists():
                raise FileNotFoundError(
                    f"Curriculum mode requires {args.atepp_file}. "
                    "Re-run tokenize-combined.py to generate per-source files."
                )
            print(f"Curriculum learning enabled.")
            print(f"  Loading ASAP dataset (high quality) from {args.asap_file}...")
            asap_dataset = TokenizedDataset(args.asap_file, **dataset_kwargs)
            print(f"  Loading ATEPP dataset (low  quality) from {args.atepp_file}...")
            atepp_dataset = TokenizedDataset(args.atepp_file, **dataset_kwargs)
            # Cover the full global training run so one logical pass spans all
            # micro-batches across every distributed process.
            total_samples = (
                args.max_steps
                * args.gradient_accumulation_steps
                * args.batch_size
                * accelerator.num_processes
            )
            train_dataset = CurriculumDataset(asap_dataset, atepp_dataset, total_samples)
            shuffle_train = False  # order must be preserved for curriculum
        else:
            print(f"Loading training dataset from {args.data_file}...")
            train_dataset = TokenizedDataset(args.data_file, **dataset_kwargs)
            shuffle_train = True

        dataloader_kwargs = dict(
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available() and not args.force_cpu,
            num_workers=args.num_workers,
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=shuffle_train,
            **dataloader_kwargs,
        )
        
        # Load validation dataset (NO augmentation)
        print(f"Loading validation dataset from {args.val_file}...")
        val_dataset = TokenizedDataset(
            args.val_file,
            is_training=False
        )
        
        val_dataloader = DataLoader(
            val_dataset, 
            batch_size=args.val_batch_size,
            shuffle=False,  # No need to shuffle validation data
            **dataloader_kwargs,
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
            print("Model embeddings resized successfully")
        else:
            print(f"Model vocabulary size matches tokenization ({VOCAB_SIZE})")
        
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
        
        # Training loop
        print("Starting training...")
        model.train()
        optimizer.zero_grad(set_to_none=True)
        completed_steps = 0
        micro_batches_seen = 0
        accumulation_loss_sum = 0.0
        accumulation_loss_count = 0
        
        # Lists to track losses and metrics
        train_losses = []
        val_losses = []
        val_accuracies = []
        val_autoregressive_accuracies = []
        validation_steps = []
        
        # Use standard tqdm with disable=False to ensure it always displays
        progress_bar = tqdm(
            total=args.max_steps,
            desc="Training",
            disable=not accelerator.is_local_main_process,
        )
        training_completed = False
        
        try:
            while completed_steps < args.max_steps:
                for batch in train_dataloader:
                    try:
                        with accelerator.accumulate(model):
                            # Forward pass with gradient scaling
                            outputs = model(**batch)
                            loss = outputs.loss
                            micro_batches_seen += 1
                            
                            # Check for NaN loss
                            if torch.isnan(loss).any() or torch.isinf(loss).any():
                                print(f"WARNING: NaN or Inf loss detected: {loss.item()}")
                                # Skip this backward pass
                                optimizer.zero_grad(set_to_none=True)
                                accumulation_loss_sum = 0.0
                                accumulation_loss_count = 0
                                continue

                            accumulation_loss_sum += loss.detach().item()
                            accumulation_loss_count += 1

                            if accelerator.is_local_main_process:
                                accumulation_step = (
                                    (micro_batches_seen - 1) % args.gradient_accumulation_steps
                                ) + 1
                                progress_bar.set_postfix(
                                    loss=f"{(accumulation_loss_sum / accumulation_loss_count):.4f}",
                                    accum=f"{accumulation_step}/{args.gradient_accumulation_steps}",
                                    opt_step=completed_steps,
                                )
                                
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
                                    optimizer.zero_grad(set_to_none=True)
                                    accumulation_loss_sum = 0.0
                                    accumulation_loss_count = 0
                                    continue
                                
                                # Only update optimizer and scheduler here
                                optimizer.step()
                                scheduler.step()
                                optimizer.zero_grad(set_to_none=True)
                                accumulation_loss_metrics = torch.tensor(
                                    [
                                        accumulation_loss_sum,
                                        float(accumulation_loss_count),
                                    ],
                                    dtype=torch.float64,
                                    device=accelerator.device,
                                )
                                global_accumulation_loss_metrics = accelerator.reduce(
                                    accumulation_loss_metrics,
                                    reduction="sum",
                                )
                                reduced_loss = (
                                    global_accumulation_loss_metrics[0].item()
                                    / max(global_accumulation_loss_metrics[1].item(), 1.0)
                                )
                                accumulation_loss_sum = 0.0
                                accumulation_loss_count = 0
                                
                                # Only update step counters when we actually update weights
                                completed_steps += 1
                                progress_bar.update(1)
                                train_losses.append(reduced_loss)
                                
                                # Log progress
                                if (
                                    accelerator.is_main_process
                                    and (
                                        completed_steps == 1
                                        or completed_steps % args.log_every_steps == 0
                                    )
                                ):
                                    # Print more precise learning rate
                                    tqdm.write(
                                        f"Step: {completed_steps}/{args.max_steps}, Loss: {reduced_loss:.4f}, "
                                        f"LR: {scheduler.get_last_lr()[0]:.8e}"
                                    )
                                    
                                    # Check for NaN parameters periodically
                                    if check_model_for_nans(model):
                                        print("NaN parameters detected in model! Training may be unstable.")
                                    
                                    # Check memory periodically
                                    if completed_steps % 100 == 0:
                                        print_gpu_memory_stats()
                                
                                # Run validation periodically (but skip if we're about to checkpoint, which also validates)
                                is_checkpoint_step = (completed_steps % args.save_steps == 0)
                                if completed_steps % args.eval_steps == 0 and not is_checkpoint_step:
                                    accelerator.print(f"\nRunning validation at step {completed_steps}...")
                                    val_loss, val_acc, val_auto_acc = evaluate_model(model, val_dataloader, accelerator)
                                    validation_steps.append(completed_steps)
                                    val_losses.append(val_loss)
                                    val_accuracies.append(val_acc * 100)  # Store as percentage
                                    val_autoregressive_accuracies.append(val_auto_acc * 100)  # Store as percentage
                                    accelerator.print(f"Validation Loss: {val_loss:.4f}, Teacher-Forced Accuracy: {val_acc*100:.2f}%, Autoregressive Accuracy: {val_auto_acc*100:.2f}%")
                                    
                                    # Return to training mode
                                    model.train()
                                    
                                    # Free up memory after validation
                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                        gc.collect()
                                
                                # Save checkpoint (with validation)
                                if is_checkpoint_step:
                                    # Run validation before saving checkpoint
                                    accelerator.print(f"\nRunning validation at checkpoint step {completed_steps}...")
                                    val_loss, val_acc, val_auto_acc = evaluate_model(model, val_dataloader, accelerator)
                                    validation_steps.append(completed_steps)
                                    val_losses.append(val_loss)
                                    val_accuracies.append(val_acc * 100)
                                    val_autoregressive_accuracies.append(val_auto_acc * 100)
                                    accelerator.print(f"Validation Loss: {val_loss:.4f}, Teacher-Forced Accuracy: {val_acc*100:.2f}%, Autoregressive Accuracy: {val_auto_acc*100:.2f}%")
                                    
                                    # Return to training mode
                                    model.train()
                                    
                                    checkpoint_dir = args.output_dir / f"checkpoint-{completed_steps}"
                                    if accelerator.is_main_process:
                                        os.makedirs(checkpoint_dir, exist_ok=True)
                                    accelerator.wait_for_everyone()
                                    
                                    # Unwrap model before saving
                                    unwrapped_model = accelerator.unwrap_model(model)
                                    unwrapped_model.save_pretrained(
                                        checkpoint_dir,
                                        is_main_process=accelerator.is_main_process,
                                        save_function=accelerator.save,
                                    )
                                    accelerator.wait_for_everyone()
                                    if accelerator.is_main_process:
                                        print(f"Saved checkpoint to {checkpoint_dir}")
                                    
                                    # Save the losses and metrics so far
                                    if accelerator.is_main_process:
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
                            optimizer.zero_grad(set_to_none=True)
                            accumulation_loss_sum = 0.0
                            accumulation_loss_count = 0
                            continue
                        else:
                            print(f"Runtime error: {str(e)}")
                            print(traceback.format_exc())
                            raise

            training_completed = completed_steps >= args.max_steps
            
        except Exception as e:
            print(f"Error during training: {e}")
            print(traceback.format_exc())
            raise
        finally:
            # Make sure we always close the progress bar
            progress_bar.close()
            
            if not training_completed:
                accelerator.print(
                    "Skipping final validation/save because training did not complete cleanly."
                )
                return

            # Save the final state and generate the final plot after a clean run
            try:
                # Final validation run
                accelerator.print("\nRunning final validation...")
                final_val_loss, final_val_acc, final_auto_acc = evaluate_model(model, val_dataloader, accelerator)
                validation_steps.append(completed_steps)
                val_losses.append(final_val_loss)
                val_accuracies.append(final_val_acc * 100)
                val_autoregressive_accuracies.append(final_auto_acc * 100)
                accelerator.print(f"Final validation Loss: {final_val_loss:.4f}, Teacher-Forced Accuracy: {final_val_acc*100:.2f}%, Autoregressive Accuracy: {final_auto_acc*100:.2f}%")
                
                # Final save
                final_dir = args.output_dir / "final"
                if accelerator.is_main_process:
                    os.makedirs(final_dir, exist_ok=True)
                accelerator.wait_for_everyone()
                unwrapped_model = accelerator.unwrap_model(model)
                unwrapped_model.save_pretrained(
                    final_dir,
                    is_main_process=accelerator.is_main_process,
                    save_function=accelerator.save,
                )
                accelerator.wait_for_everyone()
                if accelerator.is_main_process:
                    print(f"Saved final model to {final_dir}")
                
                # Save the final losses
                if accelerator.is_main_process:
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
