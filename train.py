import argparse
import os
from pathlib import Path
import random
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import numpy as np
from accelerate import Accelerator
from transformers import AutoModelForCausalLM
from torch.optim import AdamW
from tqdm import tqdm
import gc
import traceback
import matplotlib.pyplot as plt
import warnings
from anticipation.config import CONTEXT_SIZE, EVENT_SIZE

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
    print(f"[ok] CUDA is available with {device_count} device(s)")
    for i in range(device_count):
        device_name = torch.cuda.get_device_name(i)
        print(f"  Device {i}: {device_name}")
        props = torch.cuda.get_device_properties(i)
        print(f"    - Total memory: {props.total_memory / 1024**3:.2f} GB")
        print(f"    - CUDA capability: {props.major}.{props.minor}")
else:
    device = torch.device("cpu")
    print("[warn] CUDA is not available! Training will be much slower on CPU.")

# Explicitly print which device we're using
print(f"Using device: {device}")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA version: {torch.version.cuda}")

PACKED_SEQUENCE_LENGTH = CONTEXT_SIZE - 4
PREFIX_CONTROLS = 33
ALTERNATING_START = PREFIX_CONTROLS * 2 * EVENT_SIZE

class TokenizedDataset(Dataset):
    """Dataset that loads packed ASAP sequences and applies augmentation on-the-fly."""

    def __init__(
        self,
        file_path,
        onset_jitter_std=0.0,
        dur_jitter_range=0.0,
        mask_prob=0.75,
        transpose_range_semitones=0,
        tempo_scale_range=0.0,
        loss_mask_performance_tokens=False,
        is_training=True,
    ):
        self.onset_jitter_std = onset_jitter_std if is_training else 0.0
        self.dur_jitter_range = dur_jitter_range if is_training else 0.0
        self.mask_prob = mask_prob if is_training else 0.0
        self.transpose_range_semitones = transpose_range_semitones if is_training else 0
        self.tempo_scale_range = tempo_scale_range if is_training else 0.0
        self.loss_mask_performance_tokens = loss_mask_performance_tokens
        self.is_training = is_training
        self._vocab_warning_counts = {}

        self.file_path = str(file_path)
        self.offsets = []
        self.sequence_length = 0

        print(f"Scanning {file_path} for line offsets...")
        with open(self.file_path, "rb") as f:
            offset = 0
            for raw_line in f:
                stripped = raw_line.strip()
                if stripped:
                    self.offsets.append(offset)
                offset += len(raw_line)

        print(f"Found {len(self.offsets)} sequences")

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
            print(
                "  Training mode: "
                f"onset_jitter_std={self.onset_jitter_std}, "
                f"dur_jitter_range={self.dur_jitter_range}, "
                f"score_mask_ratio={self.mask_prob}, "
                f"transpose_range={self.transpose_range_semitones}, "
                f"tempo_scale_range={self.tempo_scale_range}, "
                f"loss_mask_performance_tokens={self.loss_mask_performance_tokens}"
            )
        else:
            print(
                "  Validation mode: no augmentation, "
                f"loss_mask_performance_tokens={self.loss_mask_performance_tokens}"
            )

    def __len__(self):
        return len(self.offsets)

    def _read_tokens(self, idx):
        with open(self.file_path, "rb") as f:
            f.seek(self.offsets[idx])
            raw_line = f.readline().decode("utf-8").strip()
        if "|" in raw_line:
            token_str, _ = raw_line.split("|", 1)
            tokens = list(map(int, token_str.strip().split()))
        else:
            tokens = list(map(int, raw_line.split()))
        return tokens

    def _sample_augmentation_params(self):
        transpose_shift = 0
        if self.transpose_range_semitones > 0:
            transpose_shift = torch.randint(
                -self.transpose_range_semitones,
                self.transpose_range_semitones + 1,
                (1,),
            ).item()

        tempo_factor = 1.0
        if self.tempo_scale_range > 0.0:
            tempo_factor = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.tempo_scale_range

        return transpose_shift, tempo_factor

    def _augment_sequence(
        self,
        tokens,
        transpose_shift=0,
        tempo_factor=1.0,
        apply_timing_augmentation=True,
    ):
        from anticipation.vocab import (
            CONTROL_OFFSET,
            SEPARATOR,
            REST,
            ATIME_OFFSET,
            ADUR_OFFSET,
            ANOTE_OFFSET,
            TIME_OFFSET,
            DUR_OFFSET,
            NOTE_OFFSET,
        )
        from anticipation.config import MAX_TIME, MAX_DUR, MAX_PITCH

        if not self.is_training:
            return tokens.clone()

        augmented = tokens.clone()
        midi_min = 0
        midi_max = MAX_PITCH - 1

        def _transpose_note(raw_tok, note_base):
            raw_note = raw_tok - note_base
            instr = raw_note // MAX_PITCH
            pitch = raw_note % MAX_PITCH
            new_pitch = pitch + transpose_shift
            while new_pitch > midi_max:
                new_pitch -= 12
            while new_pitch < midi_min:
                new_pitch += 12
            new_pitch = max(midi_min, min(midi_max, new_pitch))
            return note_base + instr * MAX_PITCH + new_pitch

        def _scale_time(raw_tok, time_base):
            scaled = int(round((raw_tok - time_base) * tempo_factor))
            return time_base + max(0, min(MAX_TIME - 1, scaled))

        def _scale_dur(raw_tok, dur_base):
            scaled = int(round((raw_tok - dur_base) * tempo_factor))
            return dur_base + max(0, min(MAX_DUR - 1, scaled))

        ctrl_positions = []

        i = 0
        while i < len(augmented) - 2:
            tok0 = augmented[i].item()
            tok1 = augmented[i + 1].item()
            tok2 = augmented[i + 2].item()

            is_event_triplet = tok0 < CONTROL_OFFSET and tok1 < CONTROL_OFFSET and tok2 < CONTROL_OFFSET
            is_control_triplet = (
                tok0 >= CONTROL_OFFSET
                and tok1 >= CONTROL_OFFSET
                and tok2 >= CONTROL_OFFSET
                and tok0 != SEPARATOR
            )

            if is_event_triplet:
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
                i += 1

        new_ctrl_times = None
        if apply_timing_augmentation and self.onset_jitter_std > 0 and len(ctrl_positions) >= 2:
            raw_times = [pos[1] - ATIME_OFFSET for pos in ctrl_positions]
            new_time = float(raw_times[0])
            jittered = [new_time]
            for k in range(1, len(raw_times)):
                ioi = raw_times[k] - raw_times[k - 1]
                scale = 1.0 + torch.randn(1).item() * self.onset_jitter_std
                new_time = new_time + ioi * scale
                jittered.append(new_time)
            new_ctrl_times = [max(0, min(MAX_TIME - 1, int(round(t)))) for t in jittered]

        for k, (pos_i, tok0, tok1, tok2) in enumerate(ctrl_positions):
            if new_ctrl_times is not None:
                tok0 = ATIME_OFFSET + new_ctrl_times[k]

            if apply_timing_augmentation and self.dur_jitter_range > 0:
                dur_factor = 1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.dur_jitter_range
                base_dur = tok1 - ADUR_OFFSET
                tok1 = ADUR_OFFSET + max(0, min(MAX_DUR - 1, int(round(base_dur * dur_factor))))

            if apply_timing_augmentation and tempo_factor != 1.0:
                tok0 = _scale_time(tok0, ATIME_OFFSET)
                tok1 = _scale_dur(tok1, ADUR_OFFSET)

            if transpose_shift != 0:
                tok2 = _transpose_note(tok2, ANOTE_OFFSET)

            augmented[pos_i] = tok0
            augmented[pos_i + 1] = tok1
            augmented[pos_i + 2] = tok2

        return augmented

    def _build_score_token_mask(self, tokens):
        from anticipation.vocab import CONTROL_OFFSET, REST

        score_token_mask = torch.zeros_like(tokens, dtype=torch.bool)
        i = 0
        while i < len(tokens) - 2:
            tok0 = tokens[i].item()
            tok1 = tokens[i + 1].item()
            tok2 = tokens[i + 2].item()

            is_score_triplet = (
                tok0 < CONTROL_OFFSET
                and tok1 < CONTROL_OFFSET
                and tok2 < CONTROL_OFFSET
                and tok2 != REST
            )

            if is_score_triplet:
                score_token_mask[i:i + 3] = True
                i += 3
            else:
                i += 1

        return score_token_mask

    def _sample_score_mask(self, score_token_mask):
        score_mask = torch.zeros_like(score_token_mask, dtype=torch.bool)
        if not self.is_training or self.mask_prob <= 0:
            return score_mask

        prev_score_token_mask = torch.zeros_like(score_token_mask)
        prev_score_token_mask[1:] = score_token_mask[:-1]
        score_triplet_starts = torch.nonzero(
            score_token_mask & ~prev_score_token_mask,
            as_tuple=False,
        ).flatten()
        if len(score_triplet_starts) == 0:
            return score_mask

        num_to_mask = int(round(len(score_triplet_starts) * self.mask_prob))
        num_to_mask = max(0, min(len(score_triplet_starts), num_to_mask))
        if num_to_mask == 0:
            return score_mask

        selected = torch.randperm(len(score_triplet_starts))[:num_to_mask]
        for idx in selected.tolist():
            start = score_triplet_starts[idx].item()
            score_mask[start:start + 3] = True

        return score_mask

    def _build_performance_loss_mask(self, tokens):
        from anticipation.vocab import CONTROL_OFFSET, SEPARATOR

        loss_mask = torch.zeros_like(tokens, dtype=torch.bool)
        if not self.loss_mask_performance_tokens:
            return loss_mask

        i = 0
        while i < len(tokens) - 2:
            tok0 = tokens[i].item()
            tok1 = tokens[i + 1].item()
            tok2 = tokens[i + 2].item()

            is_control_triplet = (
                tok0 >= CONTROL_OFFSET
                and tok1 >= CONTROL_OFFSET
                and tok2 >= CONTROL_OFFSET
                and tok0 != SEPARATOR
            )

            if is_control_triplet:
                loss_mask[i:i + 3] = True
                i += 3
            else:
                i += 1

        return loss_mask

    def _clamp_tokens_to_vocab(self, tokens, tensor_name, sample_idx):
        from anticipation.vocab import VOCAB_SIZE

        invalid_mask = (tokens < 0) | (tokens >= VOCAB_SIZE)
        if not torch.any(invalid_mask).item():
            return tokens

        invalid_positions = torch.nonzero(invalid_mask, as_tuple=False).flatten()
        preview_positions = invalid_positions[:5].tolist()
        preview = ", ".join(
            f"pos {pos} -> {tokens[pos].item()}"
            for pos in preview_positions
        )
        invalid_count = int(invalid_mask.sum().item())
        warning_count = self._vocab_warning_counts.get(tensor_name, 0)
        if warning_count < 5:
            suffix = ""
            if warning_count == 4:
                suffix = " Further warnings for this tensor type will be suppressed."
            warnings.warn(
                f"{tensor_name} contains {invalid_count} out-of-vocab token(s) for sample {sample_idx}: "
                f"{preview}. Clamping to [0, {VOCAB_SIZE - 1}].{suffix}",
                RuntimeWarning,
                stacklevel=2,
            )
        self._vocab_warning_counts[tensor_name] = warning_count + 1
        return torch.clamp(tokens, 0, VOCAB_SIZE - 1)

    def __getitem__(self, idx):
        tokens = torch.tensor(self._read_tokens(idx), dtype=torch.long)
        tokens = self._clamp_tokens_to_vocab(tokens, "Raw tokens", idx)

        no_augmentation = (
            not self.is_training
            or (
                self.onset_jitter_std == 0
                and self.dur_jitter_range == 0
                and self.mask_prob == 0
                and self.transpose_range_semitones == 0
                and self.tempo_scale_range == 0.0
            )
        )

        if no_augmentation:
            augmented_tokens = tokens.clone()
            labels = tokens.clone()
        else:
            transpose_shift, tempo_factor = self._sample_augmentation_params()
            augmented_tokens = self._augment_sequence(
                tokens,
                transpose_shift=transpose_shift,
                tempo_factor=tempo_factor,
                apply_timing_augmentation=True,
            )
            labels = self._augment_sequence(
                tokens,
                transpose_shift=transpose_shift,
                tempo_factor=tempo_factor,
                apply_timing_augmentation=False,
            )

        augmented_tokens = self._clamp_tokens_to_vocab(augmented_tokens, "Augmented input tokens", idx)
        labels = self._clamp_tokens_to_vocab(labels, "Training labels", idx)

        score_token_mask = self._build_score_token_mask(augmented_tokens)
        score_mask = self._sample_score_mask(score_token_mask)

        performance_loss_mask = self._build_performance_loss_mask(labels)
        if torch.any(performance_loss_mask).item():
            labels[performance_loss_mask] = -100

        attention_mask = torch.ones_like(augmented_tokens)

        return {
            "input_ids": augmented_tokens,
            "attention_mask": attention_mask,
            "labels": labels,
            "score_token_mask": score_token_mask,
            "score_mask": score_mask,
        }

def _get_input_embedding_layer(model):
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    if hasattr(model, "module") and hasattr(model.module, "get_input_embeddings"):
        return model.module.get_input_embeddings()
    raise AttributeError("Model does not expose get_input_embeddings()")


def _count_flagged_processes(accelerator, flag):
    flag_tensor = torch.tensor(
        int(bool(flag)),
        device=accelerator.device,
        dtype=torch.int64,
    )
    reduced = accelerator.reduce(flag_tensor, reduction="sum")
    return int(reduced.item())


def _find_invalid_gradient_parameter(model):
    for name, param in model.named_parameters():
        if param.grad is None:
            continue
        if torch.isnan(param.grad).any() or torch.isinf(param.grad).any():
            return name
    return None


def forward_batch(model, batch):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    score_mask = batch.get("score_mask")

    if score_mask is None or not torch.any(score_mask).item():
        return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    inputs_embeds = _get_input_embedding_layer(model)(input_ids)
    inputs_embeds = inputs_embeds.masked_fill(score_mask.unsqueeze(-1), 0.0)
    return model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)


def _move_batch_to_device(batch, device):
    moved = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            moved[key] = value.to(device, non_blocking=True)
        else:
            moved[key] = value
    return moved


def _broadcast_int_from_main(accelerator, value):
    value_tensor = torch.tensor(
        int(value) if accelerator.is_main_process else 0,
        device=accelerator.device,
        dtype=torch.int64,
    )
    synchronized = accelerator.reduce(value_tensor, reduction="sum")
    return int(synchronized.item())


def _resolve_eval_subset_size(dataset_size, requested_size, world_size):
    if dataset_size <= 0:
        raise ValueError("Validation dataset is empty.")

    sample_count = dataset_size if requested_size is None or requested_size <= 0 else min(requested_size, dataset_size)
    if world_size <= 1:
        return sample_count

    if sample_count >= world_size:
        adjusted = (sample_count // world_size) * world_size
        if adjusted > 0:
            return adjusted

    return min(dataset_size, world_size)


def _build_random_eval_dataloader(
    dataset,
    accelerator,
    batch_size,
    collate_fn,
    pin_memory,
    num_workers,
    requested_size,
    description,
):
    sample_count = _resolve_eval_subset_size(len(dataset), requested_size, accelerator.num_processes)
    sample_seed = _broadcast_int_from_main(accelerator, random.randrange(1, 2**31))
    rng = random.Random(sample_seed)

    if sample_count >= len(dataset):
        sampled_indices = list(range(len(dataset)))
    else:
        sampled_indices = rng.sample(range(len(dataset)), sample_count)

    sampled_subset = Subset(dataset, sampled_indices)
    dataloader = DataLoader(
        sampled_subset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    dataloader = accelerator.prepare_data_loader(dataloader, device_placement=False)

    if accelerator.is_main_process:
        requested_label = "full dataset" if requested_size is None or requested_size <= 0 else str(requested_size)
        print(
            f"{description}: using {sample_count}/{len(dataset)} validation sequences "
            f"(requested {requested_label}, seed={sample_seed}, world_size={accelerator.num_processes})."
        )

    return dataloader


def _capture_reference_parameters(model):
    reference_parameters = {}
    tensor_count = 0
    parameter_count = 0
    bytes_used = 0

    for name, param in model.named_parameters():
        if not param.requires_grad or not torch.is_floating_point(param):
            continue
        reference = param.detach().clone()
        reference_parameters[name] = reference
        tensor_count += 1
        parameter_count += param.numel()
        bytes_used += reference.numel() * reference.element_size()

    return reference_parameters, tensor_count, parameter_count, bytes_used


def _compute_original_weight_l2_penalty(model, reference_parameters):
    total_penalty = None
    total_elements = 0

    for name, param in model.named_parameters():
        if not param.requires_grad or not torch.is_floating_point(param):
            continue
        reference = reference_parameters.get(name)
        if reference is None or reference.shape != param.shape:
            continue

        param_fp32 = param.float()
        reference_fp32 = reference.float()
        penalty = torch.sum((param_fp32 - reference_fp32) ** 2)
        total_penalty = penalty if total_penalty is None else total_penalty + penalty
        total_elements += param.numel()

    if total_penalty is None:
        first_param = next(model.parameters(), None)
        device = first_param.device if first_param is not None else torch.device("cpu")
        return torch.zeros((), device=device, dtype=torch.float32)

    return total_penalty / max(1, total_elements)


def evaluate_model(
    model,
    accelerator,
    dataset,
    batch_size,
    collate_fn,
    pin_memory,
    num_workers=0,
    max_samples=500,
    autoregressive_samples=100,
):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    correct_pitches = 0
    total_pitches = 0

    from anticipation.vocab import CONTROL_OFFSET, REST

    accelerator.wait_for_everyone()
    teacher_dataloader = _build_random_eval_dataloader(
        dataset=dataset,
        accelerator=accelerator,
        batch_size=batch_size,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=num_workers,
        requested_size=max_samples,
        description="Teacher-forced eval",
    )

    with torch.no_grad():
        for batch in tqdm(
            teacher_dataloader,
            desc="Evaluating",
            leave=False,
            disable=not accelerator.is_local_main_process,
        ):
            batch = _move_batch_to_device(batch, accelerator.device)
            outputs = forward_batch(model, batch)
            loss = outputs.loss
            logits = outputs.logits
            input_ids = batch["input_ids"]
            labels = batch["labels"]
            local_batch_size = input_ids.size(0)

            total_loss += loss.item() * local_batch_size
            total_samples += local_batch_size

            for b in range(local_batch_size):
                seq_input = input_ids[b]
                seq_labels = labels[b]
                seq_logits = logits[b]

                i = 0
                while i < len(seq_input) - 2:
                    if (
                        seq_input[i] < CONTROL_OFFSET
                        and seq_input[i + 1] < CONTROL_OFFSET
                        and seq_input[i + 2] < CONTROL_OFFSET
                        and seq_input[i + 2] != REST
                    ):
                        note_pos = i + 2
                        if seq_labels[note_pos] != -100:
                            predicted_token = seq_logits[note_pos - 1].argmax().item()
                            true_token = seq_labels[note_pos].item()
                            if predicted_token == true_token:
                                correct_pitches += 1
                            total_pitches += 1
                    i += 3

    teacher_stats = torch.tensor(
        [total_loss, total_samples, correct_pitches, total_pitches],
        device=accelerator.device,
        dtype=torch.float64,
    )
    teacher_stats = accelerator.reduce(teacher_stats, reduction="sum")
    total_loss = float(teacher_stats[0].item())
    total_samples = int(teacher_stats[1].item())
    correct_pitches = int(teacher_stats[2].item())
    total_pitches = int(teacher_stats[3].item())

    if total_samples == 0:
        raise ValueError(
            "Validation produced zero samples. Check that the validation token file is non-empty and readable."
        )

    avg_loss = total_loss / total_samples
    teacher_forced_accuracy = correct_pitches / total_pitches if total_pitches > 0 else 0.0

    autoregressive_correct = 0
    autoregressive_total = 0

    if autoregressive_samples > 0:
        autoregressive_dataloader = _build_random_eval_dataloader(
            dataset=dataset,
            accelerator=accelerator,
            batch_size=batch_size,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
            num_workers=num_workers,
            requested_size=autoregressive_samples,
            description="Autoregressive eval",
        )

        with torch.no_grad():
            for batch in tqdm(
                autoregressive_dataloader,
                desc="Autoregressive eval",
                leave=False,
                disable=not accelerator.is_local_main_process,
            ):
                input_ids = batch["input_ids"]
                for seq in input_ids:
                    seq = seq.detach().cpu()
                    if len(seq) <= ALTERNATING_START:
                        continue

                    context = seq[:ALTERNATING_START].tolist()
                    pos = ALTERNATING_START
                    while pos + 5 < len(seq):
                        if (
                            seq[pos] < CONTROL_OFFSET
                            and seq[pos + 1] < CONTROL_OFFSET
                            and seq[pos + 2] < CONTROL_OFFSET
                            and seq[pos + 2] != REST
                        ):
                            input_tensor = torch.tensor([context], device=accelerator.device)
                            outputs = model(input_tensor)
                            pred_time = outputs.logits[0, -1, :].argmax().item()
                            context.append(pred_time)

                            input_tensor = torch.tensor([context], device=accelerator.device)
                            outputs = model(input_tensor)
                            pred_dur = outputs.logits[0, -1, :].argmax().item()
                            context.append(pred_dur)

                            input_tensor = torch.tensor([context], device=accelerator.device)
                            outputs = model(input_tensor)
                            pred_pitch = outputs.logits[0, -1, :].argmax().item()
                            context.append(pred_pitch)

                            true_pitch = seq[pos + 2].item()
                            if pred_pitch == true_pitch:
                                autoregressive_correct += 1
                            autoregressive_total += 1

                            pos += 3
                            if pos + 2 < len(seq):
                                context.extend([seq[pos].item(), seq[pos + 1].item(), seq[pos + 2].item()])
                                pos += 3
                        else:
                            context.append(seq[pos].item())
                            pos += 1

    autoregressive_stats = torch.tensor(
        [autoregressive_correct, autoregressive_total],
        device=accelerator.device,
        dtype=torch.float64,
    )
    autoregressive_stats = accelerator.reduce(autoregressive_stats, reduction="sum")
    autoregressive_correct = int(autoregressive_stats[0].item())
    autoregressive_total = int(autoregressive_stats[1].item())
    autoregressive_accuracy = autoregressive_correct / autoregressive_total if autoregressive_total > 0 else 0.0

    accelerator.wait_for_everyone()
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
    parser.add_argument('--output_dir', type=Path, default=Path('./regularized'))
    parser.add_argument('--batch_size', type=int, default=8) 
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4) 
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--max_steps', type=int, default=40000)
    parser.add_argument('--save_steps', type=int, default=2500)
    parser.add_argument('--eval_steps', type=int, default=500)
    parser.add_argument(
        '--eval_max_samples',
        type=int,
        default=500,
        help='Random validation sequences for teacher-forced eval. <= 0 uses the full validation set.',
    )
    parser.add_argument(
        '--eval_autoregressive_samples',
        type=int,
        default=100,
        help='Random validation sequences for autoregressive eval. <= 0 disables autoregressive eval.',
    )
    parser.add_argument(
        '--eval_num_workers',
        type=int,
        default=0,
        help='Dataloader workers used for sampled validation subsets.',
    )
    parser.add_argument('--warmup_steps', type=int, default=0)  # No warmup
    parser.add_argument('--force_cpu', action='store_true', help='Force CPU usage even if GPU is available')
    parser.add_argument('--reduce_memory', action='store_true', help='Use memory-saving techniques')
    parser.add_argument(
        '--onset_jitter_std',
        type=float,
        default=0.05,
        help='Std of N(1, std^2) multiplier applied to each inter-onset interval of control tokens (training only)',
    )
    parser.add_argument(
        '--dur_jitter_range',
        type=float,
        default=0.05,
        help='Half-range of U(1-r, 1+r) duration rescaling per control note (training only)',
    )
    parser.add_argument(
        '--mask_prob',
        type=float,
        default=0.75,
        help='Fraction of score triplets whose token embeddings are zeroed in the input context (training only)',
    )
    parser.add_argument(
        '--loss_mask_performance_tokens',
        action='store_true',
        help='Exclude performance/control triplets from the loss by setting their labels to -100',
    )
    parser.add_argument(
        '--transpose_range_semitones',
        type=int,
        default=12,
        help='Max transposition shift in semitones, uniform in [-range, +range] (training only)',
    )
    parser.add_argument(
        '--tempo_scale_range',
        type=float,
        default=0.2,
        help='Tempo scale half-range sampled uniformly from [1-range, 1+range] and applied only to performance/control timing (training only)',
    )
    parser.add_argument(
        '--original_weight_l2',
        type=float,
        default=1e5,
        help='Coefficient for L2 anchoring to the model weights immediately after load/resize. Set to 0 to disable.',
    )
    args = parser.parse_args()

    if args.original_weight_l2 < 0:
        raise ValueError("--original_weight_l2 must be non-negative.")
    if args.eval_num_workers < 0:
        raise ValueError("--eval_num_workers must be non-negative.")
    
    # Override device if requested
    global device
    if args.force_cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage as requested")
    
    print(f"Per-rank effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
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
        print(
            "Distributed setup: "
            f"type={accelerator.distributed_type}, "
            f"world_size={accelerator.num_processes}, "
            f"rank={accelerator.process_index}, "
            f"local_rank={accelerator.local_process_index}, "
            f"device={accelerator.device}"
        )
        if accelerator.is_main_process:
            print(
                "Global effective batch size: "
                f"{args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}"
            )
            if torch.cuda.is_available() and torch.cuda.device_count() > 1 and accelerator.num_processes == 1:
                print(
                    "WARNING: Multiple CUDA devices are visible, but training is running with world_size=1. "
                    "Launch with accelerate or torchrun to use multiple GPUs."
                )
        
        # Create output directory once and synchronize before any rank writes into it.
        if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
        accelerator.wait_for_everyone()
        
        # Monitor initial GPU memory
        print("Initial GPU memory stats:")
        print_gpu_memory_stats()
        
        # Load training dataset
        def collate_fn(batch):
            input_ids = torch.stack([item["input_ids"] for item in batch])
            attention_mask = torch.stack([item["attention_mask"] for item in batch])
            labels = torch.stack([item["labels"] for item in batch])
            score_token_mask = torch.stack([item["score_token_mask"] for item in batch])
            score_mask = torch.stack([item["score_mask"] for item in batch])
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "score_token_mask": score_token_mask,
                "score_mask": score_mask,
            }

        print(f"Loading training dataset from {args.data_file}...")
        train_dataset = TokenizedDataset(
            args.data_file,
            onset_jitter_std=args.onset_jitter_std,
            dur_jitter_range=args.dur_jitter_range,
            mask_prob=args.mask_prob,
            transpose_range_semitones=args.transpose_range_semitones,
            tempo_scale_range=args.tempo_scale_range,
            loss_mask_performance_tokens=args.loss_mask_performance_tokens,
            is_training=True,
        )
        if len(train_dataset) == 0:
            raise ValueError(
                "Training dataset is empty. Check the tokenized training file and rerun tokenization if needed."
            )
            
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
            loss_mask_performance_tokens=args.loss_mask_performance_tokens,
            is_training=False
        )
        if len(val_dataset) == 0:
            raise ValueError(
                f"Validation dataset is empty: {args.val_file}. Check the tokenized validation file."
            )
        
        val_loader_kwargs = {
            "batch_size": args.val_batch_size,
            "collate_fn": collate_fn,
            "pin_memory": torch.cuda.is_available() and not args.force_cpu,
            "num_workers": args.eval_num_workers,
        }
        
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
        print(f"After accelerator preparation, model device: {next(model.parameters()).device}")

        base_model = accelerator.unwrap_model(model)
        original_weight_references = {}
        if args.original_weight_l2 > 0:
            (
                original_weight_references,
                anchored_tensor_count,
                anchored_parameter_count,
                anchored_bytes,
            ) = _capture_reference_parameters(base_model)
            if accelerator.is_main_process:
                anchored_megabytes = anchored_bytes / (1024 ** 2)
                print(
                    "Original-weight L2 regularization enabled: "
                    f"lambda={args.original_weight_l2}, "
                    f"{anchored_tensor_count} tensors, "
                    f"{anchored_parameter_count:,} parameters, "
                    f"~{anchored_megabytes:.1f} MiB snapshot."
                )
        elif accelerator.is_main_process:
            print("Original-weight L2 regularization disabled.")
        
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
        
        if torch.cuda.is_available() and accelerator.device.type == "cuda":
            print(f"Clearing CUDA cache before training on {accelerator.device}")
            torch.cuda.empty_cache()
            device_index = accelerator.device.index
            if device_index is not None:
                torch.cuda.set_device(device_index)
        
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
        
        # Keep progress/logging on the main process so multi-GPU runs don't spam duplicate output.
        progress_bar = tqdm(total=args.max_steps, desc="Training", disable=not accelerator.is_main_process)
        training_failed = False
        
        try:
            while completed_steps < args.max_steps:
                for batch in train_dataloader:
                    try:
                        with accelerator.accumulate(model):
                            # Forward pass with gradient scaling
                            outputs = forward_batch(model, batch)
                            loss = outputs.loss
                            l2_penalty = None
                            if original_weight_references:
                                l2_penalty = _compute_original_weight_l2_penalty(
                                    base_model,
                                    original_weight_references,
                                )
                                loss = loss + args.original_weight_l2 * l2_penalty
                            
                            # Keep NaN/Inf recovery in lockstep across ranks to avoid DDP hangs.
                            local_invalid_loss = bool(torch.isnan(loss).any() or torch.isinf(loss).any())
                            invalid_loss_processes = _count_flagged_processes(accelerator, local_invalid_loss)
                            if invalid_loss_processes > 0:
                                if accelerator.is_main_process:
                                    print(
                                        f"WARNING: NaN or Inf loss detected on {invalid_loss_processes}/"
                                        f"{accelerator.num_processes} rank(s); skipping this synchronized step."
                                    )
                                optimizer.zero_grad()
                                continue
                                
                            # Backward pass
                            accelerator.backward(loss)
                            
                            # Only update optimizer and scheduler when gradients are synchronized
                            if accelerator.sync_gradients:
                                invalid_grad_name = _find_invalid_gradient_parameter(model)
                                invalid_grad_processes = _count_flagged_processes(
                                    accelerator,
                                    invalid_grad_name is not None,
                                )
                                if invalid_grad_processes > 0:
                                    if accelerator.is_main_process:
                                        detail = f" Example parameter: {invalid_grad_name}." if invalid_grad_name else ""
                                        print(
                                            f"WARNING: NaN or Inf gradients detected on {invalid_grad_processes}/"
                                            f"{accelerator.num_processes} rank(s); skipping optimizer step.{detail}"
                                        )
                                    optimizer.zero_grad()
                                    continue

                                # Gradient clipping - industry standard value
                                accelerator.clip_grad_norm_(model.parameters(), max_norm=2.0)
                                
                                # Only update optimizer and scheduler here
                                optimizer.step()
                                scheduler.step()
                                optimizer.zero_grad()
                                reduced_loss = accelerator.reduce(
                                    loss.detach().to(device=accelerator.device, dtype=torch.float64),
                                    reduction="mean",
                                ).item()
                                reduced_l2_penalty = None
                                reduced_anchor_term = None
                                if l2_penalty is not None:
                                    reduced_l2_penalty = accelerator.reduce(
                                        l2_penalty.detach().to(device=accelerator.device, dtype=torch.float64),
                                        reduction="mean",
                                    ).item()
                                    reduced_anchor_term = args.original_weight_l2 * reduced_l2_penalty
                                
                                # Only update step counters when we actually update weights
                                completed_steps += 1
                                progress_bar.update(1)
                                
                                # Log progress
                                if completed_steps % 10 == 0 and accelerator.is_main_process:
                                    # Store the training loss every 10 steps
                                    train_losses.append(reduced_loss)
                                    
                                    # Print more precise learning rate
                                    l2_detail = ""
                                    if reduced_l2_penalty is not None:
                                        l2_detail = (
                                            f", AnchorL2: {reduced_l2_penalty:.6e}, "
                                            f"AnchorTerm: {reduced_anchor_term:.6e}"
                                        )
                                    print(
                                        f"Step: {completed_steps}/{args.max_steps}, Loss: {reduced_loss:.4f}, "
                                        f"LR: {scheduler.get_last_lr()[0]:.8e}{l2_detail}"
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
                                    if accelerator.is_main_process:
                                        print(f"\nRunning validation at step {completed_steps}...")
                                    val_loss, val_acc, val_auto_acc = evaluate_model(
                                        model,
                                        accelerator,
                                        val_dataset,
                                        **val_loader_kwargs,
                                        max_samples=args.eval_max_samples,
                                        autoregressive_samples=args.eval_autoregressive_samples,
                                    )
                                    if accelerator.is_main_process:
                                        validation_steps.append(completed_steps)  # Store step number
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
                                    if accelerator.is_main_process:
                                        print(f"\nRunning validation at checkpoint step {completed_steps}...")
                                    val_loss, val_acc, val_auto_acc = evaluate_model(
                                        model,
                                        accelerator,
                                        val_dataset,
                                        **val_loader_kwargs,
                                        max_samples=args.eval_max_samples,
                                        autoregressive_samples=args.eval_autoregressive_samples,
                                    )
                                    if accelerator.is_main_process:
                                        validation_steps.append(completed_steps)
                                        val_losses.append(val_loss)
                                        val_accuracies.append(val_acc * 100)
                                        val_autoregressive_accuracies.append(val_auto_acc * 100)
                                        print(f"Validation Loss: {val_loss:.4f}, Teacher-Forced Accuracy: {val_acc*100:.2f}%, Autoregressive Accuracy: {val_auto_acc*100:.2f}%")
                                    
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
                                    accelerator.wait_for_everyone()
                                    
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
                            if accelerator.num_processes > 1:
                                print("Distributed run detected; aborting instead of skipping locally to avoid rank desync.")
                                raise
                            print("Trying to recover by skipping this batch...")
                            optimizer.zero_grad()
                            continue
                        else:
                            print(f"Runtime error: {str(e)}")
                            print(traceback.format_exc())
                            raise
            
        except Exception as e:
            training_failed = True
            print(f"Error during training: {e}")
            print(traceback.format_exc())
            raise
        finally:
            # Make sure we always close the progress bar
            progress_bar.close()
            
            # Only run final validation/save after a clean training loop exit.
            if training_failed:
                if accelerator.is_main_process:
                    print("Skipping final validation/save because training exited with an error.")
            else:
                try:
                    # Final validation run
                    if accelerator.is_main_process:
                        print("\nRunning final validation...")
                    final_val_loss, final_val_acc, final_auto_acc = evaluate_model(
                        model,
                        accelerator,
                        val_dataset,
                        **val_loader_kwargs,
                        max_samples=args.eval_max_samples,
                        autoregressive_samples=args.eval_autoregressive_samples,
                    )
                    if accelerator.is_main_process:
                        validation_steps.append(completed_steps)
                        val_losses.append(final_val_loss)
                        val_accuracies.append(final_val_acc * 100)
                        val_autoregressive_accuracies.append(final_auto_acc * 100)
                        print(f"Final validation Loss: {final_val_loss:.4f}, Teacher-Forced Accuracy: {final_val_acc*100:.2f}%, Autoregressive Accuracy: {final_auto_acc*100:.2f}%")
                    
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
                    accelerator.wait_for_everyone()
                    
                except Exception as save_error:
                    print(f"Error saving final model or generating plot: {save_error}")
            
    except Exception as setup_error:
        print(f"Error in setup: {setup_error}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main()
