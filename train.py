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
import warnings

import torch.nn.functional as F

try:
    import wandb
except ImportError:  # pragma: no cover - wandb is an optional dependency
    wandb = None

from anticipation.config import CONTEXT_SIZE, EVENT_SIZE, MAX_TIME, MAX_DUR, MAX_PITCH
from anticipation.packed_sequence import (
    ALTERNATING_START,
    is_control_triplet_tokens,
    is_real_score_triplet,
    iter_score_slot_positions,
)
from anticipation.vocab import (
    ADUR_OFFSET,
    ANOTE_OFFSET,
    ATIME_OFFSET,
    CONTROL_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    REST,
    TIME_OFFSET,
    VOCAB_SIZE,
)
from inference import batched_autoregressive_generate_score

# Each triplet is (onset/time, duration, pitch/note). A token's role within its
# triplet is therefore (token_index % 3). These names are reused for per-type
# loss and accuracy reporting.
TOKEN_TYPE_NAMES = ("onset", "duration", "pitch")


def iter_sequence_triplets(tokens):
    """Scan a packed sequence, yielding (pos, tok0, tok1, tok2, is_control).

    Score/prefix-placeholder triplets (all tokens < CONTROL_OFFSET) yield
    is_control=False; control triplets yield is_control=True. Anything else
    (e.g. a SEPARATOR) is skipped one token at a time so the scan re-syncs.
    """
    i = 0
    while i < len(tokens) - 2:
        tok0 = int(tokens[i])
        tok1 = int(tokens[i + 1])
        tok2 = int(tokens[i + 2])

        if tok0 < CONTROL_OFFSET and tok1 < CONTROL_OFFSET and tok2 < CONTROL_OFFSET:
            yield i, tok0, tok1, tok2, False
            i += 3
        elif is_control_triplet_tokens(tok0, tok1, tok2):
            yield i, tok0, tok1, tok2, True
            i += 3
        else:
            i += 1

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

def _parse_sm_arch(arch_name):
    if not arch_name.startswith("sm_"):
        return None
    digits = "".join(ch for ch in arch_name if ch.isdigit())
    if len(digits) < 2:
        return None
    return int(digits[:-1]), int(digits[-1])

def _get_supported_sm_arches():
    try:
        arch_list = torch.cuda.get_arch_list()
    except Exception:
        return set()

    supported = set()
    for arch_name in arch_list:
        parsed = _parse_sm_arch(arch_name)
        if parsed is not None:
            supported.add(parsed)
    return supported

def _format_sm_arch(capability):
    major, minor = capability
    return f"sm_{major}{minor}"

def _is_cubin_compatible_with_device(selected_capability, compiled_capability):
    selected_major, selected_minor = selected_capability
    compiled_major, compiled_minor = compiled_capability
    return compiled_major == selected_major and compiled_minor <= selected_minor

def validate_selected_cuda_device_or_raise(force_cpu=False):
    if force_cpu or not torch.cuda.is_available():
        return

    supported_arches = _get_supported_sm_arches()
    if not supported_arches:
        return

    try:
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    except ValueError:
        local_rank = 0

    device_index = local_rank if 0 <= local_rank < torch.cuda.device_count() else 0
    props = torch.cuda.get_device_properties(device_index)
    selected_capability = (props.major, props.minor)
    if any(
        _is_cubin_compatible_with_device(selected_capability, compiled_capability)
        for compiled_capability in supported_arches
    ):
        return

    supported_text = " ".join(
        _format_sm_arch(capability) for capability in sorted(supported_arches)
    )
    raise RuntimeError(
        "Selected CUDA device is not supported by the installed PyTorch build. "
        f"Device {device_index}: {props.name} ({_format_sm_arch(selected_capability)}). "
        f"Supported CUDA capabilities in this PyTorch build: {supported_text}. "
        "Install a newer PyTorch CUDA build that includes this GPU architecture "
        "(PyTorch currently points B200/sm_100 users to CUDA 12.8 or 12.9 builds), "
        "or rerun with --force_cpu."
    )

def report_runtime_device(force_cpu=False):
    global device

    if force_cpu:
        device = torch.device("cpu")
        print("Forcing CPU usage as requested")
        print("[warn] CUDA probing skipped because --force_cpu was set.")
    elif torch.cuda.is_available():
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

device = torch.device("cpu")

PACKED_SEQUENCE_LENGTH = CONTEXT_SIZE - 4

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
                f"onset_jitter_std={self.onset_jitter_std} (controls in input only), "
                f"dur_jitter_range={self.dur_jitter_range} (controls in input only), "
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

    def _sample_control_timing_plan(self, tokens):
        """Pre-sample jittered control onset times and duration factors.

        Sampling the randomness once (instead of inside ``_augment_sequence``)
        lets the input and label passes share identical augmentation decisions.
        """
        ctrl_positions = []
        raw_times = []
        for pos, tok0, _, _, is_control in iter_sequence_triplets(tokens):
            if is_control:
                ctrl_positions.append(pos)
                raw_times.append(tok0 - ATIME_OFFSET)

        new_ctrl_times = None
        if self.onset_jitter_std > 0 and len(raw_times) >= 2:
            # Jitter each inter-onset interval multiplicatively, accumulating so
            # local rubato is perturbed without drifting the global clock sign.
            new_time = float(raw_times[0])
            jittered = [new_time]
            for k in range(1, len(raw_times)):
                ioi = raw_times[k] - raw_times[k - 1]
                scale = 1.0 + torch.randn(1).item() * self.onset_jitter_std
                new_time = new_time + ioi * scale
                jittered.append(new_time)
            new_ctrl_times = jittered

        dur_factors = None
        if self.dur_jitter_range > 0 and ctrl_positions:
            dur_factors = [
                1.0 + (torch.rand(1).item() * 2.0 - 1.0) * self.dur_jitter_range
                for _ in ctrl_positions
            ]

        return {
            "num_controls": len(ctrl_positions),
            "new_ctrl_times": new_ctrl_times,
            "dur_factors": dur_factors,
        }

    def _augment_sequence(
        self,
        tokens,
        transpose_shift=0,
        tempo_factor=1.0,
        apply_timing_augmentation=True,
        apply_tempo_scaling_to_controls=True,
        control_timing_plan=None,
    ):
        """Apply training augmentations, returning a new token tensor.

        Transposition applies to both score and control pitches (labels and
        inputs must transpose together). Timing jitter and tempo scaling apply
        only to control (performance) triplets: score timing is the prediction
        target and is never perturbed. ``control_timing_plan`` carries the
        pre-sampled jitter so the input pass uses it while the label pass
        (``apply_timing_augmentation=False``) leaves control times clean.
        """
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

        score_positions = []
        ctrl_positions = []
        for pos, tok0, tok1, tok2, is_control in iter_sequence_triplets(augmented):
            if is_control:
                ctrl_positions.append((pos, tok0, tok1, tok2))
            else:
                score_positions.append((pos, tok0, tok1, tok2))

        # Score / prefix-placeholder triplets: transposition only.
        if transpose_shift != 0:
            for pos_i, _, _, tok2 in score_positions:
                if tok2 != REST:
                    augmented[pos_i + 2] = _transpose_note(tok2, NOTE_OFFSET)

        new_ctrl_times = None
        if apply_timing_augmentation and control_timing_plan is not None:
            planned_times = control_timing_plan.get("new_ctrl_times")
            if planned_times is not None:
                new_ctrl_times = [
                    max(0, min(MAX_TIME - 1, int(round(t))))
                    for t in planned_times
                ]
        dur_factors = (
            control_timing_plan.get("dur_factors")
            if apply_timing_augmentation and control_timing_plan is not None
            else None
        )

        for ctrl_index, (pos_i, tok0, tok1, tok2) in enumerate(ctrl_positions):
            if new_ctrl_times is not None:
                tok0 = ATIME_OFFSET + new_ctrl_times[ctrl_index]

            if dur_factors is not None:
                base_dur = tok1 - ADUR_OFFSET
                tok1 = ADUR_OFFSET + max(
                    0, min(MAX_DUR - 1, int(round(base_dur * dur_factors[ctrl_index])))
                )

            if apply_tempo_scaling_to_controls and tempo_factor != 1.0:
                tok0 = _scale_time(tok0, ATIME_OFFSET)
                tok1 = _scale_dur(tok1, ADUR_OFFSET)

            if transpose_shift != 0:
                tok2 = _transpose_note(tok2, ANOTE_OFFSET)

            augmented[pos_i] = tok0
            augmented[pos_i + 1] = tok1
            augmented[pos_i + 2] = tok2

        return augmented

    def _build_score_token_mask(self, tokens):
        score_token_mask = torch.zeros_like(tokens, dtype=torch.bool)
        for pos in iter_score_slot_positions(len(tokens), ALTERNATING_START):
            if is_real_score_triplet(tokens, pos, ALTERNATING_START):
                score_token_mask[pos : pos + 3] = True

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
        loss_mask = torch.zeros_like(tokens, dtype=torch.bool)
        if not self.loss_mask_performance_tokens:
            return loss_mask

        for pos, _, _, _, is_control in iter_sequence_triplets(tokens):
            if is_control:
                loss_mask[pos:pos + 3] = True

        return loss_mask

    def _clamp_tokens_to_vocab(self, tokens, tensor_name, sample_idx):
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
            # Onset/duration jitter applies only to control (performance) triplets in the
            # model input, not score triplets. Labels never use timing jitter (see below).
            control_timing_plan = self._sample_control_timing_plan(tokens)
            augmented_tokens = self._augment_sequence(
                tokens,
                transpose_shift=transpose_shift,
                tempo_factor=tempo_factor,
                apply_timing_augmentation=True,
                apply_tempo_scaling_to_controls=True,
                control_timing_plan=control_timing_plan,
            )
            labels = self._augment_sequence(
                tokens,
                transpose_shift=transpose_shift,
                tempo_factor=tempo_factor,
                apply_timing_augmentation=False,
                apply_tempo_scaling_to_controls=not self.loss_mask_performance_tokens,
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


def num_score_slots(sequence_length, score_start_idx=ALTERNATING_START):
    """Number of score-triplet slots in a packed sequence (heatmap width)."""
    return len(list(iter_score_slot_positions(sequence_length, score_start_idx)))


def _accumulate_per_type_teacher_forced(seq_logits, seq_labels, ce_sum, ce_count):
    """Accumulate per-token-type cross-entropy for one teacher-forced sequence.

    Causal LMs predict token ``j`` from the logits at position ``j - 1``; a
    token's type is ``j % 3`` (0=onset, 1=duration, 2=pitch). ``-100`` labels are
    ignored. ``ce_sum``/``ce_count`` are length-3 accumulators indexed by type.
    """
    shift_logits = seq_logits[:-1, :].float()
    shift_labels = seq_labels[1:]
    losses = F.cross_entropy(
        shift_logits, shift_labels, reduction="none", ignore_index=-100
    )
    predicted_positions = torch.arange(1, seq_labels.size(0), device=losses.device)
    roles = predicted_positions % EVENT_SIZE
    valid = shift_labels != -100
    for token_type in range(EVENT_SIZE):
        selected = valid & (roles == token_type)
        ce_sum[token_type] += float(losses[selected].sum().item())
        ce_count[token_type] += int(selected.sum().item())


def evaluate_model(
    model,
    accelerator,
    dataset,
    batch_size,
    collate_fn,
    pin_memory,
    num_workers=0,
    max_samples=500,
    autoregressive_samples=20,
    disable_autoregressive_pitch_eval=False,
    heatmap_slots=None,
):
    """Run teacher-forced and autoregressive validation.

    Returns a metrics dict with overall and per-token-type losses, teacher-forced
    and autoregressive accuracies, and per-slot autoregressive error vectors used
    to build the training-step x token-index heatmaps.
    """
    model.eval()
    if heatmap_slots is None:
        heatmap_slots = num_score_slots(getattr(dataset, "sequence_length", 0) or 0)

    total_loss = 0.0
    total_samples = 0
    correct_pitches = 0
    total_pitches = 0
    per_type_ce_sum = [0.0] * EVENT_SIZE
    per_type_ce_count = [0] * EVENT_SIZE

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

                _accumulate_per_type_teacher_forced(
                    seq_logits, seq_labels, per_type_ce_sum, per_type_ce_count
                )

                for pos in iter_score_slot_positions(len(seq_input), ALTERNATING_START):
                    if not is_real_score_triplet(seq_input, pos, ALTERNATING_START):
                        continue
                    note_pos = pos + 2
                    if seq_labels[note_pos] != -100:
                        predicted_token = seq_logits[note_pos - 1].argmax().item()
                        true_token = seq_labels[note_pos].item()
                        if predicted_token == true_token:
                            correct_pitches += 1
                        total_pitches += 1

    teacher_stats = torch.tensor(
        [total_loss, total_samples, correct_pitches, total_pitches]
        + per_type_ce_sum
        + [float(c) for c in per_type_ce_count],
        device=accelerator.device,
        dtype=torch.float64,
    )
    teacher_stats = accelerator.reduce(teacher_stats, reduction="sum")
    total_loss = float(teacher_stats[0].item())
    total_samples = int(teacher_stats[1].item())
    correct_pitches = int(teacher_stats[2].item())
    total_pitches = int(teacher_stats[3].item())
    per_type_ce_sum = [float(teacher_stats[4 + i].item()) for i in range(EVENT_SIZE)]
    per_type_ce_count = [int(teacher_stats[4 + EVENT_SIZE + i].item()) for i in range(EVENT_SIZE)]

    if total_samples == 0:
        raise ValueError(
            "Validation produced zero samples. Check that the validation token file is non-empty and readable."
        )

    avg_loss = total_loss / total_samples
    teacher_forced_accuracy = correct_pitches / total_pitches if total_pitches > 0 else 0.0
    per_type_loss = {
        TOKEN_TYPE_NAMES[i]: (per_type_ce_sum[i] / per_type_ce_count[i] if per_type_ce_count[i] > 0 else float("nan"))
        for i in range(EVENT_SIZE)
    }

    results = {
        "loss": avg_loss,
        "teacher_forced_pitch_accuracy": teacher_forced_accuracy,
        "loss_by_type": per_type_loss,
    }

    # Autoregressive evaluation: greedy-decode score triplets with teacher-forced
    # ground-truth controls, then score onset/duration/pitch against ground truth.
    # Per-slot vectors (indexed by score-slot ordinal = position in the sequence)
    # feed the heatmaps; aggregate counts feed the accuracy line graphs.
    pitch_err = torch.zeros(heatmap_slots, dtype=torch.float64, device=accelerator.device)
    onset_err = torch.zeros(heatmap_slots, dtype=torch.float64, device=accelerator.device)
    dur_err = torch.zeros(heatmap_slots, dtype=torch.float64, device=accelerator.device)
    onset_abs = torch.zeros(heatmap_slots, dtype=torch.float64, device=accelerator.device)
    dur_abs = torch.zeros(heatmap_slots, dtype=torch.float64, device=accelerator.device)
    slot_total = torch.zeros(heatmap_slots, dtype=torch.float64, device=accelerator.device)

    if disable_autoregressive_pitch_eval or autoregressive_samples <= 0:
        results["autoregressive"] = None
        accelerator.wait_for_everyone()
        return results

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
            if input_ids.shape[0] == 0 or input_ids.shape[1] <= ALTERNATING_START:
                continue
            # Batched decode: every row shares the identical packed-sequence slot
            # layout, so one KV-cached batch forward pass replaces a per-sequence
            # Python loop -- a single-sequence forward barely uses the GPU's
            # compute capacity, so this is a large speedup over the old approach.
            pred_ctx = batched_autoregressive_generate_score(
                model,
                input_ids,
                ALTERNATING_START,
                str(accelerator.device),
                constrain_score_tokens=True,
                ground_truth_score_tokens_to_feed=0,
            )
            input_ids_dev = input_ids.to(accelerator.device)
            for slot_idx, pos in enumerate(
                iter_score_slot_positions(input_ids.shape[1], ALTERNATING_START)
            ):
                if slot_idx >= heatmap_slots or pos + 2 >= pred_ctx.shape[1]:
                    break
                real_mask = input_ids_dev[:, pos + 2] != REST
                if not torch.any(real_mask):
                    continue
                slot_total[slot_idx] += int(real_mask.sum().item())
                onset_err[slot_idx] += int(
                    ((pred_ctx[:, pos] != input_ids_dev[:, pos]) & real_mask).sum().item()
                )
                dur_err[slot_idx] += int(
                    ((pred_ctx[:, pos + 1] != input_ids_dev[:, pos + 1]) & real_mask).sum().item()
                )
                pitch_err[slot_idx] += int(
                    ((pred_ctx[:, pos + 2] != input_ids_dev[:, pos + 2]) & real_mask).sum().item()
                )
                onset_abs[slot_idx] += float(
                    (torch.abs(pred_ctx[:, pos] - input_ids_dev[:, pos]) * real_mask).sum().item()
                )
                dur_abs[slot_idx] += float(
                    (torch.abs(pred_ctx[:, pos + 1] - input_ids_dev[:, pos + 1]) * real_mask).sum().item()
                )

    stacked = torch.stack([pitch_err, onset_err, dur_err, onset_abs, dur_abs, slot_total])
    stacked = accelerator.reduce(stacked, reduction="sum")
    pitch_err, onset_err, dur_err, onset_abs, dur_abs, slot_total = stacked.cpu().numpy()

    total_slots = float(slot_total.sum())

    def _accuracy(error_vec):
        return 1.0 - (float(error_vec.sum()) / total_slots) if total_slots > 0 else 0.0

    with np.errstate(invalid="ignore", divide="ignore"):
        results["autoregressive"] = {
            "pitch_accuracy": _accuracy(pitch_err),
            "onset_accuracy": _accuracy(onset_err),
            "duration_accuracy": _accuracy(dur_err),
            "total_notes": int(total_slots),
            # Per-slot heatmap rows (NaN where no real note occupied that slot).
            "slot_pitch_error_freq": np.where(slot_total > 0, pitch_err / slot_total, np.nan),
            "slot_onset_mae": np.where(slot_total > 0, onset_abs / slot_total, np.nan),
            "slot_duration_mae": np.where(slot_total > 0, dur_abs / slot_total, np.nan),
        }

    accelerator.wait_for_everyone()
    return results


def build_error_heatmap_chart(
    history_rows,
    validation_steps,
    *,
    output_dir=None,
    metric_key="metric",
    title=None,
    value_label="value",
    cmap="magma",
):
    """Render a (training step x score-slot) error heatmap with matplotlib.

    ``history_rows`` is a list of per-validation 1-D arrays (one value per score
    slot, NaN where no real note occupied that slot). We stack them into a matrix
    (rows = validations, columns = score-slot index) and draw it with
    ``imshow``: x = score-slot ordinal (the k-th predicted note), y = training
    step, color = the error metric. NaN cells are left blank (grey).

    The native W&B ``wandb/heatmap/v0`` Vega preset that this used to return did
    not render reliably, so we now render the figure ourselves. The PNG is saved
    locally under ``<output_dir>/heatmaps/<metric_key>.png`` and, when wandb is
    available, the same figure is returned wrapped in ``wandb.Image`` so it also
    shows up on the dashboard.
    """
    if not history_rows:
        return None

    import matplotlib
    matplotlib.use("Agg")  # headless / cluster-safe backend
    import matplotlib.pyplot as plt

    matrix = np.vstack(history_rows).astype(float)  # (n_validations, n_slots)
    masked = np.ma.masked_invalid(matrix)

    fig_height = max(3.0, 0.35 * matrix.shape[0] + 1.5)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgrey")
    im = ax.imshow(
        masked,
        aspect="auto",
        origin="lower",
        cmap=cmap_obj,
        interpolation="nearest",
    )
    ax.set_xlabel("Score-slot index (k-th predicted note)")
    ax.set_ylabel("Training step")
    if title:
        ax.set_title(title)

    # Label y ticks with the actual validation steps, thinning if there are many.
    n_rows = matrix.shape[0]
    max_yticks = 25
    stride = max(1, int(np.ceil(n_rows / max_yticks)))
    tick_idx = list(range(0, n_rows, stride))
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([str(int(validation_steps[i])) for i in tick_idx])

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value_label)
    fig.tight_layout()

    if output_dir is not None:
        heatmap_dir = Path(output_dir) / "heatmaps"
        heatmap_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(heatmap_dir / f"{metric_key}.png", dpi=150)

    image = wandb.Image(fig) if wandb is not None else None
    plt.close(fig)
    return image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_file', type=Path, default=Path('./data/train_normalized.txt'))
    parser.add_argument('--val_file', type=Path, default=Path('./data/test_normalized.txt'))
    parser.add_argument('--model_name', type=str, default='stanford-crfm/music-medium-800k')
    parser.add_argument('--output_dir', type=Path, default=Path('./no_shift'))
    parser.add_argument('--batch_size', type=int, default=8) 
    parser.add_argument('--val_batch_size', type=int, default=8)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4) 
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--max_steps', type=int, default=40000)
    parser.add_argument('--save_steps', type=int, default=2500)
    parser.add_argument('--eval_steps', type=int, default=1000)
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
        '--disable-autoregressive-pitch-eval',
        action='store_true',
        default=False,
        help='Skip autoregressive pitch decoding during validation (teacher-forced eval still runs).',
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
        help='Std of N(1, std^2) multiplier applied to each inter-onset interval of control triplets and score-side context triplets; labels are unchanged (training only)',
    )
    parser.add_argument(
        '--dur_jitter_range',
        type=float,
        default=0.05,
        help='Half-range of U(1-r, 1+r) duration rescaling per control triplet and score-side context triplet; labels are unchanged (training only)',
    )
    parser.add_argument(
        '--mask_prob',
        type=float,
        default=0.00,
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
    parser.add_argument(
        '--wandb_project',
        type=str,
        default='anticipation-asap',
        help='Weights & Biases project name for metric logging.',
    )
    parser.add_argument(
        '--wandb_run_name',
        type=str,
        default=None,
        help='Optional Weights & Biases run name (defaults to the output dir name).',
    )
    parser.add_argument(
        '--wandb_mode',
        type=str,
        default='online',
        choices=['online', 'offline', 'disabled'],
        help="Weights & Biases mode. Use 'disabled' to turn off logging entirely.",
    )
    args = parser.parse_args()

    if args.original_weight_l2 < 0:
        raise ValueError("--original_weight_l2 must be non-negative.")
    if args.eval_num_workers < 0:
        raise ValueError("--eval_num_workers must be non-negative.")
    if wandb is None and args.wandb_mode != 'disabled':
        print("WARNING: wandb is not installed; falling back to --wandb_mode disabled.")
        args.wandb_mode = 'disabled'

    validate_selected_cuda_device_or_raise(force_cpu=args.force_cpu)
    report_runtime_device(force_cpu=args.force_cpu)
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

        # Initialize Weights & Biases on the main process only. All scalar metrics,
        # per-token-type losses, autoregressive accuracies, and error heatmaps are
        # streamed here (this replaces the old matplotlib .png / .npz artifacts).
        use_wandb = accelerator.is_main_process and args.wandb_mode != 'disabled' and wandb is not None
        if use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or args.output_dir.name,
                mode=args.wandb_mode,
                dir=str(args.output_dir),
                config=vars(args),
            )

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
        
        # Heatmap history (main process only): one row per validation, each row a
        # per-slot error vector. These accumulate over training so each validation
        # re-renders the full (step x token-index) heatmap for wandb.
        heatmap_slots = num_score_slots(val_dataset.sequence_length)
        validation_steps = []
        pitch_error_history = []
        onset_mae_history = []
        duration_mae_history = []

        # Keep progress/logging on the main process so multi-GPU runs don't spam duplicate output.
        progress_bar = tqdm(total=args.max_steps, desc="Training", disable=not accelerator.is_main_process)
        training_failed = False

        def run_validation(validation_label):
            validation_step = int(completed_steps)
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                print(f"\nRunning validation at {validation_label}...")

            results = evaluate_model(
                model,
                accelerator,
                val_dataset,
                **val_loader_kwargs,
                max_samples=args.eval_max_samples,
                autoregressive_samples=args.eval_autoregressive_samples,
                disable_autoregressive_pitch_eval=args.disable_autoregressive_pitch_eval,
                heatmap_slots=heatmap_slots,
            )

            if accelerator.is_main_process:
                val_loss = results["loss"]
                val_acc = results["teacher_forced_pitch_accuracy"]
                loss_by_type = results["loss_by_type"]
                autoregressive = results["autoregressive"]

                log_data = {
                    "val/loss": val_loss,
                    "val/teacher_forced_pitch_accuracy": val_acc * 100,
                    "val/loss_onset": loss_by_type["onset"],
                    "val/loss_duration": loss_by_type["duration"],
                    "val/loss_pitch": loss_by_type["pitch"],
                }

                if autoregressive is not None:
                    ar_pitch = autoregressive["pitch_accuracy"] * 100
                    ar_onset = autoregressive["onset_accuracy"] * 100
                    ar_dur = autoregressive["duration_accuracy"] * 100
                    log_data.update(
                        {
                            "val/ar_pitch_accuracy": ar_pitch,
                            "val/ar_onset_accuracy": ar_onset,
                            "val/ar_duration_accuracy": ar_dur,
                        }
                    )
                    ar_msg = (
                        f"pitch {ar_pitch:.2f}%, onset {ar_onset:.2f}%, duration {ar_dur:.2f}%"
                    )

                    validation_steps.append(validation_step)
                    pitch_error_history.append(autoregressive["slot_pitch_error_freq"])
                    onset_mae_history.append(autoregressive["slot_onset_mae"])
                    duration_mae_history.append(autoregressive["slot_duration_mae"])

                    # Render heatmaps with matplotlib. PNGs are always written
                    # locally under <output_dir>/heatmaps/; the same figures are
                    # attached to the wandb log (as images) when wandb is enabled.
                    pitch_map = build_error_heatmap_chart(
                        pitch_error_history, validation_steps,
                        output_dir=args.output_dir,
                        metric_key="pitch_error_freq",
                        title="Autoregressive pitch error frequency",
                        value_label="pitch error frequency",
                        cmap="magma",
                    )
                    onset_map = build_error_heatmap_chart(
                        onset_mae_history, validation_steps,
                        output_dir=args.output_dir,
                        metric_key="onset_mae",
                        title="Autoregressive onset MAE (10 ms bins)",
                        value_label="onset MAE (bins)",
                        cmap="viridis",
                    )
                    duration_map = build_error_heatmap_chart(
                        duration_mae_history, validation_steps,
                        output_dir=args.output_dir,
                        metric_key="duration_mae",
                        title="Autoregressive duration MAE (10 ms bins)",
                        value_label="duration MAE (bins)",
                        cmap="viridis",
                    )
                    if use_wandb:
                        if pitch_map is not None:
                            log_data["heatmaps/pitch_error_freq"] = pitch_map
                        if onset_map is not None:
                            log_data["heatmaps/onset_mae"] = onset_map
                        if duration_map is not None:
                            log_data["heatmaps/duration_mae"] = duration_map
                else:
                    ar_msg = "(skipped)"

                if use_wandb:
                    wandb.log(log_data, step=validation_step)

                print(
                    f"Validation Loss: {val_loss:.4f} "
                    f"(onset {loss_by_type['onset']:.4f}, "
                    f"duration {loss_by_type['duration']:.4f}, "
                    f"pitch {loss_by_type['pitch']:.4f}), "
                    f"Teacher-Forced Pitch Accuracy: {val_acc * 100:.2f}%, "
                    f"Autoregressive Accuracy: {ar_msg}"
                )

            accelerator.wait_for_everyone()
            model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()
        
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
                                    current_lr = scheduler.get_last_lr()[0]
                                    if use_wandb:
                                        train_log = {
                                            "train/loss": reduced_loss,
                                            "train/learning_rate": current_lr,
                                        }
                                        if reduced_l2_penalty is not None:
                                            train_log["train/anchor_l2"] = reduced_l2_penalty
                                            train_log["train/anchor_term"] = reduced_anchor_term
                                        wandb.log(train_log, step=completed_steps)

                                    # Print more precise learning rate
                                    l2_detail = ""
                                    if reduced_l2_penalty is not None:
                                        l2_detail = (
                                            f", AnchorL2: {reduced_l2_penalty:.6e}, "
                                            f"AnchorTerm: {reduced_anchor_term:.6e}"
                                        )
                                    print(
                                        f"Step: {completed_steps}/{args.max_steps}, Loss: {reduced_loss:.4f}, "
                                        f"LR: {current_lr:.8e}{l2_detail}"
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
                                    run_validation(f"step {completed_steps}")
                                
                                # Save checkpoint (with validation)
                                if is_checkpoint_step:
                                    # Run validation before saving checkpoint
                                    run_validation(f"checkpoint step {completed_steps}")
                                    
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
                            print("Current memory usage:")
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
                    run_validation(f"final step {completed_steps}")
                    
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
                    accelerator.wait_for_everyone()

                except Exception as save_error:
                    print(f"Error saving final model: {save_error}")

            if use_wandb:
                wandb.finish()

    except Exception as setup_error:
        print(f"Error in setup: {setup_error}")
        print(traceback.format_exc())
        # Re-raise so the process exits non-zero: cluster schedulers (and job
        # dependencies like afterok) must see failed training as failed.
        raise

if __name__ == "__main__":
    main()
