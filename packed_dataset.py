"""The packed ASAP window dataset and its on-the-fly augmentation.

Extracted verbatim from ``train.py`` so that code which must NOT import
``transformers`` can still read the training data. ``moonbeam_encoder`` shadows
the installed transformers with Moonbeam's 4.42 fork for the whole process, so
stage 1 of the plan pipeline cannot go anywhere near ``train.py``'s imports.

``train.py`` imports these names back from here, so there is exactly one copy.
"""
from __future__ import annotations

import warnings

import torch
from torch.utils.data import Dataset

from anticipation.config import CONTEXT_SIZE, MAX_DUR, MAX_PITCH, MAX_TIME
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
