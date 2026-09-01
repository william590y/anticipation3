"""CPU tests for LTLM AR decode, slot accuracies, and piano-roll tensors."""

from __future__ import annotations

import types

import matplotlib.pyplot as plt
import torch

from anticipation.config import MAX_DUR, MAX_PITCH, MAX_TIME
from anticipation.ltlm_eval import (
    EVAL_MODES,
    accumulate_score_slot_stats,
    checkpoint_step,
    empty_slot_accums,
    finalize_slot_accums,
    fixed_piano_roll_indices,
    is_complete_checkpoint,
    list_complete_checkpoints,
    ltlm_autoregressive_generate_score,
    packed_notes_for_rolls,
    remap_checkpoint_state,
    wandb_ar_payload,
)
from anticipation.ltlm_model import LTLMOutput
from anticipation.packed_sequence import ALTERNATING_START, dummy_rest_triplet, iter_score_slot_positions
from anticipation.vocab import (
    ADUR_OFFSET,
    ANOTE_OFFSET,
    ATIME_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    REST,
    TIME_OFFSET,
)
from plan_vq import BODY_SLOTS, NUM_PERF_NOTES, PACKED_LENGTH
from plan_vq_viz import piano_roll_figure


class _ScriptedLTLM(torch.nn.Module):
    """Tiny stand-in: each forward's last-position argmax is the next scripted token."""

    def __init__(self, vocab: int, script: list[int]):
        super().__init__()
        self.vocab = vocab
        self.script = list(script)
        self.calls = 0
        self.blocks = [types.SimpleNamespace(thoughts=None)]
        self.thoughts_seen = []

    def set_thoughts(self, z):
        self.blocks[0].thoughts = z

    def forward(self, input_ids, past_key_values=None, use_cache=True, **kwargs):
        self.thoughts_seen.append(self.blocks[0].thoughts)
        bsz, seqlen = input_ids.shape
        logits = torch.zeros(bsz, seqlen, self.vocab)
        idx = min(self.calls, len(self.script) - 1)
        logits[:, -1, self.script[idx]] = 20.0
        self.calls += 1
        return LTLMOutput(loss=None, logits=logits, past_key_values=0)


def _short_packed():
    """Prefix of 6 tokens, then one score triplet + one control triplet (length 12)."""
    score = [TIME_OFFSET + 10, DUR_OFFSET + 5, NOTE_OFFSET + 60]
    control = [ATIME_OFFSET + 12, ADUR_OFFSET + 6, ANOTE_OFFSET + 61]
    prefix = [
        ATIME_OFFSET + 1, ADUR_OFFSET + 2, ANOTE_OFFSET + 40,
        TIME_OFFSET, DUR_OFFSET, REST,
    ]
    return torch.tensor([prefix + score + control], dtype=torch.long)


def test_ar_decode_writes_scripted_score_and_copies_controls():
    tokens = _short_packed()
    script = [TIME_OFFSET + 3, DUR_OFFSET + 4, NOTE_OFFSET + 50, 0, 0, 0, 0]
    model = _ScriptedLTLM(vocab=NOTE_OFFSET + 200, script=script)
    z = torch.ones(1, 1, 4)
    generated = ltlm_autoregressive_generate_score(
        model,
        tokens,
        z,
        device=torch.device("cpu"),
        constrain_score_tokens=False,
        ground_truth_score_tokens_to_feed=0,
        score_start_idx=6,
    )
    assert generated.shape == tokens.shape
    assert generated[0, :6].tolist() == tokens[0, :6].tolist()
    assert generated[0, 6:9].tolist() == script[:3]
    assert generated[0, 9:12].tolist() == tokens[0, 9:12].tolist()
    assert all(t is not None for t in model.thoughts_seen)
    assert model.blocks[0].thoughts is None


def test_slot_accuracy_counts_one_pitch_mismatch():
    length = ALTERNATING_START + 6
    reference = torch.zeros(2, length, dtype=torch.long)
    first = next(iter_score_slot_positions(length))
    reference[:, first] = TIME_OFFSET + 8
    reference[:, first + 1] = DUR_OFFSET + 3
    reference[:, first + 2] = NOTE_OFFSET + 40
    generated = reference.clone()
    generated[0, first + 2] = NOTE_OFFSET + 41
    slots = 1
    accums = empty_slot_accums(slots, torch.device("cpu"))
    accumulate_score_slot_stats(generated, reference, slots, accums)
    stats = finalize_slot_accums(accums)
    assert stats["total_notes"] == 2
    assert abs(stats["pitch_accuracy"] - 0.5) < 1e-9
    assert stats["onset_accuracy"] == 1.0
    assert stats["duration_accuracy"] == 1.0


def test_rest_slots_are_ignored():
    length = ALTERNATING_START + 6
    reference = torch.zeros(1, length, dtype=torch.long)
    pos = next(iter_score_slot_positions(length))
    reference[0, pos + 2] = REST
    generated = reference.clone()
    generated[0, pos] = TIME_OFFSET + 99
    accums = empty_slot_accums(1, torch.device("cpu"))
    accumulate_score_slot_stats(generated, reference, 1, accums)
    stats = finalize_slot_accums(accums)
    assert stats["total_notes"] == 0
    assert stats["pitch_accuracy"] == 0.0


def _make_packed_window(seed=0):
    rng = torch.Generator().manual_seed(seed)
    spacing_p = max(1, (MAX_TIME - 20) // NUM_PERF_NOTES)
    spacing_s = max(1, (MAX_TIME - 3) // BODY_SLOTS)

    def rand(high):
        return int(torch.randint(0, high, (1,), generator=rng).item())

    perf = [
        (spacing_p * i + rand(5), 20 + rand(10), 21 + rand(40))
        for i in range(NUM_PERF_NOTES)
    ]
    score = [
        (spacing_s * i + rand(2), 25 + rand(10), 21 + rand(40))
        for i in range(BODY_SLOTS)
    ]
    tokens = []
    for i in range(ALTERNATING_START // 6):
        onset, duration, pitch = perf[i]
        tokens += [ATIME_OFFSET + onset, ADUR_OFFSET + duration, ANOTE_OFFSET + pitch]
        tokens += dummy_rest_triplet(0)
    for slot in range(BODY_SLOTS):
        onset, duration, pitch = score[slot]
        tokens += [TIME_OFFSET + onset, DUR_OFFSET + duration, NOTE_OFFSET + pitch]
        onset, duration, pitch = perf[ALTERNATING_START // 6 + slot]
        tokens += [ATIME_OFFSET + onset, ADUR_OFFSET + duration, ANOTE_OFFSET + pitch]
    assert len(tokens) == PACKED_LENGTH
    return torch.tensor(tokens, dtype=torch.long), perf, score


def test_packed_notes_match_slot_layout():
    packed, perf_notes, score_notes = _make_packed_window(1)
    perf, score, valid = packed_notes_for_rolls(packed.unsqueeze(0))
    assert tuple(perf.shape) == (1, NUM_PERF_NOTES, 3)
    assert tuple(score.shape) == (1, BODY_SLOTS, 3)
    assert bool(valid.all())
    assert perf[0].tolist() == [list(note) for note in perf_notes]
    assert score[0].tolist() == [list(note) for note in score_notes]
    assert (perf[..., 0] < MAX_TIME).all()
    assert (score[..., 1] < MAX_DUR).all()
    assert (score[..., 2] < MAX_PITCH).all()


def test_piano_roll_figure_three_columns():
    packed, _, _ = _make_packed_window(2)
    batch = packed.unsqueeze(0)
    perf, score, valid = packed_notes_for_rolls(batch)
    predicted = score.clone()
    predicted[0, 0, 2] = (predicted[0, 0, 2] + 7) % MAX_PITCH
    fig = piano_roll_figure(
        perf, score, valid, predicted,
        step=2500, max_examples=1,
        suptitle="LTLM AR decode (planner)",
        pred_heading="generated score (planner)",
        window_ids=[0],
    )
    assert fig.axes is not None and len(fig.axes) == 3
    titles = [ax.get_title() or "" for ax in fig.axes]
    assert any("window 0" in title for title in titles)
    assert any("performance" in title for title in titles)
    assert "planner" in fig._suptitle.get_text()
    plt.close(fig)


def test_wandb_keys_match_previous_ltlm_runs():
    ar = {
        mode: {"pitch_accuracy": 0.5, "onset_accuracy": 0.25, "duration_accuracy": 0.1}
        for mode in EVAL_MODES
    }
    payload = wandb_ar_payload(ar, figures={"oracle": "fig"})
    for mode in EVAL_MODES:
        assert f"val/{mode}/ar_pitch_accuracy" in payload
        assert f"val/{mode}/ar_onset_accuracy" in payload
        assert f"val/{mode}/ar_duration_accuracy" in payload
    assert payload["val/oracle/ar_pitch_accuracy"] == 50.0
    assert payload["val/ar_pitch_accuracy"] == 50.0
    assert payload["val/ar_onset_accuracy"] == 25.0
    assert payload["val/ar_duration_accuracy"] == 10.0
    assert payload["val/oracle/piano_roll"] == "fig"
    assert "val/planner/ar_pitch_accuracy" in payload
    captioned = wandb_ar_payload(ar, figures={"oracle": "fig"}, wandb_module=None, step=2500)
    assert captioned["val/oracle/piano_roll"] == "fig"


def test_remap_keeps_planner_and_prefixes_base_model():
    safetensors = {"transformer.h.0.block.ln_1.weight": torch.ones(2)}
    extra = {
        "planner.to_mu.weight": torch.ones(3, 3),
        "blocks.0.block.ln_1.weight": torch.zeros(2),
    }
    state = remap_checkpoint_state(safetensors, extra)
    assert "base_model.transformer.h.0.block.ln_1.weight" in state
    assert "planner.to_mu.weight" in state
    assert "blocks.0.block.ln_1.weight" not in state


def test_checkpoint_discovery(tmp_path):
    ready = tmp_path / "checkpoint-2500"
    ready.mkdir()
    (ready / "model.safetensors").write_bytes(b"x")
    (ready / "ltlm_extra.pt").write_bytes(b"y")
    partial = tmp_path / "checkpoint-5000"
    partial.mkdir()
    (partial / "model.safetensors").write_bytes(b"x")
    assert is_complete_checkpoint(ready)
    assert not is_complete_checkpoint(partial)
    assert list_complete_checkpoints(tmp_path) == [ready]
    assert checkpoint_step(ready) == 2500


def test_fixed_piano_roll_indices():
    class _Val:
        def __len__(self):
            return 10

    assert fixed_piano_roll_indices(_Val(), 3) == [0, 1, 2]
    assert fixed_piano_roll_indices(_Val(), 0) == []
    assert fixed_piano_roll_indices(_Val(), 99) == list(range(10))
