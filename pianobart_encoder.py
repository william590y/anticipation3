#!/usr/bin/env python
"""Frozen PianoBART as the plan encoder.

PianoBART (Liang et al., ICME 2024) is a pretrained BART over *Octuple* note
tokens, pretrained on piano corpora including ASAP itself. We use its **encoder**
as a frozen feature extractor for stage 1 instead of training one from scratch:
one bidirectional forward over the score's Octuple tokens gives a contextual
1024-d vector per score note, which ``plan_vq.py`` pools into ``num_codes``
latents with learned cross-attention queries.

Why the encoder is given the **score only**, not the performance
----------------------------------------------------------------
Octuple has no absolute-time field. A note's timing is (bar, position-in-bar) on
a metrical grid quantized to 1/16 of a beat, so the representation can only
describe music that has already been placed on a beat grid:

- The **score** is exactly that. ``tokenize-asap-sliding.py`` normalizes score
  onsets to a fixed ``TARGET_BEAT_INTERVAL = 0.5`` s beat, i.e. a literal 120 BPM
  quarter-note grid, so the conversion below is lossless up to Octuple's own
  1/16-beat resolution.
- The **performance** is not. Putting it in Octuple needs a beat grid, and there
  are only two on offer: the score's (which would hand the encoder the metrical
  positions the whole model exists to predict) or a nominal constant one (which
  quantizes expressive timing to 31 ms and misstates every metrical position).

So the performance stays outside PianoBART, encoded by the small trainable
module in ``plan_vq.py`` that reads raw 10 ms bins. That module conditions both
the pooling head and the decoder, which is what keeps the plan carrying only
what the performance does not already determine.

Unlike the Moonbeam encoder this replaces, PianoBART runs on **stock
transformers** -- it is a plain ``BartModel`` with custom input embeddings -- so
there is no fork shadowing ``transformers`` process-wide and stage 2 can call
this module directly instead of reading precomputed codes.
"""
from __future__ import annotations

import importlib.util
import json
import math
import pickle
from dataclasses import dataclass
from pathlib import Path

import torch

from anticipation.config import TIME_RESOLUTION
from anticipation.asap_aligned_stream import TARGET_BEAT_INTERVAL

REPO_ROOT = Path(__file__).resolve().parent
PIANOBART_ROOT = REPO_ROOT / "external" / "pianobart"
PIANOBART_REPO = "RS2002/PianoBART"

# Octuple grid constants, from PianoBART's own Data/data_generation/make_dict.py.
POS_RESOLUTION = 16      # position/duration units per beat (quarter note)
DURATION_MAX = 8         # longest *linearly* coded duration, in beats
VELOCITY_QUANT = 4
TEMPO_QUANT = 12         # tempo bins per octave
MIN_TEMPO = 16.0
MAX_BAR = 255

# Our score grid: 50 bins of 10 ms per beat, i.e. a 120 BPM quarter note.
BINS_PER_BEAT = round(TIME_RESOLUTION * TARGET_BEAT_INTERVAL)
SCORE_TEMPO_BPM = 60.0 / TARGET_BEAT_INTERVAL

# A packed window carries no meter, so score bars are cut at a fixed 4/4. The
# phase is unknown too: the window's first score note is treated as a downbeat.
# Only the *relative* metrical structure inside the window has to be right for
# the pooled plan, and that is unaffected by where the barlines land.
BEATS_PER_BAR = 4
SCORE_TIME_SIGNATURE = (4, 4)
# The score has no velocity (tokenization drops it); a constant mid-range value
# keeps that field from carrying spurious variation.
SCORE_VELOCITY = 64
PIANO_INSTRUMENT = 0

OCTUPLE_FIELDS = ("Bar", "Position", "Instrument", "Pitch", "Duration", "Velocity", "TimeSig", "Tempo")


def _duration_table():
    """Octuple's duration buckets: index by duration in 1/16-beat units.

    Bucket width doubles every ``POS_RESOLUTION`` buckets, so the 128 buckets
    span 255 beats. Copied structurally from ``convert.py``'s ``dur_enc``.
    """
    table = []
    buckets = 0
    for i in range(DURATION_MAX):
        for _ in range(POS_RESOLUTION):
            table.extend([buckets] * (2**i))
            buckets += 1
    return table


@dataclass
class PianoBartMeta:
    """Everything the conversion needs, resolved from the released vocabulary.

    Field indices are looked up by *name* rather than hardcoded, so a change to
    the vocabulary file fails loudly here instead of silently shifting every
    token by a constant.
    """

    hidden_size: int
    field_order: tuple
    pad_word: torch.Tensor          # (8,) the per-field <PAD> index
    duration_table: torch.Tensor    # (4080,) units -> bucket
    max_position: int
    max_pitch: int
    instrument: int
    velocity: int
    time_signature: int
    tempo: int

    def to(self, device):
        self.pad_word = self.pad_word.to(device)
        self.duration_table = self.duration_table.to(device)
        return self


def _load_model_module():
    """Import the released ``model.py`` without claiming the name ``model``."""
    path = PIANOBART_ROOT / "model.py"
    if not path.is_file():
        raise FileNotFoundError(
            f"PianoBART is not vendored at {PIANOBART_ROOT}. Fetch model.py, config.json "
            f"and Octuple.pkl from https://huggingface.co/{PIANOBART_REPO}."
        )
    spec = importlib.util.spec_from_file_location("pianobart_released_model", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_meta(e2w, w2e, hidden_size):
    """Resolve the constant Octuple fields against the released vocabulary.

    The bucketed fields (velocity, tempo) are computed with PianoBART's own
    quantizers and then *checked* against the name the vocabulary gives that
    index. Looking them up by name alone would hinge on float formatting;
    computing them alone would silently survive a reordered vocabulary.
    """

    def index(field, name):
        try:
            return e2w[field][f"{field} {name}"]
        except KeyError as error:
            raise KeyError(f"Octuple.pkl has no '{field} {name}' entry") from error

    def check(field, position, expected, tolerance):
        name = w2e[field][position].removeprefix(f"{field} ")
        if abs(float(name) - expected) > tolerance:
            raise RuntimeError(
                f"Octuple {field} bucket {position} is '{name}', which is not the bucket "
                f"for {expected}; the vocabulary is not the one this conversion assumes."
            )
        return position

    numerator, denominator = SCORE_TIME_SIGNATURE
    velocity = SCORE_VELOCITY // VELOCITY_QUANT
    tempo = round(math.log2(SCORE_TEMPO_BPM / MIN_TEMPO) * TEMPO_QUANT)
    return PianoBartMeta(
        hidden_size=hidden_size,
        field_order=OCTUPLE_FIELDS,
        pad_word=torch.tensor([index(field, "<PAD>") for field in OCTUPLE_FIELDS], dtype=torch.long),
        duration_table=torch.tensor(_duration_table(), dtype=torch.long),
        # The six special tokens live at the end of each field.
        max_position=len(e2w["Position"]) - 6,
        max_pitch=len(e2w["Pitch"]) - 6,
        instrument=index("Instrument", PIANO_INSTRUMENT),
        velocity=check("Velocity", velocity, SCORE_VELOCITY, VELOCITY_QUANT),
        time_signature=index("TimeSig", f"{numerator}/{denominator}"),
        tempo=check("Tempo", tempo, SCORE_TEMPO_BPM, 0.05 * SCORE_TEMPO_BPM),
    )


# The released checkpoint ties these parameters to a second name and safetensors
# keeps only one copy of a shared tensor. Loading the survivor also fills its
# alias (they are the same ``Parameter`` object), but the load then *reports*
# the alias as missing -- so without this map a strict check would reject a
# perfectly good checkpoint, and a non-strict load would hide a real one.
TIED_ALIASES = {
    "encoder_linear.weight": "decoder_linear.weight",
    "encoder_linear.bias": "decoder_linear.bias",
    "bart.shared.weight": "bart.decoder.embed_tokens.weight",
    "bart.encoder.embed_tokens.weight": "bart.decoder.embed_tokens.weight",
}


def load_pianobart(device="cpu", dtype=torch.float32, checkpoint_path=None):
    """Load the frozen pretrained PianoBART. Returns ``(model, meta)``.

    Only the encoder half is kept: the plan reads contextual note states, and
    the decoder is 100M parameters of dead weight in this pipeline.
    """
    from transformers import BartConfig

    released = _load_model_module()
    with open(PIANOBART_ROOT / "Octuple.pkl", "rb") as handle:
        e2w, w2e = pickle.load(handle)
    settings = json.loads((PIANOBART_ROOT / "config.json").read_text())

    config = BartConfig(
        max_position_embeddings=settings["max_position_embeddings"],
        d_model=settings["hidden_size"],
        encoder_layers=settings["layers"],
        encoder_ffn_dim=settings["ffn_dims"],
        encoder_attention_heads=settings["heads"],
        decoder_layers=settings["layers"],
        decoder_ffn_dim=settings["ffn_dims"],
        decoder_attention_heads=settings["heads"],
    )
    model = released.PianoBart(bartConfig=config, e2w=e2w, w2e=w2e)

    if checkpoint_path is None:
        from huggingface_hub import hf_hub_download

        checkpoint_path = hf_hub_download(PIANOBART_REPO, "model.safetensors")

    from safetensors.torch import load_file

    state = load_file(str(checkpoint_path))
    state = {(k[len("model.") :] if k.startswith("model.") else k): v for k, v in state.items()}
    for name, alias in TIED_ALIASES.items():
        if name not in state and alias in state:
            state[name] = state[alias]

    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            f"PianoBART did not load cleanly: {len(missing)} missing, {len(unexpected)} "
            f"unexpected keys. First missing: {missing[:5]}. A partly-random 'pretrained' "
            f"encoder is the failure mode this check exists to prevent."
        )

    del model.bart.decoder
    model.to(device=device, dtype=dtype)
    model.eval()
    model.requires_grad_(False)
    return model, _build_meta(e2w, w2e, config.d_model).to(device)


def _to_units(bins):
    """10 ms bins -> Octuple 1/16-beat units, rounded half up."""
    return torch.div(bins * POS_RESOLUTION + BINS_PER_BEAT // 2, BINS_PER_BEAT, rounding_mode="floor")


def score_to_octuple(score, score_valid, meta):
    """Score notes ``(B, N, 3)`` of raw bins -> ``(octuple, attention_mask)``.

    ``octuple`` is ``(B, N, 8)`` of vocabulary indices in ``meta.field_order``;
    ``attention_mask`` is 1 on real notes. Invalid slots are filled with
    PianoBART's own ``<PAD>`` word and masked out.
    """
    onset, duration, pitch = score[..., 0], score[..., 1], score[..., 2]
    if score_valid is None:
        score_valid = torch.ones_like(pitch, dtype=torch.bool)

    # Anchor the window at its first real note so bar indices stay in range.
    anchored = onset - onset.masked_fill(~score_valid, torch.iinfo(onset.dtype).max).amin(
        dim=1, keepdim=True
    ).clamp_min(0)
    units = _to_units(anchored.clamp_min(0))
    measure = BEATS_PER_BAR * POS_RESOLUTION

    bar = torch.div(units, measure, rounding_mode="floor").clamp_(0, MAX_BAR)
    position = (units % measure).clamp_(0, meta.max_position - 1)
    duration_units = _to_units(duration).clamp_(0, meta.duration_table.numel() - 1)
    fields = {
        "Bar": bar,
        "Position": position,
        "Instrument": torch.full_like(bar, meta.instrument),
        "Pitch": pitch.clamp(0, meta.max_pitch - 1),
        "Duration": meta.duration_table[duration_units],
        "Velocity": torch.full_like(bar, meta.velocity),
        "TimeSig": torch.full_like(bar, meta.time_signature),
        "Tempo": torch.full_like(bar, meta.tempo),
    }

    octuple = torch.stack([fields[name] for name in meta.field_order], dim=-1)
    pad = meta.pad_word.view(1, 1, -1).expand_as(octuple)
    octuple = torch.where(score_valid.unsqueeze(-1), octuple, pad)
    return octuple, score_valid.to(torch.long)


@torch.no_grad()
def pianobart_features(model, octuple, attention_mask=None):
    """Frozen PianoBART encoder states, ``(B, N, hidden_size)``."""
    parameter = next(model.parameters())
    octuple = octuple.to(parameter.device)
    if attention_mask is not None:
        attention_mask = attention_mask.to(parameter.device)
    output = model(input_ids_encoder=octuple, encoder_attention_mask=attention_mask)
    return output.last_hidden_state.float()


@torch.no_grad()
def score_features(model, meta, score, score_valid):
    """``split_packed_batch`` score tensors -> frozen features, ``(B, N, H)``."""
    octuple, mask = score_to_octuple(score, score_valid, meta)
    return pianobart_features(model, octuple, mask)
