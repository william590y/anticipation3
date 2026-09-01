#!/usr/bin/env python
"""Stage 1: a vector-quantized "plan" for the score, conditioned on the performance.

The score encoder is **frozen pretrained PianoBART** (see
``pianobart_encoder.py``): a bidirectional pass over the score's Octuple tokens
gives one contextual 1024-d state per score note. ``num_codes`` learned queries
cross-attend over those states to pool them into latents, each quantized against
a shared codebook; a decoder then has to rebuild the score from (performance,
codes) alone.

The **performance** is encoded by the small trainable module here rather than by
PianoBART, because Octuple cannot represent it: all of its timing is
(bar, position) on a metrical grid, and putting the performance on one would
either leak the score's grid or quantize away the expressive timing that is the
model's entire input. See ``pianobart_encoder``'s module docstring.

**Both the pooling head and the decoder see the performance**, so the codes only
have to carry what the performance does not already determine -- which, on this
task, is almost entirely score *timing* (pitch is a near-free copy of the
aligned control; see CLAUDE.md's exposure-gap table). Nothing routes the score
to the decoder except through the quantizer, so the bottleneck is real.

Only the performance encoder, the pooling queries, the codebook and the decoder
are trained; PianoBART never receives a gradient. This module takes its frozen
features as plain tensors, so it has no dependency on the encoder itself.

The codes are what stage 2 (``train_plan_lm.py``) prepends to the packed
sequence as a plan.

Everything is derived from the packed alternating window, so this module needs
no new tokenization: score slot ``k`` pairs with control ``k`` (packed format
invariant), which lets the encoder and decoder align a score slot with its
performance note by a shared note-index embedding.

Layout of a 1020-token packed window::

    [0, 192)      32 x (control, dummy-rest)   -> controls 0..31   (lookahead)
    [192, 1020)   138 x (score, control)       -> controls 32..169, score 0..137

so a window holds 170 performance notes and 138 score notes.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from anticipation.config import MAX_DUR, MAX_PITCH, MAX_TIME
from anticipation.packed_sequence import ALTERNATING_START, PREFIX_CONTROLS
from anticipation.vocab import (
    ADUR_OFFSET,
    ANOTE_OFFSET,
    ATIME_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    REST,
    TIME_OFFSET,
)

PACKED_LENGTH = 1020
BODY_SLOTS = (PACKED_LENGTH - ALTERNATING_START) // 6      # 138 score notes
NUM_PERF_NOTES = PREFIX_CONTROLS + BODY_SLOTS              # 170 controls

TOKEN_TYPE_NAMES = ("onset", "duration", "pitch")


# ---------------------------------------------------------------------------
# packed window -> note tensors
# ---------------------------------------------------------------------------

def packed_slot_positions(device=None):
    """Token offsets of the performance notes and score notes in a packed window."""
    prefix_controls = torch.arange(0, ALTERNATING_START, 6, device=device)
    score_slots = ALTERNATING_START + 6 * torch.arange(BODY_SLOTS, device=device)
    perf_slots = torch.cat([prefix_controls, score_slots + 3])
    return perf_slots, score_slots


def split_packed_batch(input_ids):
    """Split packed windows into raw (onset, duration, pitch) note tensors.

    Returns ``(perf, score, score_valid)`` with shapes ``(B, 170, 3)``,
    ``(B, 138, 3)`` and ``(B, 138)``. Values are raw bin/pitch numbers, not
    vocabulary tokens. Performance note ``k`` is the partner of score slot
    ``k`` for ``k < 138``; the trailing 32 controls have no score note.
    """
    if input_ids.dim() != 2:
        raise ValueError(f"expected (batch, length) input_ids, got {tuple(input_ids.shape)}")
    if input_ids.shape[1] != PACKED_LENGTH:
        raise ValueError(
            f"plan_vq expects {PACKED_LENGTH}-token packed windows, got {input_ids.shape[1]}"
        )

    perf_slots, score_slots = packed_slot_positions(input_ids.device)

    def gather(slots):
        return torch.stack(
            [input_ids[:, slots], input_ids[:, slots + 1], input_ids[:, slots + 2]], dim=-1
        )

    perf_tok = gather(perf_slots)
    score_tok = gather(score_slots)

    perf = torch.stack(
        [
            (perf_tok[..., 0] - ATIME_OFFSET).clamp_(0, MAX_TIME - 1),
            (perf_tok[..., 1] - ADUR_OFFSET).clamp_(0, MAX_DUR - 1),
            # note = instrument * 128 + pitch; ASAP is single-instrument piano.
            (perf_tok[..., 2] - ANOTE_OFFSET).remainder_(MAX_PITCH),
        ],
        dim=-1,
    )

    score_valid = score_tok[..., 2] != REST
    score = torch.stack(
        [
            (score_tok[..., 0] - TIME_OFFSET).clamp_(0, MAX_TIME - 1),
            (score_tok[..., 1] - DUR_OFFSET).clamp_(0, MAX_DUR - 1),
            torch.where(
                score_valid,
                (score_tok[..., 2] - NOTE_OFFSET).remainder(MAX_PITCH),
                torch.zeros_like(score_tok[..., 2]),
            ),
        ],
        dim=-1,
    )
    return perf, score, score_valid


# ---------------------------------------------------------------------------
# modules
# ---------------------------------------------------------------------------

def _transformer(d_model, n_head, n_layer, dropout):
    layer = nn.TransformerEncoderLayer(
        d_model=d_model,
        nhead=n_head,
        dim_feedforward=4 * d_model,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(
        layer, num_layers=n_layer, norm=nn.LayerNorm(d_model), enable_nested_tensor=False
    )


def _cross_attention_stack(d_model, n_head, n_layer, dropout):
    """Queries attend to each other, then to the memory. No causal masking."""
    layer = nn.TransformerDecoderLayer(
        d_model=d_model,
        nhead=n_head,
        dim_feedforward=4 * d_model,
        dropout=dropout,
        activation="gelu",
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerDecoder(layer, num_layers=n_layer, norm=nn.LayerNorm(d_model))


class FourierTime(nn.Module):
    """Sinusoidal features of a time in 10 ms bins.

    Times are continuous quantities spanning thousands of bins, so an embedding
    table per bin would be both large and unable to say that two adjacent bins
    are nearly the same instant -- which matters here, where the score onset the
    decoder must predict differs from the performance onset by a small offset.
    Periods are log-spaced from 2 bins up to the field's full range.
    """

    def __init__(self, num_bands, max_period):
        super().__init__()
        periods = torch.exp(torch.linspace(math.log(2.0), math.log(float(max_period)), num_bands))
        self.register_buffer("frequencies", 2.0 * math.pi / periods)
        self.output_dim = 2 * num_bands

    def forward(self, bins):
        angles = bins.float().unsqueeze(-1) * self.frequencies
        return torch.cat([angles.sin(), angles.cos()], dim=-1)


class PerformanceEncoder(nn.Module):
    """Trainable encoder over raw performance notes, at full 10 ms resolution."""

    def __init__(self, config):
        super().__init__()
        d = config.d_model
        self.onset = FourierTime(config.fourier_bands, MAX_TIME)
        self.duration = FourierTime(config.fourier_bands, MAX_DUR)
        self.pitch = nn.Embedding(MAX_PITCH, d)
        self.project = nn.Linear(self.onset.output_dim + self.duration.output_dim + d, d)
        self.transformer = _transformer(d, config.n_head, config.perf_layers, config.dropout)

    def forward(self, perf, note_index):
        features = torch.cat(
            [self.onset(perf[..., 0]), self.duration(perf[..., 1]), self.pitch(perf[..., 2])],
            dim=-1,
        )
        return self.transformer(self.project(features) + note_index)


class VectorQuantizerEMA(nn.Module):
    """VQ with an EMA-updated codebook, dead-code restarts, and DDP-safe stats.

    The EMA statistics are buffers, which DDP does not reduce, so the batch
    statistics are all-reduced here before the update: every rank then applies
    the *same* update and the codebooks stay bit-identical without relying on
    ``broadcast_buffers``.
    """

    def __init__(
        self,
        codebook_size,
        code_dim,
        decay=0.99,
        eps=1e-5,
        commitment=0.25,
        l2_normalize=True,
        restart_threshold=0.01,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.code_dim = code_dim
        self.decay = decay
        self.eps = eps
        self.commitment = commitment
        self.l2_normalize = l2_normalize
        self.restart_threshold = restart_threshold

        codebook = torch.randn(codebook_size, code_dim)
        if l2_normalize:
            codebook = F.normalize(codebook, dim=-1)
        self.register_buffer("codebook", codebook)
        self.register_buffer("cluster_size", torch.zeros(codebook_size))
        self.register_buffer("codebook_avg", codebook.clone())

    def lookup(self, indices):
        return F.embedding(indices, self.codebook)

    def _all_reduce(self, tensor):
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        return tensor

    def _restart_dead_codes(self, flat):
        """Re-seed codes nothing is claiming with random encoder outputs.

        The test is scale-free -- a code is dead when its EMA share of the
        assignments falls below ``restart_threshold`` times an equal share --
        rather than an absolute count. An absolute threshold is wrong in both
        directions here: early in training every count is small because the EMA
        is still filling in, and at convergence the expected count per code is
        ``batch * num_codes / codebook_size``, which for a large codebook is
        well under 1.

        A revived code is seeded at an equal share so it gets roughly
        ``1 / (1 - decay)`` steps to earn its keep before it is eligible again.
        Rank 0's candidates are broadcast so every rank re-seeds identically.
        """
        uniform_share = self.cluster_size.sum() / self.codebook_size
        dead = self.cluster_size < self.restart_threshold * uniform_share
        num_dead = int(dead.sum().item())
        if num_dead == 0:
            return 0

        candidates = flat.detach()
        if dist.is_available() and dist.is_initialized():
            candidates = candidates.contiguous()
            dist.broadcast(candidates, src=0)

        pick = torch.randint(0, candidates.shape[0], (num_dead,), device=candidates.device)
        replacement = candidates[pick]
        if self.l2_normalize:
            replacement = F.normalize(replacement, dim=-1)
        self.codebook[dead] = replacement
        self.codebook_avg[dead] = replacement * uniform_share
        self.cluster_size[dead] = uniform_share
        return num_dead

    def forward(self, z_e):
        """``z_e``: (B, K, code_dim). Returns (z_q, indices, loss, stats).

        Runs in fp32 even under bf16 autocast: there are only ``B * K`` vectors
        so it costs nothing, and the EMA accumulators need the precision.
        """
        input_dtype = z_e.dtype
        z_e = z_e.float()
        flat = z_e.reshape(-1, self.code_dim)
        if self.l2_normalize:
            flat = F.normalize(flat, dim=-1)

        # ||x - e||^2 = ||x||^2 - 2 x.e + ||e||^2 (the ||x||^2 term is constant
        # per row, but keeping it makes the returned distance meaningful).
        distances = (
            flat.pow(2).sum(dim=1, keepdim=True)
            - 2.0 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(dim=1)
        )
        indices = distances.argmin(dim=1)
        quantized = self.lookup(indices)

        counts = torch.zeros(self.codebook_size, device=flat.device, dtype=flat.dtype)
        counts.scatter_add_(0, indices, torch.ones_like(indices, dtype=flat.dtype))
        probs = counts / counts.sum().clamp_min(1.0)
        perplexity = torch.exp(-(probs * torch.log(probs + 1e-10)).sum())

        num_restarted = 0
        if self.training:
            with torch.no_grad():
                code_sum = torch.zeros_like(self.codebook_avg)
                code_sum.index_add_(0, indices, flat.detach())
                self._all_reduce(counts)
                self._all_reduce(code_sum)

                self.cluster_size.mul_(self.decay).add_(counts, alpha=1.0 - self.decay)
                self.codebook_avg.mul_(self.decay).add_(code_sum, alpha=1.0 - self.decay)

                total = self.cluster_size.sum()
                smoothed = (
                    (self.cluster_size + self.eps)
                    / (total + self.codebook_size * self.eps)
                    * total
                )
                updated = self.codebook_avg / smoothed.unsqueeze(1)
                if self.l2_normalize:
                    updated = F.normalize(updated, dim=-1)
                self.codebook.copy_(updated)

                num_restarted = self._restart_dead_codes(flat)

        # The commitment term is the only gradient the encoder gets from the
        # codebook side; the codebook itself is EMA-updated, not learned.
        commitment_loss = F.mse_loss(flat, quantized.detach())

        # Straight-through: identity in the forward pass, identity gradient back.
        z_q = flat + (quantized - flat).detach()
        z_q = z_q.view_as(z_e).to(input_dtype)

        stats = {
            "perplexity": perplexity.detach(),
            "usage": (counts > 0).float().sum().detach(),
            "restarted": torch.tensor(float(num_restarted), device=flat.device),
            "commitment": commitment_loss.detach(),
        }
        return z_q, indices.view(z_e.shape[0], z_e.shape[1]), self.commitment * commitment_loss, stats


@dataclass
class PlanVQConfig:
    num_codes: int = 8
    codebook_size: int = 512
    code_dim: int = 32
    d_model: int = 512
    n_head: int = 8
    perf_layers: int = 2
    encoder_layers: int = 2
    decoder_layers: int = 4
    fourier_bands: int = 32
    dropout: float = 0.1
    commitment: float = 0.25
    ema_decay: float = 0.99
    l2_normalize: bool = True
    pianobart_dim: int = 1024

    def to_dict(self):
        return asdict(self)


class PlanVQ(nn.Module):
    """VQ over frozen PianoBART score features, with a performance-conditioned decoder.

    Trainable: the performance encoder, the cross-attention queries that pool
    the frozen score states into ``num_codes`` latents, the codebook, and the
    decoder. PianoBART is frozen and lives outside this module -- its features
    arrive as plain tensors.
    """

    def __init__(self, config: PlanVQConfig):
        super().__init__()
        self.config = config
        d = config.d_model

        self.score_proj = nn.Linear(config.pianobart_dim, d)
        self.score_norm = nn.LayerNorm(d)
        # Shared note-index embedding: performance note k and score slot k get the
        # same vector, which is what tells the decoder they are partners.
        self.note_index = nn.Embedding(NUM_PERF_NOTES, d)
        self.performance = PerformanceEncoder(config)

        self.encoder_queries = nn.Parameter(torch.randn(config.num_codes, d) * 0.02)
        self.encoder = _cross_attention_stack(
            d, config.n_head, config.encoder_layers, config.dropout
        )
        self.to_code = nn.Linear(d, config.code_dim)

        self.quantizer = VectorQuantizerEMA(
            config.codebook_size,
            config.code_dim,
            decay=config.ema_decay,
            commitment=config.commitment,
            l2_normalize=config.l2_normalize,
        )

        self.from_code = nn.Linear(config.code_dim, d)
        self.code_position = nn.Parameter(torch.randn(config.num_codes, d) * 0.02)
        self.slot_queries = nn.Parameter(torch.randn(BODY_SLOTS, d) * 0.02)
        self.decoder = _transformer(d, config.n_head, config.decoder_layers, config.dropout)

        self.onset_head = nn.Linear(d, MAX_TIME)
        self.duration_head = nn.Linear(d, MAX_DUR)
        self.pitch_head = nn.Linear(d, MAX_PITCH)

    # -- pieces ------------------------------------------------------------

    def encode_performance(self, perf):
        """Raw performance notes (B, 170, 3) -> states (B, 170, d_model)."""
        index = self.note_index.weight[: perf.shape[1]].unsqueeze(0)
        return self.performance(perf, index)

    def _project_score(self, score_features):
        index = self.note_index.weight[: score_features.shape[1]].unsqueeze(0)
        return self.score_norm(self.score_proj(score_features)) + index

    def encode(self, perf_state, score_features, score_valid=None):
        """Pool frozen score features into latents, shape (B, num_codes, code_dim).

        The queries cross-attend over the score states *and* the performance
        states together: knowing the performance is what lets the pooling head
        spend its codes on the residual instead of on what the performance
        already implies.
        """
        batch = score_features.shape[0]
        memory = torch.cat([self._project_score(score_features), perf_state], dim=1)
        padding = None
        if score_valid is not None and not bool(score_valid.all()):
            padding = torch.cat(
                [~score_valid, torch.zeros_like(perf_state[..., 0], dtype=torch.bool)], dim=1
            )
        queries = self.encoder_queries.unsqueeze(0).expand(batch, -1, -1)
        hidden = self.encoder(queries, memory, memory_key_padding_mask=padding)
        return self.to_code(hidden)

    def decode(self, z_q, perf_state):
        """(codes, performance states) -> per-slot onset/duration/pitch logits."""
        batch = perf_state.shape[0]
        codes = self.from_code(z_q) + self.code_position.unsqueeze(0)
        slots = self.slot_queries.unsqueeze(0).expand(batch, -1, -1) + self.note_index.weight[
            :BODY_SLOTS
        ].unsqueeze(0)
        sequence = torch.cat([codes, perf_state, slots], dim=1)
        hidden = self.decoder(sequence)[:, -BODY_SLOTS:]
        return self.onset_head(hidden), self.duration_head(hidden), self.pitch_head(hidden)

    # -- entry points ------------------------------------------------------

    @torch.no_grad()
    def encode_codes(self, perf, score_features, score_valid=None):
        """(performance, score features) -> plan codes (B, num_codes). Frozen use."""
        z_e = self.encode(self.encode_performance(perf), score_features, score_valid)
        _, indices, _, _ = self.quantizer(z_e)
        return indices

    @torch.no_grad()
    def reconstruct(self, perf, score_features, score_valid=None, shuffle_codes=False):
        """-> ``(predicted_score, codes)``; ``predicted_score`` is (B, 138, 3) raw values."""
        perf_state = self.encode_performance(perf)
        z_e = self.encode(perf_state, score_features, score_valid)
        z_q, indices, _, _ = self.quantizer(z_e)
        if shuffle_codes:
            z_q = z_q.roll(shifts=1, dims=0)
        onset_logits, duration_logits, pitch_logits = self.decode(z_q, perf_state)
        predicted = torch.stack(
            [
                onset_logits.argmax(dim=-1),
                duration_logits.argmax(dim=-1),
                pitch_logits.argmax(dim=-1),
            ],
            dim=-1,
        )
        return predicted, indices

    def forward(self, perf, score_features, score, score_valid, shuffle_codes=False):
        perf_state = self.encode_performance(perf)
        z_e = self.encode(perf_state, score_features, score_valid)
        z_q, indices, vq_loss, stats = self.quantizer(z_e)

        if shuffle_codes:
            # Diagnostic: reconstruct from ANOTHER window's plan. The accuracy
            # gap against the true plan is the information the plan carries.
            z_q = z_q.roll(shifts=1, dims=0)

        onset_logits, duration_logits, pitch_logits = self.decode(z_q, perf_state)

        valid = score_valid.reshape(-1)
        losses, accuracies = {}, {}
        total = None
        for name, logits, target in (
            ("onset", onset_logits, score[..., 0]),
            ("duration", duration_logits, score[..., 1]),
            ("pitch", pitch_logits, score[..., 2]),
        ):
            flat_logits = logits.reshape(-1, logits.shape[-1])[valid].float()
            flat_target = target.reshape(-1)[valid]
            loss = F.cross_entropy(flat_logits, flat_target)
            losses[name] = loss
            accuracies[name] = (flat_logits.argmax(dim=-1) == flat_target).float().mean().detach()
            total = loss if total is None else total + loss

        return {
            "loss": total + vq_loss,
            "reconstruction_loss": total.detach(),
            "vq_loss": vq_loss.detach(),
            "losses": {k: v.detach() for k, v in losses.items()},
            "accuracies": accuracies,
            "codes": indices,
            "stats": stats,
        }


# ---------------------------------------------------------------------------
# checkpoint I/O
# ---------------------------------------------------------------------------

def save_plan_vq(model, path):
    torch.save({"config": model.config.to_dict(), "state_dict": model.state_dict()}, path)


def load_plan_vq(path, device="cpu"):
    payload = torch.load(path, map_location=device, weights_only=False)
    model = PlanVQ(PlanVQConfig(**payload["config"]))
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model
