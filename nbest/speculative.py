"""Exact speculative decoding for the packed alternating score/performance format.

Standard speculative sampling (Leviathan et al. 2023 / Chen et al. 2023) drafts
`k` tokens with a cheap model and verifies them with one forward of the target,
accepting a prefix under a rejection rule that leaves the sampled distribution
*exactly* equal to the target's. Two things about our format make the textbook
formulation not directly applicable, and one of them is a gift:

1. The draft cannot free-run.  A window is
   ``[32 (control, dummy-rest) pairs] then score/control/score/control/...``
   from ``ALTERNATING_START = 192``.  Only the score triplets are generated; the
   performance CONTROL triplet after each one is teacher-forced.  So a draft that
   ran 12 tokens ahead unaided would be conditioning half of them on tokens it
   invented in place of the real performance.
2. ...but those control triplets are *known in advance* (they come from the
   window, not from the model).  So the draft can be marched through them exactly
   as the target would be, and -- the actual win -- the target's verification
   forward can cover the whole ``score,score,score,ctrl,ctrl,ctrl,...`` block in
   one launch: the 3 control tokens per slot ride along for free and, because the
   block ends on a control triplet, the target's last logit column is the
   distribution of the *next* slot's onset, i.e. the usual free "bonus" token
   falls out of the block structure rather than costing anything.

Cost model (this is what decides whether the technique is worth using here):
`rollout_score_slots` spends exactly one target forward per generated score
token (onset, duration, then pitch batched with the 3 controls => 3 forwards per
slot, 3 score tokens per slot).  So the baseline is **1.0 score tokens per target
forward**, and speculative decoding's only currency is raising that number:

    speedup  =  (tokens per target forward)  /  (1 + sum_levels n_fwd_l * c_l/c_T)

Incremental decode here is kernel-launch bound, not compute bound, so ``c_l/c_T``
is much closer to ``n_layer_l / n_layer_T`` than to the FLOP ratio -- which is
why the draft is a *shallow* copy of the target at full width rather than a
narrow model trained from scratch (see `train_draft.py`), and why a table-based
draft that costs no forward at all (`nbest/draft_ngram.py`) is competitive.

Levels
------
`ModelProposer` is a neural draft; `NgramProposer` (in `nbest/draft_ngram.py`) is
a launch-free table draft; `StagedProposer` is staged / cascade speculative
decoding (Spector & Re, arXiv:2308.04623; Chen et al., arXiv:2312.11462): D1's
own proposals are produced by running speculative sampling of D1 against a
cheaper D2.  **Exactness composes**: the inner level returns tokens distributed
exactly as D1's constrained distribution *and* returns that distribution, so the
outer accept/reject against the target is the ordinary rule with proposal q =
q_D1.  D2's quality therefore affects speed only, never the sampled law.

Batching caveat (measured, see `bench/bench_speculative.py`): one KV cache is
shared by the whole rollout batch, so the cache can only be rolled back to the
*minimum* accepted position across rows.  Rows that accepted more keep their
tokens (they are final and correct -- they are simply re-fed, for free, inside
the next block's chunk), but they cannot run ahead of the block window, so the
batch advances at roughly the pace of its unluckiest row.  Per-sequence
acceptance is therefore reported separately from batched throughput.
"""

from __future__ import annotations

import copy
import math

import torch

from anticipation.packed_sequence import ALTERNATING_START
from anticipation.score_constraints import constrain_score_token_logits
from anticipation.vocab import REST, VOCAB_SIZE
from onpolicy_rollout import body_score_slot_starts, score_token_positions

TOKEN_TYPE_NAMES = ("onset", "duration", "pitch")


# ---------------------------------------------------------------------------
# Draft model construction
# ---------------------------------------------------------------------------


def select_layer_indices(n_target_layers: int, n_draft_layers: int, strategy: str = "spaced"):
    """Which target blocks to seed the draft's blocks from.

    ``spaced`` keeps both endpoints (DistilBERT's recipe): the last target block
    is the one whose output ``ln_f``/``lm_head`` were trained to read, so dropping
    it and asking the tied head to decode a mid-stack representation throws away
    most of the initialisation.  ``first`` is the plain truncation baseline.
    """
    if n_draft_layers > n_target_layers:
        raise ValueError(f"draft cannot have more layers ({n_draft_layers}) than target")
    if strategy == "first":
        return list(range(n_draft_layers))
    if strategy == "spaced":
        if n_draft_layers == 1:
            return [n_target_layers - 1]
        step = (n_target_layers - 1) / (n_draft_layers - 1)
        return [int(round(i * step)) for i in range(n_draft_layers)]
    raise ValueError(f"unknown layer selection strategy: {strategy}")


def build_shallow_draft(target, n_draft_layers: int, strategy: str = "spaced"):
    """A shallow GPT-2 seeded from `target`'s embeddings, ln_f, head and K blocks.

    The width, vocabulary and position table are the target's, so the tied
    ``lm_head`` starts already calibrated -- the only thing the draft has to
    relearn is how to reach a decodable representation in K blocks instead of 24.

    ``scale_attn_by_inverse_layer_idx`` is on in this config family, i.e. every
    block divides its attention logits by ``layer_idx + 1``.  A block copied from
    target depth ``j`` into draft depth ``i`` would silently be rescaled by
    ``(j+1)/(i+1)``; ``layer_idx`` also indexes the KV cache so it cannot just be
    left at ``j``.  Folding ``(i+1)/(j+1)`` into the block's *query* projection
    reproduces the original attention logits exactly, so a `spaced` init is a
    true function-preserving copy of each block.
    """
    from transformers import GPT2LMHeadModel

    config = copy.deepcopy(target.config)
    n_target_layers = config.n_layer
    indices = select_layer_indices(n_target_layers, n_draft_layers, strategy)
    config.n_layer = n_draft_layers
    config.use_cache = True

    draft = GPT2LMHeadModel(config)
    target_sd = target.state_dict()

    new_sd = {}
    for key, value in target_sd.items():
        if key.startswith("transformer.h."):
            continue
        new_sd[key] = value.clone()

    embed_dim = config.n_embd
    for new_layer, old_layer in enumerate(indices):
        prefix_old = f"transformer.h.{old_layer}."
        prefix_new = f"transformer.h.{new_layer}."
        scale = (new_layer + 1) / (old_layer + 1)
        for key, value in target_sd.items():
            if not key.startswith(prefix_old):
                continue
            suffix = key[len(prefix_old) :]
            tensor = value.clone()
            if scale != 1.0 and suffix == "attn.c_attn.weight":
                tensor[:, :embed_dim] = tensor[:, :embed_dim] * scale
            elif scale != 1.0 and suffix == "attn.c_attn.bias":
                tensor[:embed_dim] = tensor[:embed_dim] * scale
            new_sd[prefix_new + suffix] = tensor

    missing, unexpected = draft.load_state_dict(new_sd, strict=False)
    unexpected = [k for k in unexpected]
    missing = [k for k in missing if k != "lm_head.weight"]
    if missing or unexpected:
        raise RuntimeError(f"draft init mismatch: missing={missing} unexpected={unexpected}")
    draft.tie_weights()
    draft.config.use_cache = True
    return draft


def load_draft(path, device=None):
    """Load a trained draft checkpoint (plain HF save_pretrained directory)."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(path), local_files_only=True)
    model.config.use_cache = True
    model.eval()
    if device is not None:
        model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Packed-format geometry
# ---------------------------------------------------------------------------


def score_position_list(length, score_start=ALTERNATING_START):
    """Every position the policy generates, in order (3 per body score slot)."""
    return [start + role for start in body_score_slot_starts(length) for role in range(3)]


def next_generated_position(pos):
    """Position of the score token generated after `pos`.

    Within a triplet the next role is one token later; after the pitch token
    (``pos % 3 == 2``) come the 3 teacher-forced control tokens, so the next
    generated position is 4 later.  ``ALTERNATING_START`` is a multiple of 3, so
    ``pos % 3`` is the triplet role.
    """
    return pos + 1 if pos % 3 != 2 else pos + 4


class PackedGeometry:
    """Cached per-length layout so the decode loop never re-derives offsets."""

    def __init__(self, length):
        starts = body_score_slot_starts(length)
        if not starts:
            raise ValueError(f"No body score slots in a sequence of length {length}.")
        self.length = length
        self.slot_starts = starts
        self.gen_start = starts[0]
        self.score_positions = score_position_list(length)
        self.last_score_pos = self.score_positions[-1]
        self.is_score = [False] * length
        for pos in self.score_positions:
            self.is_score[pos] = True
        self.next_score_at = {}
        nxt = None
        for pos in range(length - 1, -1, -1):
            self.next_score_at[pos] = nxt
            if self.is_score[pos]:
                nxt = pos

    def block_after(self, base, n_score):
        """The next `n_score` score positions at or after `base`, and the
        exclusive end of the region (one past the block's trailing controls)."""
        pos = base if self.is_score[base] else self.next_score_at[base]
        positions = []
        while pos is not None and len(positions) < n_score:
            positions.append(pos)
            pos = self.next_score_at[pos]
        end = min(next_generated_position(positions[-1]), self.length)
        return positions, end


_GEOMETRY_CACHE = {}


def geometry(length):
    geom = _GEOMETRY_CACHE.get(length)
    if geom is None:
        geom = _GEOMETRY_CACHE[length] = PackedGeometry(length)
    return geom


# ---------------------------------------------------------------------------
# Constrained policy distributions (must match the baseline decoder exactly)
# ---------------------------------------------------------------------------


def constrained_probs(logits, role, temperature):
    """The distribution `rollout_score_slots` samples from, as explicit probabilities.

    Baseline order of operations is constrain -> divide by temperature ->
    softmax; ``-inf / T`` is still ``-inf`` so this is the identical support and
    the identical renormalisation.  ``temperature <= 0`` is greedy, represented
    here as a one-hot distribution so that the accept/reject rule below needs no
    special case: ``min(1, p(x)/q(x))`` is then 1 exactly when the draft's argmax
    equals the target's, and the residual ``(p-q)_+`` is the target's argmax.
    """
    constrained = constrain_score_token_logits(logits.float(), role)
    if temperature is None or temperature <= 0:
        probs = torch.zeros_like(constrained)
        probs.scatter_(1, constrained.argmax(dim=-1, keepdim=True), 1.0)
        return probs
    return torch.softmax(constrained / temperature, dim=-1)


def _sample(probs, generator):
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)


def accept_reject(p_probs, q_probs, token, live, generator):
    """One step of the exact speculative-sampling rule, vectorised over the batch.

    Returns ``(final_token, accepted, rejected_now)``.  ``token`` was drawn from
    ``q``; it is kept with probability ``min(1, p(x)/q(x))`` and otherwise
    replaced by a draw from the normalised residual ``(p - q)_+``.  The composite
    law of the returned token is exactly ``p`` -- that identity is the whole
    reason the technique is distribution-preserving, and it holds for *any* q.
    """
    p_x = p_probs.gather(1, token.unsqueeze(1)).squeeze(1)
    q_x = q_probs.gather(1, token.unsqueeze(1)).squeeze(1)
    # accept iff u < p(x)/q(x); the multiply form avoids a divide-by-zero when
    # q(x) underflows to 0 (then the token is rejected, as it should be).
    uniform = torch.rand(token.shape[0], device=token.device, generator=generator)
    accept = uniform * q_x < p_x
    rejected_now = live & ~accept
    final = token
    if bool(rejected_now.any()):
        residual = (p_probs - q_probs).clamp_min(0.0)
        total = residual.sum(dim=-1, keepdim=True)
        # p == q everywhere cannot reject, so this branch is unreachable in exact
        # arithmetic; fall back to p rather than let an underflowed residual
        # (all-zero row) make multinomial sample uniformly.
        residual = torch.where(total > 0, residual / total.clamp_min(1e-30), p_probs)
        resampled = _sample(residual, generator)
        final = torch.where(rejected_now, resampled, token)
    return final, live & accept, rejected_now


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


class SpeculativeStats:
    """Accept/reject bookkeeping, kept on GPU until the rollout is over.

    Forward counts are *row weighted* (``+= batch`` per forward): one forward
    decodes the whole batch, so "tokens per target forward" only compares to the
    baseline's exact 1.0 once the batch dimension is divided back out.
    """

    def __init__(self, device):
        self.device = device
        self.finalized = torch.zeros(3, dtype=torch.float64, device=device)
        self.bonus = torch.zeros(3, dtype=torch.float64, device=device)
        self.proposed = {}
        self.accepted = {}
        self.forward_rows = {}
        self.forwards = {}
        self.blocks = 0
        self.rows = 0

    def count_forward(self, level, batch):
        self.forwards[level] = self.forwards.get(level, 0) + 1
        self.forward_rows[level] = self.forward_rows.get(level, 0) + batch

    def count_verify(self, level, role, live, accepted):
        if level not in self.proposed:
            self.proposed[level] = torch.zeros(3, dtype=torch.float64, device=self.device)
            self.accepted[level] = torch.zeros(3, dtype=torch.float64, device=self.device)
        self.proposed[level][role] += live.sum()
        self.accepted[level][role] += accepted.sum()

    def as_dict(self):
        finalized = self.finalized.tolist()
        target_rows = self.forward_rows.get("target", 0)
        out = {
            "blocks": self.blocks,
            "rows": self.rows,
            "score_tokens": sum(finalized),
            "bonus_tokens": sum(self.bonus.tolist()),
            # Currency of the technique: the baseline is exactly 1.0.
            "tokens_per_target_forward": sum(finalized) / target_rows
            if target_rows
            else float("nan"),
        }
        for level, rows in self.forward_rows.items():
            out[f"forwards_{level}"] = self.forwards[level]
            out[f"forwards_per_target_forward_{level}"] = (
                rows / target_rows if target_rows else float("nan")
            )
            # Per-window forward count: the axis on which this is comparable to a
            # block-proposal drafter that emits many tokens from one forward.
            out[f"forwards_per_window_{level}"] = (
                rows / self.rows if self.rows else float("nan")
            )
        for level in self.proposed:
            proposed = self.proposed[level].tolist()
            accepted = self.accepted[level].tolist()
            out[f"acceptance_{level}"] = (
                sum(accepted) / sum(proposed) if sum(proposed) else float("nan")
            )
            for i, name in enumerate(TOKEN_TYPE_NAMES):
                out[f"acceptance_{level}_{name}"] = (
                    accepted[i] / proposed[i] if proposed[i] else float("nan")
                )
                out[f"proposed_{level}_{name}"] = proposed[i]
        return out


# ---------------------------------------------------------------------------
# Proposers
#
# A proposer fills a list of score positions with tokens and returns, for each,
# the *distribution it drew them from* -- the outer verifier needs the density,
# not just the sample.  Contract:
#   prime(out, upto)        cache covers [0, upto)
#   rollback(length)        cache is cut back to at most [0, length)
#   propose(out, positions, frontier, geom, stats)
#       For each position p in `positions` (ascending, consecutive score
#       positions), write a token into out[:, p] for every row whose
#       frontier[b] <= p, leave rows with frontier[b] > p untouched (their token
#       is already final -- but still feed it, so the cache stays consistent with
#       that row's real prefix), and return {p: (batch, vocab) probabilities}.
# ---------------------------------------------------------------------------


class ModelProposer:
    """A neural draft: one forward per proposed score token.

    The trailing control triplet after a pitch token is folded into the *next*
    flush, so a slot costs 3 forwards, not 4 -- same shape as the baseline's own
    per-slot schedule.
    """

    level = "draft"

    def __init__(self, model, temperature=1.0, generator=None, level=None):
        self.model = model
        self.temperature = temperature
        self.generator = generator
        self.cache = None
        self.cache_len = 0
        if level is not None:
            self.level = level

    def prime(self, out, upto, stats):
        # Trunk only: the prime's logits are never read (the first block's flush
        # recomputes the distribution for `upto`), and a full logits tensor over
        # the 191-token prefix is hundreds of MB at a large batch.
        primed = self.model.transformer(out[:, :upto], use_cache=True)
        self.cache = primed.past_key_values
        self.cache_len = upto
        stats.count_forward(self.level, out.shape[0])

    def rollback(self, length):
        if self.cache_len > length:
            self.cache.crop(length)
            self.cache_len = length

    def propose(self, out, positions, frontier, geom, stats):
        batch = out.shape[0]
        probs_by_pos = {}
        pending = [out[:, self.cache_len : positions[0]]]
        for pos in positions:
            chunk = torch.cat([t for t in pending if t.shape[1] > 0], dim=1)
            step = self.model(chunk, past_key_values=self.cache, use_cache=True)
            self.cache = step.past_key_values
            self.cache_len += chunk.shape[1]
            stats.count_forward(self.level, batch)
            probs = constrained_probs(step.logits[:, -1, :], pos % 3, self.temperature)
            token = _sample(probs, self.generator)
            token = torch.where(frontier > pos, out[:, pos], token)
            out[:, pos] = token
            probs_by_pos[pos] = probs
            nxt = geom.next_score_at[pos]
            stop = min(nxt if nxt is not None else geom.length, geom.length)
            pending = [token.unsqueeze(1), out[:, pos + 1 : stop]]
        return probs_by_pos


class StagedProposer:
    """Staged speculative decoding: D1's proposals are drafted by a cheaper D2.

    Exactness of the *outer* level is untouched by D2.  The inner loop is the
    same rejection rule with (verifier, proposer) = (D1, D2), so every token it
    returns is distributed exactly as D1's constrained distribution, and it also
    returns that distribution (D1's verification forward computes it anyway).
    The outer level therefore runs the ordinary rule with proposal ``q = q_D1``.
    D2 only changes how many D1 forwards were needed to get there.
    """

    level = "d1"

    def __init__(self, model, inner, temperature=1.0, generator=None,
                 inner_score_tokens=3, level=None, inner_level=None):
        self.model = model
        self.inner = inner
        self.temperature = temperature
        self.generator = generator
        self.inner_score_tokens = int(inner_score_tokens)
        self.cache = None
        self.cache_len = 0
        if level is not None:
            self.level = level
        if inner_level is not None:
            self.inner.level = inner_level

    def prime(self, out, upto, stats):
        # Trunk only: the prime's logits are never read (the first block's flush
        # recomputes the distribution for `upto`), and a full logits tensor over
        # the 191-token prefix is hundreds of MB at a large batch.
        primed = self.model.transformer(out[:, :upto], use_cache=True)
        self.cache = primed.past_key_values
        self.cache_len = upto
        stats.count_forward(self.level, out.shape[0])
        self.inner.prime(out, upto, stats)

    def rollback(self, length):
        if self.cache_len > length:
            self.cache.crop(length)
            self.cache_len = length
        self.inner.rollback(length)

    def propose(self, out, positions, frontier, geom, stats):
        batch = out.shape[0]
        device = out.device
        m = len(positions)
        pos_tensor = torch.tensor(positions, dtype=torch.long, device=device)
        probs_by_pos = {
            p: torch.zeros(batch, VOCAB_SIZE, device=device, dtype=torch.float32)
            for p in positions
        }
        # `front[b]` is the next of `positions` that row b still needs; rows the
        # OUTER level already carried past a position enter with front beyond it.
        front = frontier.clone()

        while True:
            done = (pos_tensor.unsqueeze(0) < front.unsqueeze(1)).sum(dim=1)
            base_idx = int(done.min())
            if base_idx >= m:
                break
            start_pos = positions[base_idx]
            sub = positions[base_idx : base_idx + self.inner_score_tokens]
            end = min(next_generated_position(sub[-1]), geom.length)

            self.rollback(start_pos - 1)
            resume = self.cache_len
            inner_probs = self.inner.propose(out, sub, front, geom, stats)

            step = self.model(out[:, resume:end], past_key_values=self.cache, use_cache=True)
            self.cache = step.past_key_values
            self.cache_len = end
            stats.count_forward(self.level, batch)
            logits = step.logits

            rejected = torch.zeros(batch, dtype=torch.bool, device=device)
            # Same stale-key/value trap as the outer level (see the rollback at
            # the end of `speculative_rollout_score_slots`): a rejected pitch
            # advances the frontier past 3 control tokens, so rolling back only
            # to the frontier would leave D1 conditioning on a token it rejected.
            earliest_change = end
            for pos in sub:
                role = pos % 3
                p_probs = constrained_probs(
                    logits[:, pos - resume - 1, :], role, self.temperature
                )
                live = (front == pos) & ~rejected
                token, accepted, rejected_now = accept_reject(
                    p_probs, inner_probs[pos], out[:, pos], live, self.generator
                )
                out[:, pos] = torch.where(live, token, out[:, pos])
                probs_by_pos[pos] = torch.where(live.unsqueeze(1), p_probs, probs_by_pos[pos])
                stats.count_verify(self.level, role, live, accepted)
                if bool(rejected_now.any()):
                    earliest_change = min(earliest_change, pos)
                rejected = rejected | rejected_now
                front = torch.where(
                    live, torch.full_like(front, next_generated_position(pos)), front
                )

            # Free bonus token: D1's last logit column is the distribution of the
            # position just past the inner block, and that position is the next
            # one the outer level wants (unless the block ended the request).
            if end in probs_by_pos:
                live = (front == end) & ~rejected
                if bool(live.any()):
                    p_probs = constrained_probs(logits[:, -1, :], end % 3, self.temperature)
                    sampled = _sample(p_probs, self.generator)
                    out[:, end] = torch.where(live, sampled, out[:, end])
                    probs_by_pos[end] = torch.where(
                        live.unsqueeze(1), p_probs, probs_by_pos[end]
                    )
                    front = torch.where(
                        live, torch.full_like(front, next_generated_position(end)), front
                    )

            if earliest_change < self.cache_len:
                self.cache.crop(earliest_change)
                self.cache_len = earliest_change
                self.inner.rollback(earliest_change)
        return probs_by_pos


# ---------------------------------------------------------------------------
# The rollout
# ---------------------------------------------------------------------------


@torch.no_grad()
def speculative_rollout_score_slots(
    target,
    proposer,
    input_ids,
    *,
    slots_per_block=2,
    temperature=1.0,
    constrain=True,
    generator=None,
    stats=None,
    debug_greedy_check=None,
    debug_snapshots=None,
):
    """Fill every body score slot, sampling *exactly* the target's distribution.

    Drop-in for `onpolicy_rollout.rollout_score_slots` at the level of the
    ``rolled`` / ``positions`` / ``valid`` keys (the CE/log-prob bookkeeping is
    deliberately not reproduced: those quantities are for training, and computing
    them would defeat the point of the speculation).

    ``slots_per_block`` sets the lookahead: the proposer produces
    ``3 * slots_per_block`` score tokens (marching through the known control
    triplets in between), then one target forward verifies them all.

    Per-row bookkeeping
    -------------------
    ``frontier[b]`` is the next position row ``b`` still has to generate.  The KV
    caches are shared by the batch, so they are only ever rolled back to
    ``base = min(frontier)``; a row that got further keeps its tokens (they are
    final -- re-feeding them in the next chunk costs nothing extra because the
    chunk is a single rectangular forward either way) and simply skips the
    accept/reject test at positions it has already passed.
    """
    if not constrain:
        raise ValueError(
            "speculative decoding is only implemented for the constrained decoder "
            "(the constrained renormalised distribution is the one being matched)"
        )

    device = input_ids.device
    batch, length = input_ids.shape
    geom = geometry(length)
    gen_start = geom.gen_start
    last_score_pos = geom.last_score_pos

    if stats is None:
        stats = SpeculativeStats(device)
    stats.rows += batch

    was_training = target.training
    target.eval()

    out = input_ids.clone()
    frontier = torch.full((batch,), gen_start, dtype=torch.long, device=device)
    n_score = 3 * int(slots_per_block)

    try:
        # BOTH the target's cache and the proposer's are deliberately kept one
        # token behind the frontier.
        #
        # For the target this is what makes the scheme exact: a rejected token is
        # replaced *after* the verification forward ran, so every logit column of
        # that forward at or beyond the rejection point was conditioned on a
        # token that no longer exists.  Carrying the last column forward as "the
        # distribution for the next position" therefore silently conditions the
        # next block on the rejected draft token (this was a real bug: greedy
        # rollouts then diverged from the baseline at the second token of every
        # block).  Re-feeding position `base - 1` inside the next chunk costs
        # nothing -- it is one more column of a forward we were doing anyway --
        # and every distribution is then conditioned on final tokens only.
        #
        # For the proposer it also guarantees each block's first flush has >= 1
        # token, so no block ever needs an extra "catch-up" forward.
        primed = target(out[:, :gen_start], use_cache=True)
        target_cache = primed.past_key_values
        target_cache.crop(gen_start - 1)
        target_len = gen_start - 1
        stats.count_forward("target", batch)
        proposer.prime(out, gen_start - 1, stats)

        base = gen_start
        while base <= last_score_pos:
            block_positions, end = geom.block_after(base, n_score)

            proposer.rollback(target_len)
            draft_probs = proposer.propose(out, block_positions, frontier, geom, stats)

            # The chunk resumes wherever the cache was rolled back to, so column
            # `j` of the logits is the distribution for position
            # `target_len + j + 1`.
            record = None
            if debug_snapshots is not None:
                record = {
                    "base": base, "end": end, "resume": target_len,
                    "positions": tuple(block_positions),
                    "frontier": frontier.cpu().clone(), "out": out.cpu().clone(),
                }
                debug_snapshots.append(record)
            step = target(out[:, target_len:end], past_key_values=target_cache, use_cache=True)
            target_cache = step.past_key_values
            stats.count_forward("target", batch)
            chunk_logits = step.logits

            rejected = torch.zeros(batch, dtype=torch.bool, device=device)
            # Earliest position this block rewrote.  Every cached key/value at or
            # after it was computed from a token that no longer exists, so the
            # cache has to be rolled back at least this far -- see the rollback
            # below for why `min(frontier) - 1` is NOT far enough.
            earliest_change = end
            for pos in block_positions:
                role = pos % 3
                p_probs = constrained_probs(
                    chunk_logits[:, pos - target_len - 1, :], role, temperature
                )
                live = (frontier == pos) & ~rejected
                token, accepted, rejected_now = accept_reject(
                    p_probs, draft_probs[pos], out[:, pos], live, generator
                )
                out[:, pos] = torch.where(live, token, out[:, pos])
                if debug_greedy_check is not None:
                    # At T=0 the rule collapses to "emit the target's argmax",
                    # so any live row that ends up with something else means the
                    # accept/reject (not the conditioning) is at fault.
                    bad = live & (out[:, pos] != p_probs.argmax(dim=-1))
                    if bool(bad.any()):
                        debug_greedy_check.append(
                            ("verify", pos, base, end, bad.nonzero().flatten().tolist())
                        )
                stats.count_verify("target", role, live, accepted)
                stats.finalized[role] += live.sum()
                if bool(rejected_now.any()):
                    earliest_change = min(earliest_change, pos)
                rejected = rejected | rejected_now
                frontier = torch.where(
                    live, torch.full_like(frontier, next_generated_position(pos)), frontier
                )

            # Free bonus token: the block ends on the control triplet, so the
            # target's final logit column already is the next score token's
            # distribution.  Rows that accepted the whole block get it for free.
            if end <= last_score_pos and geom.is_score[end]:
                live = (frontier == end) & ~rejected
                if bool(live.any()):
                    p_probs = constrained_probs(chunk_logits[:, -1, :], end % 3, temperature)
                    sampled = _sample(p_probs, generator)
                    out[:, end] = torch.where(live, sampled, out[:, end])
                    if debug_greedy_check is not None:
                        bad = live & (out[:, end] != p_probs.argmax(dim=-1))
                        if bool(bad.any()):
                            debug_greedy_check.append(
                                ("bonus", end, base, end, bad.nonzero().flatten().tolist())
                            )
                    stats.bonus[end % 3] += live.sum()
                    stats.finalized[end % 3] += live.sum()
                    frontier = torch.where(
                        live, torch.full_like(frontier, next_generated_position(end)), frontier
                    )

            # Nothing after the last score triplet needs generating.
            frontier = torch.where(
                frontier > last_score_pos, torch.full_like(frontier, length), frontier
            )

            stats.blocks += 1
            new_base = int(torch.clamp(frontier.min(), max=end).item())
            if new_base <= base:
                raise RuntimeError(f"speculative loop failed to advance at {base}")

            # Rolling the cache back to `min(frontier) - 1` is NOT enough, and
            # getting this wrong is silent: a rejected token is replaced *after*
            # the verification forward already wrote its key/value into the
            # shared cache.  For a rejected onset or duration the frontier lands
            # one position later, so cropping to `frontier - 1` happens to drop
            # the stale entry -- but a rejected PITCH advances the frontier by 4
            # (over the slot's 3 teacher-forced control tokens), so cropping to
            # `frontier - 1` keeps the rejected pitch's key/value AND the three
            # controls that attended to it.  Every later logit is then computed
            # against a prefix that no longer exists; greedy rollouts stopped
            # matching the baseline at exactly those slots (measured: 71-85% of
            # tokens, with 3-15 nat logit gaps -- far too large to blame on
            # floating point).  Roll back to the earliest rewritten position.
            safe_len = min(new_base - 1, earliest_change)
            if record is not None:
                record.update(earliest_change=earliest_change, new_base=new_base,
                              safe_len=safe_len)
            target_cache.crop(safe_len)
            target_len = safe_len
            base = new_base
    finally:
        target.train(was_training)

    positions = score_token_positions(length, device=device)
    # `valid` mirrors rollout_score_slots: a slot counts iff its ground-truth
    # pitch is not REST, repeated across the slot's three roles.
    pitch_positions = torch.tensor(
        [start + 2 for start in geom.slot_starts for _ in range(3)],
        dtype=torch.long,
        device=device,
    )
    valid = input_ids[:, pitch_positions] != REST

    return {
        "rolled": out,
        "positions": positions,
        "valid": valid,
        "stats": stats,
        "logprob": None,
        "gt_ce": None,
    }


# ---------------------------------------------------------------------------
# Analytic cost model (kept here so results stay interpretable if the target
# step later gets cheaper -- e.g. under CUDA graphs / TensorRT)
# ---------------------------------------------------------------------------


def predicted_speedup(tokens_per_target_forward, level_forwards, cost_ratios):
    """Speedup over the 1-token-per-target-forward baseline, in target-step units.

    ``level_forwards`` maps a level name to its forwards per target forward (the
    ``forwards_per_target_forward_*`` stats), ``cost_ratios`` maps the same names
    to (one forward of that level) / (one target forward) *measured on the same
    hardware and batch size*.  If the target step is later made ``s`` times
    cheaper, re-evaluate with every ratio multiplied by ``s``.
    """
    overhead = sum(level_forwards.get(k, 0.0) * cost_ratios.get(k, 0.0) for k in cost_ratios)
    return tokens_per_target_forward / (1.0 + overhead)


def crossover_cost_ratio(tokens_per_target_forward, draft_forwards_per_target_forward):
    """The draft/target per-forward cost ratio at which speculation breaks even."""
    if draft_forwards_per_target_forward <= 0:
        return math.inf
    return (tokens_per_target_forward - 1.0) / draft_forwards_per_target_forward
