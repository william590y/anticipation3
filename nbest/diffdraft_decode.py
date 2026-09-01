"""Speculative decoding of the packed score format with a masked-diffusion drafter.

The decode loop is unchanged in what it *emits*: it still produces, for every body
score slot, tokens drawn from the target model under `constrain_score_token_logits`.
What changes is how many target forwards that costs. Instead of 414 sequential
single-token forwards it runs rounds of

    K drafter forwards over a block of B slots      (proposes 3B score tokens)
    1 target forward over the same 6B tokens        (verifies all of them at once)

and the block is laid out so that the 3B control tokens interleaved with the draft
ride along inside the same target forward for free -- they are teacher-forced
ground truth, so they never need verifying, and because the block ends on a
control triplet the target's final logit column is already the next slot's onset
distribution: the classic speculative-decoding "bonus token" costs nothing here.


EXACTNESS
=========

Greedy (temperature 0) -- EXACT for any drafter, no assumptions
---------------------------------------------------------------
Verification accepts the longest prefix of the drafted block on which the
target's own constrained argmax agrees with the draft, then overwrites the first
disagreement with the target's argmax. By induction on position: if every token
before p was emitted by the baseline, the target's state at p is the baseline's
state at p, so the token emitted at p (whether accepted or resampled) is the
baseline's token at p. The drafter's quality affects only how many target
forwards this takes. This is the Medusa / lookahead-decoding block verification
rule and it does not care that the proposal came from a diffusion model.

The one thing that can break bit-identity is *floating point*, not the algorithm:
the baseline reaches position p through a 1-token forward while verification
reaches it through a 96-token chunked forward, and those reduce their matmuls in
different orders. `bench/bench_diffdraft.py --check-exact` measures the resulting
argmax-flip rate directly, and `--oracle-draft` isolates it by drafting with the
baseline's own tokens (acceptance is then 100% by construction and any residual
difference is purely numerical).

Sampled (temperature 1) -- EXACT ONLY UNDER A LEFT-TO-RIGHT SCHEDULE
--------------------------------------------------------------------
Speculative sampling (Leviathan et al. 2023; Chen et al. 2023) needs a proposal
that is autoregressive in the *same* order the target factorises in: it must be
possible to evaluate q(x_i | x_<i) at the token that was actually drawn. A
block-diffusion proposal does not obviously have that -- it draws many positions
at once and in a data-dependent order. Working it out:

  * A denoising step draws every position it commits *independently*, conditioned
    on the current partially-unmasked state. So for a fixed schedule the joint is
        q(x_block) = prod_j prod_{i in G_j} p_theta(x_i | state before step j),
    where G_j is the set committed at step j and `state before step j` is
    determined by G_1..G_{j-1} and their sampled values.
  * If the groups G_j are **data-independent** and **left-to-right contiguous**
    -- G_1 the first n/K positions, G_2 the next n/K, and so on -- then for
    i in G_j the set of positions preceding i is exactly G_1 u ... u G_{j-1} plus
    the earlier members of G_j, and members of G_j are conditionally independent
    given G_1..G_{j-1}. Hence
        q(x_i | x_<i) = p_theta(x_i | state before step j)
    which is the number the denoiser already computed. The proposal is a genuine
    autoregressive distribution in the target's own order, and the ordinary
    token-level accept/reject rule applies unchanged. This is `order="ltr"`.
  * One subtlety worth stating because it looks like a hole and is not: the
    denoiser conditions on the block's *future* control tokens, which the target
    has not consumed yet at position i. That does not break anything. The accept
    test needs q_i to be the distribution x_i was actually drawn from given the
    accepted prefix -- it never requires q_i to be a function of the prefix alone.
    The controls are fixed conditioning data of the window (they are teacher-forced,
    not generated), so they are side information like the prompt, and the proposal
    is free to use them. That is exactly why this drafter can be *better* than the
    target's own 32-note lookahead without costing exactness.
  * With MaskGIT confidence ordering the group membership is a function of the
    sampled values, so q(x_block) is a sum over orders and only a lower bound is
    computable; and with role ordering (all onsets, then all durations) the groups
    are data-independent but NOT prefix-closed -- position i's group predecessors
    are not the positions before i -- so q(x_i | x_<i) is not the computed factor.
    Both are therefore **approximate** and are labelled as such
    (`exact_sampled=False` in the returned stats). The deviation is measured, not
    assumed, by `bench/bench_diffdraft.py --dist-check`.

So the honest summary is: the greedy production path is exact by construction; the
sampled path is exact if and only if you give up confidence-ordered unmasking,
which costs draft quality. Both are implemented here.


BATCHING CAVEAT
===============
One KV cache is shared by the whole rollout batch, so the cache can only be rolled
forward to the *minimum* committed position across rows. Rows that accepted more
do not lose those tokens -- `committed_len` is per row and their extra tokens are
re-fed as clean context inside the next block, where they re-verify trivially --
but the batch does advance in rounds paced by its unluckiest row. Per-sequence
acceptance is therefore reported separately from wall-clock throughput. (The same
caveat is documented in `nbest/speculative.py` for the autoregressive drafter.)
"""

from __future__ import annotations

import torch

from anticipation.packed_sequence import ALTERNATING_START
from anticipation.score_constraints import constrain_score_token_logits
from anticipation.vocab import VOCAB_SIZE
from nbest.diffdraft import (
    MASK_ID,
    N_BODY_SLOTS,
    PACKED_LENGTH,
    block_token_end,
    constrain_by_role,
    denoise_block,
)

BODY_END = block_token_end(N_BODY_SLOTS)  # 1020
LAST_SCORE_POS = block_token_end(N_BODY_SLOTS - 1) + 2  # 1016


def score_positions_in_range(lo: int, hi: int, device=None) -> torch.Tensor:
    """Absolute score-token positions in [lo, hi) -- lo need not be a slot start.

    The frontier after a partial acceptance can land on a control token (the
    verifier commits through `first_mismatch + 1`, and that may be the first token
    of a control triplet), so this cannot assume slot alignment.
    """
    positions = [
        p for p in range(max(lo, ALTERNATING_START), min(hi, BODY_END))
        if (p - ALTERNATING_START) % 6 < 3
    ]
    return torch.tensor(positions, dtype=torch.long, device=device)


class DecodeStats:
    """Currency for the speed table.

    `target_forwards` / `drafter_forwards` are launch-count currency (the decode is
    kernel-launch bound, so this is what the wall clock tracks at small batch);
    `target_tokens` / `drafter_tokens` are FLOP currency (what it tracks once the
    batch is large enough to saturate the GPU). Both are reported because they
    diverge, and because the other agent's compile/CUDA-graph work is about to
    change the constant in front of the first one.
    """

    def __init__(self):
        self.rounds = 0
        self.target_forwards = 0
        self.drafter_forwards = 0
        self.target_tokens = 0
        self.drafter_tokens = 0
        self.accepted_score_tokens = 0
        self.resampled_score_tokens = 0
        self.bonus_tokens = 0
        self.per_round_accept = []

    def as_dict(self, batch, windows_done):
        emitted = self.accepted_score_tokens + self.resampled_score_tokens + self.bonus_tokens
        return {
            "rounds": self.rounds,
            "target_forwards_per_window": self.target_forwards,
            "drafter_forwards_per_window": self.drafter_forwards,
            "target_tokens_per_window": self.target_tokens,
            "drafter_tokens_per_window": self.drafter_tokens,
            "mean_accepted_score_tokens_per_round": (
                sum(self.per_round_accept) / max(len(self.per_round_accept), 1)
            ),
            "accepted_tokens_per_target_forward": emitted / max(self.target_forwards, 1) / batch,
            "emitted_score_tokens_per_row": emitted / max(batch, 1),
        }


@torch.no_grad()
def diffdraft_decode(
    target,
    drafter,
    windows,
    *,
    block_slots=16,
    steps=4,
    order="confidence",
    temperature=0.0,
    generator=None,
    oracle_draft=None,
    drafter_autocast=None,
    stats=None,
):
    """Speculative decode of every body score slot. Returns (committed, stats_dict).

    `windows` is (batch, 1020) of ground-truth packed windows: prefix and control
    triplets are read from it, score triplets are overwritten by the decode.

    `oracle_draft`, if given, is a (batch, 1020) tensor whose score tokens are used
    as the proposal instead of the drafter's. Passing the baseline's own greedy
    rollout makes acceptance 100% by construction, which isolates the pure
    floating-point difference between chunked verification and single-token decode.
    """
    device = windows.device
    batch = windows.shape[0]
    if windows.shape[1] != PACKED_LENGTH:
        raise ValueError(f"expected packed length {PACKED_LENGTH}, got {windows.shape[1]}")
    sampled = temperature is not None and temperature > 0
    if sampled and order not in ("ltr",) and oracle_draft is None:
        exact_sampled = False
    else:
        exact_sampled = True
    stats = stats or DecodeStats()

    committed = windows.clone()
    committed_len = torch.full((batch,), ALTERNATING_START, dtype=torch.long, device=device)

    target.config.use_cache = True
    primed = target(windows[:, :ALTERNATING_START], use_cache=True)
    past = primed.past_key_values
    pending = primed.logits[:, -1, :].float()
    stats.target_forwards += 1
    stats.target_tokens += ALTERNATING_START

    d_cache = None
    d_prefix_len = 0
    dctx = drafter_autocast if drafter_autocast is not None else _null_ctx

    pos = ALTERNATING_START
    while pos <= LAST_SCORE_POS:
        slot = (pos - ALTERNATING_START) // 6
        end = block_token_end(min(slot + block_slots, N_BODY_SLOTS))
        absolute = score_positions_in_range(pos, end, device)
        n_pos = absolute.numel()
        if n_pos == 0:
            break
        local = absolute - pos
        roles = absolute % 3

        # --- propose -------------------------------------------------------
        unknown = absolute.unsqueeze(0) >= committed_len.unsqueeze(1)  # (batch, n)
        if oracle_draft is not None:
            drafted = committed[:, pos:end].clone()
            drafted.scatter_(
                1,
                local.unsqueeze(0).expand(batch, -1),
                torch.where(unknown, oracle_draft[:, absolute], committed[:, absolute]),
            )
            proposal = None
        else:
            with dctx():
                if d_prefix_len < pos:
                    d_cache = drafter.encode_prefix(
                        committed[:, d_prefix_len:pos], d_prefix_len, d_cache
                    )
                    stats.drafter_tokens += pos - d_prefix_len
                    d_prefix_len = pos
                block = committed[:, pos:end].clone()
                block.scatter_(
                    1,
                    local.unsqueeze(0).expand(batch, -1),
                    torch.where(unknown, torch.full_like(unknown, MASK_ID, dtype=torch.long),
                                committed[:, absolute]),
                )
                drafted, info = denoise_block(
                    drafter, block, pos, d_cache, local, steps=steps, order=order,
                    temperature=temperature, generator=generator,
                    collect_proposal=sampled,
                )
            stats.drafter_forwards += info["forwards"]
            stats.drafter_tokens += info["forwards"] * (end - pos)
            proposal = info.get("proposal_logits") if sampled else None

        # --- verify: ONE target forward over the whole block ---------------
        out = target(drafted, past_key_values=past, use_cache=True)
        past = out.past_key_values
        stats.target_forwards += 1
        stats.target_tokens += end - pos
        logits = out.logits

        # dists[:, j] is the distribution over the token at pos+j: the column
        # before it, except for j == 0 which was carried over from the previous
        # forward's last column.
        idx = (local - 1).clamp(min=0)
        score_dists = logits[:, idx, :].float()
        if int(local[0]) == 0:
            score_dists[:, 0, :] = pending
        score_dists = constrain_by_role(score_dists, roles)

        rows = torch.arange(batch, device=device)
        draft_tokens = drafted[:, local]
        # A block can contain no un-committed score position at all -- the last
        # one does whenever every row rejected on the same final token, and the
        # bonus token can produce it mid-window too. `denoise_block` then returns
        # early with no proposal, so the sampled accept test has no q to divide
        # by; but it also has nothing to test, since every position is
        # force-accepted. Fall through to the force-accept branch (job 464077).
        all_known = not bool(unknown.any())
        if not sampled or all_known:
            target_tokens = score_dists.argmax(dim=-1)
            # `~unknown` positions were emitted by the target in an earlier round
            # and are already final; re-verifying them would re-derive the same
            # token through a differently-shaped matmul, so force-accept instead.
            # This is also what guarantees the loop makes progress after a
            # rejection at the very first position of a block.
            accept = (target_tokens == draft_tokens) | ~unknown
            n_accept = accept.float().cumprod(dim=1).sum(dim=1).long()
            clipped = n_accept.clamp(max=n_pos - 1)
            resample_at_bad = target_tokens[rows, clipped]
        else:
            logp = torch.log_softmax(score_dists / temperature, dim=-1)
            logq = proposal.float()
            p_x = logp.gather(2, draft_tokens.unsqueeze(-1)).squeeze(-1)
            q_x = logq.gather(2, draft_tokens.unsqueeze(-1)).squeeze(-1)
            uniform = torch.rand(batch, n_pos, device=device, generator=generator)
            accept = (uniform.log() < (p_x - q_x)) | ~unknown
            n_accept = accept.float().cumprod(dim=1).sum(dim=1).long()
            clipped = n_accept.clamp(max=n_pos - 1)
            # The residual norm(p - q)+ is only ever needed at the one rejected
            # position per row, so slice first: the full (batch, n, 55028)
            # residual would be hundreds of MB for nothing.
            p_sel = logp[rows, clipped].exp()
            residual = (p_sel - logq[rows, clipped].exp()).clamp(min=0)
            total = residual.sum(dim=-1, keepdim=True)
            # A row's residual can legitimately carry no mass: the proposal may
            # dominate the target everywhere at that position, and a row whose
            # `clipped` position was force-accepted never had a proposal there at
            # all. norm(p - q)+ tends to p as q -> 0, and p is also the correct law
            # for a position that was never proposed, so fall back to it rather
            # than handing `multinomial` an all-zero row.
            residual = torch.where(total > 0, residual / total.clamp(min=1e-20), p_sel)
            resample_at_bad = torch.multinomial(residual, 1, generator=generator).squeeze(1)

        full = n_accept >= n_pos
        first_bad = absolute[clipped]
        # The accepted prefix already sits in `drafted`; positions past the first
        # rejection are garbage but are below every row's committed_len and get
        # overwritten by a later round.
        committed[:, pos:end] = drafted
        committed[rows, first_bad] = torch.where(
            full, committed[rows, first_bad], resample_at_bad
        )
        new_len = torch.where(full, torch.full_like(first_bad, end), first_bad + 1)
        # The cache may only advance to the last *accepted drafted* token: the
        # resampled token at `first_bad` was never in the verification forward, so
        # the logits after it are conditioned on the wrong token. Keeping the
        # frontier at `first_bad` leaves `pending` valid and costs one re-fed
        # (force-accepted) token next round.
        frontier = torch.where(full, torch.full_like(first_bad, end), first_bad)

        in_prefix = torch.arange(n_pos, device=device).unsqueeze(0) < n_accept.unsqueeze(1)
        stats.accepted_score_tokens += int((in_prefix & unknown).sum())
        stats.resampled_score_tokens += int((~full).sum())
        stats.per_round_accept.append(float((in_prefix & unknown).sum(dim=1).float().mean()))

        # Free bonus token: rows that accepted the whole block get the next slot's
        # onset out of the same forward's last column at no extra cost.
        if bool(full.any()) and end <= LAST_SCORE_POS:
            bonus_dist = constrain_score_token_logits(logits[:, -1, :].float(), end % 3)
            bonus = (
                bonus_dist.argmax(dim=-1)
                if not sampled
                else torch.multinomial(
                    torch.softmax(bonus_dist / temperature, dim=-1), 1, generator=generator
                ).squeeze(1)
            )
            committed[rows, end] = torch.where(full, bonus, committed[rows, end])
            new_len = torch.where(full, new_len + 1, new_len)
            stats.bonus_tokens += int(full.sum())

        committed_len = torch.maximum(committed_len, new_len)
        stats.rounds += 1

        next_pos = int(frontier.min())
        if next_pos < pos:
            raise RuntimeError(f"decode went backwards at {pos}")
        # Carry the distribution for the first token of the next block out of this
        # forward -- no extra target call is needed to restart. Valid for every
        # row because every row's tokens in [pos, next_pos) are accepted draft
        # tokens, so `drafted[:, :j]` is what `committed[pos:next_pos]` holds.
        j = next_pos - pos
        if j > 0:
            pending = logits[:, j - 1, :].float()
        # The crop is UNCONDITIONAL, and `pending` is not. The verification
        # forward always grew the cache from `pos` to `end`, so it always has to
        # be trimmed back -- including in the j == 0 case, where a row rejected on
        # the block's very first score token and the frontier did not move. Making
        # the crop share the `if j > 0` guard left 6*B stale keys in the cache and
        # every subsequent token was conditioned on them: the emitted sequence
        # stopped being a fixed point of greedy decoding at the *second* score
        # token (`bench/debug_diffdraft.py`, job 464278, "speculative
        # self-consistent 3273/3312, first failure at score index 1"). The oracle
        # control could never catch it, because 100% acceptance makes every crop a
        # no-op.
        past.crop(next_pos)
        pos = next_pos

    result = stats.as_dict(batch, 1)
    result["exact_sampled"] = exact_sampled
    result["temperature"] = temperature
    result["order"] = order
    result["steps"] = steps
    result["block_slots"] = block_slots
    return committed, result


class _null_ctx:
    def __enter__(self):
        return None

    def __exit__(self, *args):
        return False
