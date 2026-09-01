"""CPU correctness tests for `nbest.speculative` (run: `python -m nbest.test_speculative`).

These are the exactness proofs that do not need a GPU or the real checkpoint:
a tiny randomly-initialised target/draft pair exercises the accept/reject and
residual-resampling paths far harder than the trained pair ever will (the draft
is an *independent* random model, so acceptance is near chance and almost every
token goes through the residual branch).
"""

from __future__ import annotations

import math

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from anticipation.packed_sequence import ALTERNATING_START
from anticipation.vocab import (
    ADUR_OFFSET,
    ANOTE_OFFSET,
    ATIME_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    TIME_OFFSET,
    VOCAB_SIZE,
)
from nbest.draft_ngram import NgramProposer, fit_ngram_tables
from nbest.speculative import (
    ModelProposer,
    StagedProposer,
    build_shallow_draft,
    score_position_list,
    speculative_rollout_score_slots,
)
from onpolicy_rollout import rollout_score_slots

N_SLOTS = 4
LENGTH = ALTERNATING_START + 6 * N_SLOTS


def tiny_model(seed, n_layer=2, logit_scale=1.0):
    """A tiny random GPT-2 over the real vocabulary.

    ``logit_scale`` scales the (tied) embedding table.  At the default init the
    logits of a random model are so flat that its softmax is essentially uniform
    over 55028 tokens, which makes a frequency test powerless -- every sampled
    token is unique.  Scaling the table concentrates the distribution on a
    handful of tokens so the chi-square below actually has something to detect.
    """
    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=VOCAB_SIZE,
        n_positions=LENGTH + 8,
        n_embd=32,
        n_layer=n_layer,
        n_head=2,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        scale_attn_by_inverse_layer_idx=True,
    )
    model = GPT2LMHeadModel(config)
    if logit_scale != 1.0:
        with torch.no_grad():
            model.transformer.wte.weight.mul_(logit_scale)
        model.tie_weights()
    model.eval()
    return model


def perturbed_copy(model, sigma=0.02, seed=11):
    """A draft that is the target plus weight noise -- gives partial acceptance,
    so the *accept* branch of the rule is exercised as hard as the residual one."""
    import copy as _copy

    clone = _copy.deepcopy(model)
    generator = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for name, parameter in clone.named_parameters():
            if "wte" in name or "wpe" in name:
                continue
            parameter.add_(torch.randn(parameter.shape, generator=generator) * sigma)
    clone.tie_weights()
    clone.eval()
    return clone


def fake_window(batch, seed=0):
    """A structurally valid packed window: control/dummy prefix then alternation."""
    rng = torch.Generator().manual_seed(seed)
    tokens = torch.zeros(batch, LENGTH, dtype=torch.long)

    def control(n):
        return torch.stack(
            [
                ATIME_OFFSET + torch.randint(0, 500, (n,), generator=rng),
                ADUR_OFFSET + torch.randint(0, 200, (n,), generator=rng),
                ANOTE_OFFSET + torch.randint(0, 128, (n,), generator=rng),
            ],
            dim=-1,
        )

    def score(n):
        return torch.stack(
            [
                TIME_OFFSET + torch.randint(0, 500, (n,), generator=rng),
                DUR_OFFSET + torch.randint(0, 200, (n,), generator=rng),
                NOTE_OFFSET + torch.randint(0, 128, (n,), generator=rng),
            ],
            dim=-1,
        )

    for b in range(batch):
        pos = 0
        while pos < ALTERNATING_START:
            tokens[b, pos : pos + 3] = control(1)[0]
            tokens[b, pos + 3 : pos + 6] = score(1)[0]
            pos += 6
        while pos < LENGTH:
            tokens[b, pos : pos + 3] = score(1)[0]
            tokens[b, pos + 3 : pos + 6] = control(1)[0]
            pos += 6
    return tokens


def test_shallow_draft_is_function_preserving():
    """A `spaced` 1-layer draft must reproduce the target's LAST block exactly.

    Guards the `scale_attn_by_inverse_layer_idx` fold: without the query
    rescale, a block moved from depth 5 to depth 0 divides its attention logits
    by 1 instead of 6 and the "copied" init is silently a different function.
    """
    target = tiny_model(1, n_layer=6)
    draft = build_shallow_draft(target, 1, strategy="spaced")
    hidden = torch.randn(2, 11, target.config.n_embd)
    with torch.no_grad():
        ref = target.transformer.h[5](hidden)[0]
        got = draft.transformer.h[0](hidden)[0]
    # Only float rounding may differ: folding 1/(j+1) into W_q reassociates the
    # same product, so the tolerance is relative to the block's output scale.
    err = (ref - got).abs().max().item() / ref.abs().max().item()
    assert err < 1e-4, f"rescaled block output differs by {err} (relative)"
    print(f"  shallow-draft block copy: max relative diff {err:.2e}  OK")

    full = build_shallow_draft(target, 6, strategy="spaced")
    ids = fake_window(1)
    with torch.no_grad():
        a = target(ids).logits
        b = full(ids).logits
    err = (a - b).abs().max().item()
    assert err < 1e-4, f"full-depth copy differs by {err}"
    print(f"  full-depth copy: max abs diff {err:.2e}  OK")


def test_greedy_is_bit_identical():
    """T=0: the accept rule reduces to `draft argmax == target argmax`, so the
    speculative rollout must reproduce the baseline token for token."""
    target = tiny_model(1, logit_scale=20.0)
    ids = fake_window(6, seed=3)
    base = rollout_score_slots(
        target, ids, temperature=0.0, collect_gt_ce=False, collect_logprobs=False
    )
    for label, draft in (("independent", tiny_model(2, logit_scale=20.0)),
                         ("perturbed", perturbed_copy(target))):
        for slots in (1, 2, 4):
            spec = speculative_rollout_score_slots(
                target, ModelProposer(draft, temperature=0.0), ids,
                slots_per_block=slots, temperature=0.0,
            )
            same = torch.equal(base["rolled"], spec["rolled"])
            assert same, f"greedy mismatch: draft={label} slots_per_block={slots}"
            stats = spec["stats"].as_dict()
            print(
                f"  greedy draft={label} slots_per_block={slots}: identical, "
                f"accept={stats['acceptance_target']:.3f} "
                f"tok/target-fwd={stats['tokens_per_target_forward']:.3f}  OK"
            )


def test_staged_greedy_is_bit_identical():
    """Staged decoding (T <- D1 <- ngram D2) must still reproduce greedy exactly.

    This is the composition claim: the inner level returns tokens drawn from D1's
    distribution *and* that distribution, so the outer rule is unchanged and D2
    can only affect speed.  At T=0 that means bit-identical output no matter how
    bad D2 is -- and the D2 here is a table fitted to random tokens.
    """
    target = tiny_model(1, logit_scale=20.0)
    d1 = perturbed_copy(target)
    tables = fit_ngram_tables(fake_window(64, seed=21), top_m=8)
    ids = fake_window(6, seed=3)
    base = rollout_score_slots(
        target, ids, temperature=0.0, collect_gt_ce=False, collect_logprobs=False
    )
    for slots, inner in ((1, 3), (2, 3), (2, 6)):
        proposer = StagedProposer(
            d1,
            NgramProposer(tables, temperature=0.0, level="d2"),
            temperature=0.0,
            inner_score_tokens=inner,
        )
        spec = speculative_rollout_score_slots(
            target, proposer, ids, slots_per_block=slots, temperature=0.0
        )
        assert torch.equal(base["rolled"], spec["rolled"]), (
            f"staged greedy mismatch at slots={slots} inner={inner}"
        )
        stats = spec["stats"].as_dict()
        print(
            f"  staged greedy slots={slots} inner={inner}: identical, "
            f"tok/T-fwd={stats['tokens_per_target_forward']:.3f} "
            f"D1 fwd/window={stats.get('forwards_per_window_d1', float('nan')):.1f}  OK"
        )


def test_ngram_only_greedy_is_bit_identical():
    """The launch-free table draft alone must also reproduce greedy exactly."""
    target = tiny_model(1, logit_scale=20.0)
    tables = fit_ngram_tables(fake_window(64, seed=21), top_m=8)
    ids = fake_window(6, seed=3)
    base = rollout_score_slots(
        target, ids, temperature=0.0, collect_gt_ce=False, collect_logprobs=False
    )
    for slots in (1, 4):
        spec = speculative_rollout_score_slots(
            target, NgramProposer(tables, temperature=0.0), ids,
            slots_per_block=slots, temperature=0.0,
        )
        assert torch.equal(base["rolled"], spec["rolled"]), f"ngram greedy mismatch slots={slots}"
        stats = spec["stats"].as_dict()
        print(f"  ngram greedy slots={slots}: identical, "
              f"tok/T-fwd={stats['tokens_per_target_forward']:.3f}  OK")


def test_cache_rollback_covers_rewrites():
    """The KV cache must never resume after a position this block rewrote.

    Regression guard for a bug that was silent on the toy model and cost 15-30%
    of tokens on the real one: a rejected token is replaced *after* the
    verification forward already cached its key/value.  For a rejected onset or
    duration the frontier lands one position later and cropping to
    `frontier - 1` happens to drop the stale entry; a rejected PITCH advances
    the frontier by 4, over the slot's 3 teacher-forced control tokens, so
    cropping to `frontier - 1` keeps the rejected pitch AND the controls that
    attended to it.  This is a structural check, not an output check, because
    whether a stale key flips an argmax depends on the model.
    """
    target = tiny_model(1, logit_scale=20.0)
    draft = tiny_model(2, logit_scale=20.0)  # independent -> rejects ~everything
    ids = fake_window(4, seed=9)
    blocks = []
    speculative_rollout_score_slots(
        target, ModelProposer(draft, temperature=0.0), ids,
        slots_per_block=2, temperature=0.0, debug_snapshots=blocks,
    )
    pitch_rejections = 0
    for block in blocks:
        assert block["safe_len"] <= block["earliest_change"], block
        assert block["safe_len"] <= block["new_base"] - 1, block
        if block["earliest_change"] % 3 == 2 and block["earliest_change"] < block["end"]:
            pitch_rejections += 1
    assert pitch_rejections > 0, "test did not exercise a rejected pitch token"
    print(f"  cache rollback: {len(blocks)} blocks, {pitch_rejections} with a rejected "
          f"pitch, resume point never past a rewrite  OK")


def test_structure_preserved():
    """Controls and prefix must be untouched; every score slot must be filled."""
    target = tiny_model(1, logit_scale=20.0)
    draft = tiny_model(2, logit_scale=20.0)
    ids = fake_window(4, seed=5)
    spec = speculative_rollout_score_slots(
        target, ModelProposer(draft, temperature=1.0), ids, slots_per_block=2, temperature=1.0
    )
    rolled = spec["rolled"]
    score_positions = set(score_position_list(LENGTH))
    others = [p for p in range(LENGTH) if p not in score_positions]
    assert torch.equal(rolled[:, others], ids[:, others]), "non-score tokens were modified"
    stats = spec["stats"].as_dict()
    expected = 4 * len(score_positions)
    assert stats["score_tokens"] == expected, (stats["score_tokens"], expected)
    print(f"  structure: {len(others)} non-score positions untouched, "
          f"{expected} score tokens finalised  OK")


def _empirical(counts):
    total = counts.sum()
    return counts / total


def test_distribution_matches_baseline(partial=False, staged=False):
    """T=1: chi-square that speculative sampling reproduces the baseline law.

    Compared on the joint token frequency at every score position, pooled over
    many independent rollouts of one fixed window.  The draft here is an
    independent random model, so acceptance is low and essentially every token
    is produced by the residual `(p-q)_+` branch -- exactly the branch that a
    subtly wrong implementation gets wrong.
    """
    target = tiny_model(1, logit_scale=20.0)
    draft = perturbed_copy(target) if partial else tiny_model(2, logit_scale=20.0)
    if staged:
        tables = fit_ngram_tables(fake_window(64, seed=21), top_m=8)
        def make_proposer():
            return StagedProposer(
                draft,
                NgramProposer(tables, temperature=1.0, level="d2"),
                temperature=1.0, inner_score_tokens=3,
            )
    else:
        def make_proposer():
            return ModelProposer(draft, temperature=1.0)
    batch = 64
    reps = 24
    ids = fake_window(batch, seed=7)[:1].expand(batch, -1).contiguous()
    positions = score_position_list(LENGTH)

    base_tokens, spec_tokens = [], []
    for r in range(reps):
        torch.manual_seed(1000 + r)
        base = rollout_score_slots(
            target, ids, temperature=1.0, collect_gt_ce=False, collect_logprobs=False
        )
        base_tokens.append(base["rolled"][:, positions])
        torch.manual_seed(5000 + r)
        spec = speculative_rollout_score_slots(
            target, make_proposer(), ids, slots_per_block=2, temperature=1.0
        )
        spec_tokens.append(spec["rolled"][:, positions])
    base_tokens = torch.cat(base_tokens, dim=0)
    spec_tokens = torch.cat(spec_tokens, dim=0)
    n = base_tokens.shape[0]

    worst_p = 1.0
    for i, pos in enumerate(positions):
        a = base_tokens[:, i]
        b = spec_tokens[:, i]
        vocab = torch.cat([a, b]).unique()
        index = {int(v): j for j, v in enumerate(vocab.tolist())}
        ca = torch.zeros(len(vocab))
        cb = torch.zeros(len(vocab))
        for v in a.tolist():
            ca[index[v]] += 1
        for v in b.tolist():
            cb[index[v]] += 1
        # Pool rare categories so the chi-square approximation holds.
        keep = (ca + cb) >= 10
        ca = torch.cat([ca[keep], ca[~keep].sum().reshape(1)])
        cb = torch.cat([cb[keep], cb[~keep].sum().reshape(1)])
        expected = (ca + cb) / 2
        stat = (((ca - expected) ** 2 + (cb - expected) ** 2) / expected.clamp(min=1)).sum().item()
        dof = max(int(keep.sum().item()), 1)
        # Survival function of chi2 via the regularised upper incomplete gamma.
        p = _chi2_sf(stat, dof)
        worst_p = min(worst_p, p)
        print(f"    pos {pos} (role {pos % 3}): n={n} chi2={stat:.1f} dof={dof} p={p:.4f}")
    assert worst_p > 0.001, f"distribution mismatch (min p = {worst_p})"
    print(f"  distribution: min p-value over {len(positions)} positions = {worst_p:.4f}  OK")


def _chi2_sf(stat, dof):
    """P(X > stat) for X ~ chi2(dof), by series/continued fraction (no scipy dep)."""
    a, x = dof / 2.0, stat / 2.0
    if x <= 0:
        return 1.0
    if x < a + 1:
        term = 1.0 / a
        total = term
        n = 1
        while n < 1000:
            term *= x / (a + n)
            total += term
            if abs(term) < abs(total) * 1e-14:
                break
            n += 1
        lower = total * math.exp(-x + a * math.log(x) - math.lgamma(a))
        return max(0.0, 1.0 - lower)
    tiny = 1e-300
    b, c, d, h = x + 1.0 - a, 1.0 / tiny, 1.0 / (x + 1.0 - a), 1.0 / (x + 1.0 - a)
    for i in range(1, 1000):
        an = -i * (i - a)
        b += 2.0
        d = an * d + b
        if abs(d) < tiny:
            d = tiny
        c = b + an / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < 1e-14:
            break
    return h * math.exp(-x + a * math.log(x) - math.lgamma(a))


if __name__ == "__main__":
    torch.set_num_threads(4)
    print("test_shallow_draft_is_function_preserving")
    test_shallow_draft_is_function_preserving()
    print("test_structure_preserved")
    test_structure_preserved()
    print("test_cache_rollback_covers_rewrites")
    test_cache_rollback_covers_rewrites()
    print("test_greedy_is_bit_identical")
    test_greedy_is_bit_identical()
    print("test_ngram_only_greedy_is_bit_identical")
    test_ngram_only_greedy_is_bit_identical()
    print("test_distribution_matches_baseline (independent draft: residual branch)")
    test_distribution_matches_baseline(partial=False)
    print("test_distribution_matches_baseline (perturbed draft: both branches)")
    test_distribution_matches_baseline(partial=True)
    print("test_staged_greedy_is_bit_identical")
    test_staged_greedy_is_bit_identical()
    print("test_distribution_matches_baseline (STAGED: target <- D1 <- ngram D2)")
    test_distribution_matches_baseline(partial=True, staged=True)
    print("\nALL SPECULATIVE TESTS PASSED")
