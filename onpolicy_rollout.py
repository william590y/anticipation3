"""On-policy rollout machinery shared by the two post-training scripts.

Both `train_onpolicy_distill.py` (dense per-token CE oracle) and `train_grpo.py`
(sequence-level reward = onset + duration + pitch exact-match accuracy) need the
same three things, and they must be *identical* between the two runs for their
curves to be comparable:

1. A batched autoregressive rollout over the packed sequence's body score slots
   with the ground-truth control triplets teacher-forced after each one -- i.e.
   exactly the decode protocol `inference.batched_autoregressive_generate_score`
   (and therefore `train.evaluate_model`'s AR pass) uses, but with temperature
   sampling and per-token bookkeeping.
2. The **autoregressive validation loss**: the cross-entropy of the ground-truth
   score token at each slot *conditioned on the model's own rollout prefix*
   rather than on the ground truth. This is the headline metric for both runs --
   the teacher-forced `val/loss` the checkpoints were selected on says nothing
   about how the model behaves once it is conditioned on its own mistakes.
3. Grad-enabled recomputation of those same quantities over a rolled-out
   sequence (the CE for distillation, the sampled-token log-probs for GRPO).

Rollout bookkeeping is free: at the moment the policy samples the token for
position `p`, `next_logits` already holds the distribution over position `p`
given the on-policy prefix, so both the sampled token's log-prob and the
ground-truth token's CE come straight off it. No extra forward pass is needed
for the reward or for the AR validation loss.

Cost note: a rollout is 3 forward passes per score slot (onset, duration, then
pitch batched together with the three teacher-forced control tokens), i.e. 414
sequential KV-cached forwards for the standard 1020-token / 138-slot window.
Incremental decoding is kernel-launch bound, not compute bound, so rolling many
sequences out in one batch is nearly free -- keep the rollout batch large and
chunk only the (memory-hungry) grad-enabled update passes.
"""

from __future__ import annotations

import contextlib
import csv
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from anticipation.packed_sequence import ALTERNATING_START, iter_score_slot_positions
from anticipation.score_constraints import constrain_score_token_logits
from anticipation.vocab import REST

# Token role within a triplet: 0=onset, 1=duration, 2=pitch.
TOKEN_TYPE_NAMES = ("onset", "duration", "pitch")


def body_score_slot_starts(sequence_length):
    """Positions of the body score triplets that have a full control triplet after them."""
    return [
        pos
        for pos in iter_score_slot_positions(sequence_length, ALTERNATING_START)
        if pos + 5 < sequence_length
    ]


def score_token_positions(sequence_length, device=None):
    """The 3*n_slots token positions the policy generates, as a LongTensor."""
    positions = [
        start + role for start in body_score_slot_starts(sequence_length) for role in range(3)
    ]
    return torch.tensor(positions, dtype=torch.long, device=device)


def score_token_roles(positions):
    """Role (0/1/2) of each score-token position. ALTERNATING_START is a multiple of 3."""
    return positions % 3


_ROLE_MASK_CACHE = {}


def _role_constraint_mask(vocab_size, device):
    """(3, vocab) bool mask: True where the token is illegal for that triplet role.

    Built by asking `constrain_score_token_logits` itself, so the training-time
    policy support can never drift from the decoder's.
    """
    key = (vocab_size, str(device))
    cached = _ROLE_MASK_CACHE.get(key)
    if cached is None:
        probe = torch.zeros(vocab_size, device=device)
        cached = torch.stack(
            [torch.isinf(constrain_score_token_logits(probe, role)) for role in range(3)], dim=0
        )
        _ROLE_MASK_CACHE[key] = cached
    return cached


@torch.no_grad()
def rollout_score_slots(
    model,
    input_ids,
    *,
    targets=None,
    temperature=1.0,
    constrain=True,
    collect_logprobs=False,
    collect_gt_ce=True,
    autocast_ctx=None,
    fast=None,
    decoder=None,
):
    """Autoregressively fill every body score slot from the policy's own samples.

    `input_ids` is a (batch, length) batch of ground-truth packed sequences. The
    prefix and every control triplet are teacher-forced; only the score triplets
    come from the policy. `temperature <= 0` means greedy (the eval protocol).

    `targets` supplies the oracle tokens for `gt_ce` (defaults to `input_ids`);
    they differ only when an augmentation perturbs the model input relative to
    the labels.

    Returns a dict with:
      rolled      (batch, length) long   -- input_ids with score triplets replaced
      positions   (n_score_tokens,) long -- the generated positions
      logprob     (batch, n) float32     -- log pi(sampled token) under the
                                            constrained, temperature-scaled policy
      gt_ce       (batch, n) float32     -- unconstrained CE of the *ground-truth*
                                            token at each position, given the
                                            policy's own prefix
      valid       (batch, n) bool        -- slot holds a real (non-REST) score note

    `fast` opts into `anticipation.fast_decode.rollout_score_slots_fast`, which
    returns the same dict from the same decode protocol but removes the decode's
    per-step Python and launch overhead (static KV cache, LM head applied only
    where it is read, optional CUDA-graph capture of a whole slot). It is OFF by
    default, so every existing caller keeps the exact code path it was measured
    on. Pass `fast=True` for the default fast configuration or a dict of
    `rollout_score_slots_fast` options, e.g. `fast={"cuda_graph": True}`.
    `decoder` is an optional `fast_decode.StaticKVDecoder` to reuse across calls
    -- worth holding onto in a validation loop, since it owns the preallocated
    cache and any captured graph.

    Both fast configurations are verified **bit-identical** to this function on
    greedy rollouts by `bench/check_identical.py`; sampling stays on this path
    (see `fast_decode._SlotGraph` for why a captured graph cannot reproduce a
    seeded `torch.multinomial` sequence token-for-token).
    """
    if fast:
        from anticipation.fast_decode import rollout_score_slots_fast

        options = fast if isinstance(fast, dict) else {}
        return rollout_score_slots_fast(
            model,
            input_ids,
            targets=targets,
            temperature=temperature,
            constrain=constrain,
            collect_logprobs=collect_logprobs,
            collect_gt_ce=collect_gt_ce,
            autocast_ctx=autocast_ctx,
            decoder=decoder,
            **options,
        )

    was_training = model.training
    model.eval()
    ctx = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext

    device = input_ids.device
    length = input_ids.shape[1]
    starts = body_score_slot_starts(length)
    if not starts:
        raise ValueError(f"No body score slots in a sequence of length {length}.")
    if targets is None:
        targets = input_ids

    rolled = input_ids.clone()
    logprob_cols = []
    gt_ce_cols = []
    valid_cols = []

    try:
        with ctx():
            primed = model(input_ids[:, :ALTERNATING_START], use_cache=True)
            past = primed.past_key_values
            next_logits = primed.logits[:, -1, :]

            for start in starts:
                # A slot is "real" iff its ground-truth pitch is not REST. Body
                # slots always are (dummy rests live only in the prefix), but the
                # eval path checks it, so we do too.
                slot_valid = targets[:, start + 2] != REST
                sampled = []
                for role in range(3):
                    logits = next_logits.float()
                    if collect_gt_ce:
                        gt_ce_cols.append(
                            F.cross_entropy(logits, targets[:, start + role], reduction="none")
                        )
                    policy_logits = (
                        constrain_score_token_logits(logits, role) if constrain else logits
                    )
                    if temperature is not None and temperature > 0:
                        policy_logits = policy_logits / temperature
                        token = torch.multinomial(
                            torch.softmax(policy_logits, dim=-1), num_samples=1
                        ).squeeze(1)
                    else:
                        token = policy_logits.argmax(dim=-1)
                    if collect_logprobs:
                        logprob_cols.append(
                            torch.log_softmax(policy_logits, dim=-1)
                            .gather(1, token.unsqueeze(1))
                            .squeeze(1)
                        )
                    valid_cols.append(slot_valid)
                    sampled.append(token)

                    if role < 2:
                        step = model(token.unsqueeze(1), past_key_values=past, use_cache=True)
                        past = step.past_key_values
                        next_logits = step.logits[:, -1, :]

                # The pitch token and the three teacher-forced control tokens that
                # follow it are all known now, so one forward advances the cache
                # through the whole tail of this slot.
                chunk = torch.cat(
                    [sampled[2].unsqueeze(1), input_ids[:, start + 3 : start + 6]], dim=1
                )
                step = model(chunk, past_key_values=past, use_cache=True)
                past = step.past_key_values
                next_logits = step.logits[:, -1, :]

                for role in range(3):
                    rolled[:, start + role] = sampled[role]
    finally:
        model.train(was_training)

    return {
        "rolled": rolled,
        "positions": score_token_positions(length, device=device),
        "logprob": torch.stack(logprob_cols, dim=1) if collect_logprobs else None,
        "gt_ce": torch.stack(gt_ce_cols, dim=1) if collect_gt_ce else None,
        "valid": torch.stack(valid_cols, dim=1),
    }


def score_token_ce(logits, target_ids, positions):
    """Unconstrained CE of `target_ids` at `positions`, from the logits before them.

    A causal LM predicts token `p` from the logits at `p - 1`; `positions` never
    includes 0 (score slots start at ALTERNATING_START).
    """
    selected = logits[:, positions - 1, :].float()
    targets = target_ids[:, positions]
    return F.cross_entropy(selected.transpose(1, 2), targets, reduction="none")


def score_token_logprob(logits, token_ids, positions, *, temperature=1.0, constrain=True):
    """log pi(token_ids[p]) at each p, under the same policy the rollout sampled from."""
    # `selected` is freshly materialised by the gather above and neither the
    # index nor the dtype cast needs its value for backward, so the constraint
    # can be applied in place -- worth it, since this tensor is (batch, 3*slots,
    # vocab) and a spare copy is hundreds of MB.
    selected = logits[:, positions - 1, :].float()
    if constrain:
        mask = _role_constraint_mask(selected.shape[-1], selected.device)
        selected = selected.masked_fill_(
            mask[score_token_roles(positions)].unsqueeze(0), -float("inf")
        )
    if temperature is not None and temperature > 0 and temperature != 1.0:
        selected = selected / temperature
    log_probs = torch.log_softmax(selected, dim=-1)
    return log_probs.gather(2, token_ids[:, positions].unsqueeze(-1)).squeeze(-1)


def per_rollout_accuracy(rolled, targets, positions, valid):
    """Exact-match accuracy per rollout and token role -> (batch, 3).

    A token counts only if it equals the ground-truth token exactly; onsets and
    durations get no tolerance (this is token equality on the 10 ms bin grid, the
    same criterion `train.evaluate_model` reports as `val/ar_*_accuracy`). Only
    real score slots are counted.
    """
    matches = rolled[:, positions] == targets[:, positions]
    roles = score_token_roles(positions)
    columns = []
    for role in range(3):
        selector = valid & (roles == role).unsqueeze(0)
        hits = (matches & selector).sum(dim=1).float()
        total = selector.sum(dim=1).clamp(min=1).float()
        columns.append(hits / total)
    return torch.stack(columns, dim=1)


def sequence_accuracy_reward(accuracy):
    """Sequence-level reward in [0, 3]: onset_acc + duration_acc + pitch_acc.

    `accuracy` is the (batch, 3) tensor from `per_rollout_accuracy`. This is the
    GRPO/CRPO reward -- the same three quantities `evaluate_autoregressive`
    reports as `ar_accuracy` (and train.py as `val/ar_*_accuracy`). Training
    rollouts are sampled (T>0) so the group has spread; validation is greedy
    (T=0). The scoring criterion is identical either way: exact token match,
    no onset/duration bin tolerance.
    """
    return accuracy.sum(dim=1)


def reward_from_role_accuracies(accuracies):
    """Scalar reward in [0, 3] from an {onset, duration, pitch} accuracy dict."""
    return (
        float(accuracies["onset"])
        + float(accuracies["duration"])
        + float(accuracies["pitch"])
    )


def masked_mean(values, mask):
    """Mean of `values` over True entries of `mask` (0.0 if the mask is empty)."""
    total = (values * mask).sum()
    count = mask.sum()
    return total / count if count > 0 else total * 0.0


def per_type_means(values, mask, positions):
    """Mean of `values` by token role, as a plain dict (for step logging)."""
    sums, counts = per_type_sums(values, mask, positions)
    return {
        TOKEN_TYPE_NAMES[i]: (
            float(sums[i].item()) / float(counts[i].item())
            if float(counts[i].item()) > 0
            else float("nan")
        )
        for i in range(3)
    }


def per_type_sums(values, mask, positions):
    """Sum and count of `values` split by token role, for per-type loss reporting."""
    roles = score_token_roles(positions)
    sums, counts = [], []
    for role in range(3):
        selector = mask & (roles == role).unsqueeze(0)
        sums.append((values * selector).sum())
        counts.append(selector.sum())
    return torch.stack(sums), torch.stack(counts)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def fixed_eval_indices(dataset_size, requested, seed=1234):
    """A deterministic validation subset, so the AR-loss curve is comparable over time.

    train.py resamples a fresh random subset at every validation; that is fine for
    a smooth teacher-forced average over 500 windows but it would put visible
    sampling noise on a 64-window autoregressive curve.
    """
    count = dataset_size if requested is None or requested <= 0 else min(requested, dataset_size)
    generator = torch.Generator().manual_seed(seed)
    order = torch.randperm(dataset_size, generator=generator).tolist()
    return sorted(order[:count])


@torch.no_grad()
def evaluate_autoregressive(
    model,
    accelerator,
    dataset,
    indices,
    *,
    batch_size,
    autocast_ctx=None,
    progress=False,
):
    """Teacher-forced and autoregressive validation loss on one fixed subset.

    Both losses are the unconstrained cross-entropy of the ground-truth score
    tokens; they differ only in what the model was conditioned on:

      tf_loss  -- ground-truth prefix (what `val/loss` measures, but restricted to
                  score tokens so it is on the same footing as the AR number)
      ar_loss  -- the model's own greedy rollout prefix

    Their gap is the exposure-bias penalty both training runs are trying to close.
    Also returns the AR onset/duration/pitch accuracies so these runs stay
    comparable to the `val/ar_*_accuracy` series from the original training runs.
    """
    was_training = model.training
    model.eval()
    ctx = autocast_ctx if autocast_ctx is not None else contextlib.nullcontext
    device = accelerator.device

    # Shard whole *batches*, not individual windows, so a window is always rolled
    # out alongside the same seven others no matter how many ranks a run uses.
    # Batched matmuls do not reduce in a fixed order, and a greedy rollout is
    # chaotic -- one flipped argmax on an onset token cascades through the rest of
    # the window -- so item-level sharding made the 4-GPU and 3-GPU runs disagree
    # on the same checkpoint's AR loss (10.41 vs 10.23) purely from batch shape.
    batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
    shard = batches[accelerator.process_index :: accelerator.num_processes]

    ar_ce_sum = torch.zeros(3, dtype=torch.float64, device=device)
    ar_ce_count = torch.zeros(3, dtype=torch.float64, device=device)
    tf_ce_sum = torch.zeros(3, dtype=torch.float64, device=device)
    tf_ce_count = torch.zeros(3, dtype=torch.float64, device=device)
    correct = torch.zeros(3, dtype=torch.float64, device=device)
    slot_total = torch.zeros((), dtype=torch.float64, device=device)

    try:
        for batch_index, chunk in enumerate(shard):
            if not chunk:
                continue
            input_ids = torch.stack([dataset[i]["input_ids"] for i in chunk]).to(device)

            rollout = rollout_score_slots(
                model,
                input_ids,
                temperature=0.0,
                constrain=True,
                collect_logprobs=False,
                collect_gt_ce=True,
                autocast_ctx=autocast_ctx,
            )
            positions = rollout["positions"]
            valid = rollout["valid"]

            sums, counts = per_type_sums(rollout["gt_ce"], valid, positions)
            ar_ce_sum += sums.double()
            ar_ce_count += counts.double()

            with ctx():
                outputs = model(input_ids=input_ids, use_cache=False)
            tf_ce = score_token_ce(outputs.logits, input_ids, positions)
            sums, counts = per_type_sums(tf_ce, valid, positions)
            tf_ce_sum += sums.double()
            tf_ce_count += counts.double()

            # Rollout accuracy, per role, over real slots (matches train.evaluate_model).
            roles = score_token_roles(positions)
            matches = rollout["rolled"][:, positions] == input_ids[:, positions]
            for role in range(3):
                selector = valid & (roles == role).unsqueeze(0)
                correct[role] += (matches & selector).sum().double()
            slot_total += (valid & (roles == 2).unsqueeze(0)).sum().double()

            if progress and accelerator.is_main_process:
                print(
                    f"  AR eval: batch {batch_index + 1}/{len(shard)} (this rank)",
                    flush=True,
                )
    finally:
        model.train(was_training)

    stacked = torch.cat(
        [ar_ce_sum, ar_ce_count, tf_ce_sum, tf_ce_count, correct, slot_total.reshape(1)]
    )
    stacked = accelerator.reduce(stacked, reduction="sum")
    ar_ce_sum, ar_ce_count, tf_ce_sum, tf_ce_count, correct = (
        stacked[0:3], stacked[3:6], stacked[6:9], stacked[9:12], stacked[12:15]
    )
    slot_total = float(stacked[15].item())

    def _mean(sums, counts):
        total_sum = float(sums.sum().item())
        total_count = float(counts.sum().item())
        overall = total_sum / total_count if total_count > 0 else float("nan")
        by_type = {
            TOKEN_TYPE_NAMES[i]: (
                float(sums[i].item()) / float(counts[i].item())
                if float(counts[i].item()) > 0
                else float("nan")
            )
            for i in range(3)
        }
        return overall, by_type, total_count

    ar_loss, ar_by_type, ar_tokens = _mean(ar_ce_sum, ar_ce_count)
    tf_loss, tf_by_type, _ = _mean(tf_ce_sum, tf_ce_count)

    accuracies = {
        TOKEN_TYPE_NAMES[i]: (float(correct[i].item()) / slot_total if slot_total > 0 else float("nan"))
        for i in range(3)
    }

    return {
        "ar_loss": ar_loss,
        "ar_loss_by_type": ar_by_type,
        "tf_loss": tf_loss,
        "tf_loss_by_type": tf_by_type,
        "ar_accuracy": accuracies,
        "reward": reward_from_role_accuracies(accuracies),
        "windows": len(indices),
        "score_tokens": ar_tokens,
        "notes": slot_total,
    }


# ---------------------------------------------------------------------------
# Local (wandb-independent) metric artifacts
# ---------------------------------------------------------------------------


def append_metrics_row(output_dir, row, fieldnames):
    """Append one validation row to <output_dir>/ar_val_loss.csv."""
    path = Path(output_dir) / "ar_val_loss.csv"
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return path


def plot_ar_loss_curve(output_dir, steps, ar_losses, tf_losses, title):
    """Render the AR-vs-teacher-forced validation loss curve to a PNG.

    Written on every validation so the run is inspectable without wandb (the same
    reason train.py renders its heatmaps locally).
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    if not steps:
        return None

    figure, axis = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    axis.plot(steps, ar_losses, marker="o", markersize=3, color="#c2410c",
              label="autoregressive (own rollout)")
    axis.plot(steps, tf_losses, marker="o", markersize=3, color="#1d4ed8",
              label="teacher forced (ground-truth prefix)")
    axis.set_xlabel("training step")
    axis.set_ylabel("validation cross-entropy (score tokens)")
    axis.set_title(title)
    axis.grid(alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False, fontsize=9)
    finite = [v for v in ar_losses + tf_losses if v == v and not math.isinf(v)]
    if finite:
        span = max(finite) - min(finite)
        pad = span * 0.1 if span > 0 else 0.05
        axis.set_ylim(min(finite) - pad, max(finite) + pad)
    figure.tight_layout()

    path = Path(output_dir) / "ar_val_loss.png"
    figure.savefig(path)
    plt.close(figure)
    return path
