#!/usr/bin/env python
"""PPO on unfiltered windows with an incremental-F1 reward and a reference KL.

Differences from `train_ppo.py` (the +1/-1 exact-token-match arm), all deliberate:

  data      `data/{train,val}_paper_unfiltered.txt` -- the same windows, but
            conditioned on every performance note the pianist played, mistakes
            included, so score slot k no longer sits at a fixed offset from its
            aligned control. That is the input the results table scores.
  action    ONE TRIPLET, not one token. log pi(action) is the joint
            log-probability of (onset, duration, pitch), which is just the sum of
            the three token log-probabilities since they are generated
            autoregressively. V(s_t) is read at the state before the triplet is
            emitted. 138 actions per rollout rather than 414.
  reward    r_s = F1_s - F1_{s-1} for the triplet the action just emitted, so the
            undiscounted return of a rollout is exactly its F1. The criterion is
            the results table's third column (`onset_pitch_tol1`): pitch exact,
            onset within +-1 bin, duration ignored -- verified to reproduce that
            column's numbers on the same rollouts.
  gamma/lam 0 and 0, as in the token-level +1/-1 arm. Because the action is the
            whole triplet, its F1 delta is a genuine ONE-STEP reward, so
            A_t = r_t - V(s_t) and the critic regresses an immediate target. That
            is the setup whose critic reached EVgae ~0.32 by step 40; at
            token-level actions the same reward forced gamma>0 (the onset and
            duration tokens earn nothing directly) and the critic sat at ~0.
  KL        r_t -= kl_coef * (log pi - log pi_ref) per token against a frozen copy
            of the initial checkpoint, in the reward rather than the loss, so the
            critic prices the drift and the advantage sees it.
  regulariser  weight decay 0.01, matching the supervised run. Dropout stays OFF
            (probe_dropout_ratio.py measured what turning it on would cost: ~4% of
            score tokens clipped by dropout noise alone, p99 |log r| ~1.4).
  noise     onset/duration jitter restored to the supervised defaults (0.05/0.05).
            Score masking is NOT restored -- `train.py`'s own default is 0.00, so
            the supervised run never used it either.

Also carries a log-ratio clamp the earlier arm lacked. PPO's `max(unclipped,
clipped)` is one-sided: for a negative advantage it *selects* the exploding term,
so a single large log-ratio can produce a loss of ~1e7. Dropout's tail (p99
|log r| ~ 1.4, max ~7 before any update) makes that reachable here.
"""
from __future__ import annotations

import contextlib
import json
import time

import torch
from torch.optim import AdamW

import posttrain_common
from f1_reward import IncrementalF1, score_triplet_to_note
from onpolicy_rollout import (
    body_score_slot_starts,
    rollout_score_slots,
    score_token_logprob,
    score_token_positions,
)
from ppo import (
    ValueHead,
    compute_gae,
    discounted_returns,
    explained_variance,
    policy_loss,
    value_loss,
    whiten,
)


def parse_args():
    # --rollout_temperature, --prompts_per_step, --micro_batch and the
    # augmentation knobs already live in the shared parser.
    parser = posttrain_common.build_parser(__doc__)
    parser.add_argument("--ppo_epochs", type=int, default=1)
    # With the triplet as the action the F1 delta is an immediate reward, so
    # A_t = r_t - V(s_t) and no bootstrapping is needed -- the ppo_corrected setup.
    parser.add_argument("--gamma", type=float, default=0.0)
    parser.add_argument("--gae_lambda", type=float, default=0.0)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--value_clip_eps", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--value_lr", type=float, default=3e-5)
    parser.add_argument("--value_head_hidden", type=int, default=256)
    parser.add_argument("--value_warmup_steps", type=int, default=10)
    parser.add_argument("--target_kl", type=float, default=0.02)
    parser.add_argument("--advantage_clip", type=float, default=5.0)
    parser.add_argument("--log_ratio_clip", type=float, default=5.0,
                        help="Clamp |log pi - log pi_old| before exponentiating. "
                             "0 disables. Guards the one-sided surrogate.")
    parser.add_argument("--kl_coef", type=float, default=0.05,
                        help="Per-token KL-to-reference penalty added to the reward.")
    parser.add_argument("--reward_scale", type=float, default=100.0,
                        help="Scales the F1 delta. A single matched note is worth "
                             "~2/(138+n_gt) ~ 0.007 raw; scaling keeps the reward "
                             "and the KL penalty on comparable footing.")
    parser.add_argument("--save_best_val", action="store_true", default=True)
    # Dropout stays OFF (as in the other post-training arms), so pi_theta equals
    # the policy the rollout sampled from and the first-epoch ratio is exactly 1.
    # probe_dropout_ratio.py measured the cost of turning it on: ~4% of score
    # tokens clipped by dropout noise alone, with a p99 |log r| of ~1.4.
    parser.add_argument("--dropout_in_update", action="store_true", default=False)
    return parser.parse_args()


def gt_notes_from_batch(labels, slot_starts):
    """Ground-truth score notes per window, decoded from the label sequence."""
    out = []
    for row in labels:
        notes = []
        for start in slot_starts:
            note = score_triplet_to_note(row[start], row[start + 1], row[start + 2])
            if note is not None:
                notes.append(note)
        out.append(notes)
    return out


def per_triplet(values, reduction="sum"):
    """Fold a token-level (batch, 3*slots) tensor to one entry per triplet.

    Log-probabilities SUM because the triplet is generated autoregressively:
    log pi(triplet) = log pi(onset) + log pi(dur | onset) + log pi(pitch | onset, dur),
    which is the exact joint log-probability of the composite action. Per-token KL
    sums for the same reason. The validity mask repeats a slot's flag across its
    three roles, so it is taken rather than summed.
    """
    batch, n = values.shape
    grouped = values.view(batch, n // 3, 3)
    return grouped[:, :, 0] if reduction == "first" else grouped.sum(dim=-1)


def f1_triplet_rewards(rolled, labels, valid_slots, slot_starts, scale):
    """One immediate reward per triplet action: the F1 delta that triplet caused.

    With the triplet as the action this is a genuine one-step reward, so gamma=0
    gives A_t = r_t - V(s_t) exactly as in the token-level +1/-1 arm.

    Returns (rewards (batch, slots), achieved_f1 (batch,)).
    """
    device = rolled.device
    batch = rolled.shape[0]
    rewards = torch.zeros((batch, len(slot_starts)), dtype=torch.float32, device=device)
    gt_per_window = gt_notes_from_batch(labels.tolist(), slot_starts)
    rolled_cpu = rolled.tolist()
    achieved = []
    for b in range(batch):
        matcher = IncrementalF1(gt_per_window[b])
        for slot, start in enumerate(slot_starts):
            note = score_triplet_to_note(
                rolled_cpu[b][start], rolled_cpu[b][start + 1], rolled_cpu[b][start + 2]
            )
            rewards[b, slot] = matcher.add(note) * scale
        achieved.append(matcher.f1)
    return rewards * valid_slots, torch.tensor(
        achieved, dtype=torch.float32, device=device
    )


@torch.no_grad()
def validate_f1(trainer, args, label=None):
    """Greedy rollouts on a fixed unfiltered validation subset, scored by F1.

    Batch-level sharding (not item-level) so runs on different GPU counts agree
    bit for bit -- the same fix `evaluate_autoregressive` carries.
    """
    accelerator = trainer.accelerator
    model = trainer.base_model
    was_training = model.training
    model.eval()

    indices = list(trainer.eval_indices)
    batch_size = max(1, args.val_batch_size)
    batches = [indices[i : i + batch_size] for i in range(0, len(indices), batch_size)]
    shard = batches[accelerator.process_index :: accelerator.num_processes]

    slot_starts = body_score_slot_starts(trainer.val_dataset[0]["input_ids"].shape[0])
    # The reward's criterion plus the other two table columns, off the same
    # greedy rollouts. Duration earns no reward now, so onset_pitch_dur is the
    # canary for the model buying onset accuracy with note lengths.
    criteria = {
        "onset_pitch_tol1": dict(tolerance=1, require_duration=False),   # the reward
        "onset_pitch": dict(tolerance=0, require_duration=False),
        "onset_pitch_dur": dict(tolerance=0, require_duration=True),
    }
    sums = {
        name: torch.zeros((), dtype=torch.float64, device=accelerator.device)
        for name in criteria
    }
    count = torch.zeros((), dtype=torch.float64, device=accelerator.device)

    for group in shard:
        input_ids = torch.stack(
            [trainer.val_dataset[i]["input_ids"] for i in group]
        ).to(accelerator.device)
        labels = torch.stack(
            [trainer.val_dataset[i]["labels"] for i in group]
        ).to(accelerator.device)
        rollout = rollout_score_slots(
            model, input_ids, targets=labels, temperature=0.0, constrain=True,
            collect_logprobs=False, collect_gt_ce=False,
            autocast_ctx=trainer.autocast,
        )
        rolled = rollout["rolled"].tolist()
        gt_per_window = gt_notes_from_batch(labels.tolist(), slot_starts)
        for b in range(len(group)):
            predicted = [
                score_triplet_to_note(
                    rolled[b][start], rolled[b][start + 1], rolled[b][start + 2]
                )
                for start in slot_starts
            ]
            for name, settings in criteria.items():
                matcher = IncrementalF1(gt_per_window[b], **settings)
                for note in predicted:
                    matcher.add(note)
                sums[name] += matcher.f1
            count += 1

    count = accelerator.reduce(count, reduction="sum")
    results = {
        name: float(
            (accelerator.reduce(total, reduction="sum") / count.clamp(min=1)).item()
        )
        for name, total in sums.items()
    }
    if was_training:
        model.train()
    trainer.log(
        f"[validate{'' if label is None else ' ' + label}] "
        f"step {trainer.completed_steps} | unfiltered greedy REWARD (col 3) "
        f"{100 * results['onset_pitch_tol1']:.3f}% | onset+pitch "
        f"{100 * results['onset_pitch']:.3f}% | +duration "
        f"{100 * results['onset_pitch_dur']:.3f}% "
        f"({int(count.item())} windows)"
    )
    return results


def main():
    args = parse_args()
    if args.rollout_temperature <= 0:
        raise SystemExit("--rollout_temperature must be > 0: PPO needs stochastic actions.")

    trainer = posttrain_common.PostTrainer(args, method="PPO-F1")
    accelerator = trainer.accelerator

    # Dropout was on for the supervised run (config pdrop 0.1, train.py never
    # calls eval()); posttrain_common turns it off for ratio hygiene. Restore it
    # for the UPDATE pass only -- the behaviour policy is scored clean, so
    # pi_old is well defined.
    trainer.log(
        "Dropout in the update pass: "
        + ("ON" if args.dropout_in_update else "OFF (ratio is exactly 1 on epoch 1)")
    )

    hidden_size = trainer.base_model.config.n_embd
    value_head = ValueHead(hidden_size, args.value_head_hidden)
    value_optimizer = AdamW(value_head.parameters(), lr=args.value_lr, eps=1e-6)
    value_head, value_optimizer = accelerator.prepare(value_head, value_optimizer)

    # Frozen reference for the KL term: the checkpoint training started from.
    reference = posttrain_common.load_reference_model(args, accelerator)
    reference.eval()
    for parameter in reference.parameters():
        parameter.requires_grad_(False)

    slot_starts = body_score_slot_starts(trainer.train_dataset[0]["input_ids"].shape[0])
    # `positions` is the 3*slots token positions the policy emits; `action_positions`
    # is one per ACTION -- the first (onset) token of each triplet. The critic reads
    # the state before the action, so it indexes action_positions - 1.
    positions = score_token_positions(trainer.train_dataset[0]["input_ids"].shape[0])
    action_positions = positions[0::3]

    trainer.log(
        f"Global batch: {args.prompts_per_step} windows/rank x "
        f"{accelerator.num_processes} ranks. "
        f"reward = {args.reward_scale:g} * delta-F1 per triplet "
        f"(table col 3: pitch exact, onset +-1 bin, duration ignored) "
        f"- {args.kl_coef:g} * KL(pi || pi_ref) per token. "
        f"gamma={args.gamma}, lambda={args.gae_lambda}, {args.ppo_epochs} PPO epoch(s), "
        f"log-ratio clamp {args.log_ratio_clip:g}, advantage clip {args.advantage_clip:g}."
    )

    best_val_f1 = -float("inf")

    def validate_and_save_best(*, label=None):
        nonlocal best_val_f1
        results = validate_f1(trainer, args, label=label)
        # REWARD is this run's objective: greedy unfiltered F1 under the reward's
        # own criterion. NOTE the meaning differs from the earlier arms, where
        # val/REWARD was the accuracy sum in [0, 3]; here it is an F1 in [0, 1].
        mean_f1 = results["onset_pitch_tol1"]
        # `due_for_validation` reads this; posttrain_common.validate() maintains it
        # itself, but this arm validates by F1 through its own path, so without
        # this the cadence collapses to "every step" and 96 greedy rollouts run
        # between consecutive updates.
        trainer.last_validated_step = int(trainer.completed_steps)
        if trainer.use_wandb and accelerator.is_main_process:
            import wandb
            wandb.log(
                {
                    "val/REWARD": mean_f1,
                    "val/f1_unfiltered": mean_f1,
                    "val/f1_onset_pitch": results["onset_pitch"],
                    "val/f1_onset_pitch_dur": results["onset_pitch_dur"],
                },
                step=int(trainer.completed_steps),
            )
        if accelerator.is_main_process:
            path = args.output_dir / "val_f1.csv"
            new = not path.exists()
            with path.open("a", encoding="utf-8") as handle:
                if new:
                    handle.write("step,REWARD,f1_onset_pitch,f1_onset_pitch_dur\n")
                handle.write(
                    f"{int(trainer.completed_steps)},{mean_f1},"
                    f"{results['onset_pitch']},{results['onset_pitch_dur']}\n"
                )
        if args.save_best_val and mean_f1 > best_val_f1:
            best_val_f1 = mean_f1
            trainer.save_checkpoint(name="best-val-f1")
            if accelerator.is_main_process:
                path = args.output_dir / "best_val_f1.json"
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(
                        {
                            "step": int(trainer.completed_steps),
                            "val_f1": mean_f1,
                            "checkpoint": str(args.output_dir / "best-val-f1"),
                        },
                        indent=2,
                    )
                    + "\n",
                    encoding="utf-8",
                )
                temporary.replace(path)
                trainer.log(
                    f"New best unfiltered val F1 {100 * mean_f1:.3f}% at step "
                    f"{trainer.completed_steps}"
                )
        return mean_f1

    validate_and_save_best(label="init (before any update)")

    def forward_policy(ids):
        outputs = trainer.model(input_ids=ids, use_cache=False, output_hidden_states=True)
        return outputs.logits, outputs.hidden_states[-1]

    training_failed = False
    try:
        while trainer.completed_steps < args.max_steps:
            for batch in trainer.train_dataloader:
                if trainer.completed_steps >= args.max_steps:
                    break
                step_started = time.perf_counter()

                input_ids = batch["input_ids"].to(accelerator.device)
                labels = batch["labels"].to(accelerator.device)

                # ---- collect: rollout and scoring run WITHOUT dropout -------
                trainer.base_model.eval()
                rollout = rollout_score_slots(
                    trainer.base_model,
                    input_ids,
                    targets=labels,
                    temperature=args.rollout_temperature,
                    constrain=not args.unconstrained_rollout,
                    collect_logprobs=True,
                    collect_gt_ce=False,
                    autocast_ctx=trainer.autocast,
                )
                rolled = rollout["rolled"]
                # One entry per ACTION, and an action is a whole triplet.
                valid = per_triplet(rollout["valid"].float(), reduction="first")

                task_rewards, achieved_f1 = f1_triplet_rewards(
                    rolled, labels, valid, slot_starts, args.reward_scale,
                )

                old_values, old_logprobs, ref_logprobs = [], [], []
                with torch.no_grad():
                    for lo in range(0, rolled.shape[0], args.micro_batch):
                        hi = min(lo + args.micro_batch, rolled.shape[0])
                        with trainer.autocast():
                            outputs = trainer.base_model(
                                input_ids=rolled[lo:hi], use_cache=False,
                                output_hidden_states=True,
                            )
                        hidden = outputs.hidden_states[-1]
                        # V(s_t) is the value of the state BEFORE the triplet is
                        # emitted, so it reads the hidden state that predicts the
                        # triplet's first (onset) token.
                        old_values.append(
                            value_head(
                                hidden[:, action_positions - 1, :].float().detach()
                            )
                        )
                        old_logprobs.append(
                            per_triplet(
                                score_token_logprob(
                                    outputs.logits, rolled[lo:hi], positions,
                                    temperature=args.rollout_temperature,
                                    constrain=not args.unconstrained_rollout,
                                ).detach()
                            )
                        )
                        with trainer.autocast():
                            ref_logits = reference(
                                input_ids=rolled[lo:hi], use_cache=False
                            ).logits
                        ref_logprobs.append(
                            per_triplet(
                                score_token_logprob(
                                    ref_logits, rolled[lo:hi], positions,
                                    temperature=args.rollout_temperature,
                                    constrain=not args.unconstrained_rollout,
                                ).detach()
                            )
                        )
                old_values = torch.cat(old_values, dim=0)
                old_logprob = torch.cat(old_logprobs, dim=0)
                ref_logprob = torch.cat(ref_logprobs, dim=0)

                # KL-to-reference enters the REWARD, so the critic prices drift and
                # the advantage carries it, rather than being a flat loss term.
                kl_to_ref = (old_logprob - ref_logprob) * valid
                rewards = task_rewards - args.kl_coef * kl_to_ref

                advantages, returns = compute_gae(
                    rewards, old_values, valid, gamma=args.gamma, lam=args.gae_lambda,
                )
                mc_returns = discounted_returns(rewards, valid, gamma=args.gamma)
                advantages = whiten(advantages, valid)
                if args.advantage_clip > 0:
                    advantages = advantages.clamp(-args.advantage_clip, args.advantage_clip)
                warming = trainer.completed_steps < args.value_warmup_steps

                # Normalise per ACTION (triplet), not per token.
                token_total = valid.sum().clamp(min=1)
                chunks = list(range(0, rolled.shape[0], args.micro_batch))
                pg_sum = vf_sum = clip_sum = 0.0
                pre_update_kl = 0.0
                epochs_run = 0
                policy_steps_applied = 0
                grad_norm = None

                # ---- optimise: dropout ON here, if requested ----------------
                if args.dropout_in_update and not warming:
                    trainer.model.train()

                for epoch_index in range(args.ppo_epochs):
                    trainer.optimizer.zero_grad(set_to_none=True)
                    value_optimizer.zero_grad(set_to_none=True)
                    epoch_pg_sum = epoch_vf_sum = epoch_clip_sum = epoch_kl_sum = 0.0

                    for index, lo in enumerate(chunks):
                        hi = min(lo + args.micro_batch, rolled.shape[0])
                        is_last = index == len(chunks) - 1
                        stack = contextlib.ExitStack()
                        if not is_last and accelerator.num_processes > 1:
                            if not warming:
                                stack.enter_context(accelerator.no_sync(trainer.model))
                            stack.enter_context(accelerator.no_sync(value_head))
                        with stack:
                            if warming:
                                with torch.no_grad(), trainer.autocast():
                                    hidden = trainer.base_model(
                                        input_ids=rolled[lo:hi], use_cache=False,
                                        output_hidden_states=True,
                                    ).hidden_states[-1]
                                values = value_head(
                                    hidden[:, action_positions - 1, :].float().detach()
                                )
                                vf = value_loss(
                                    values, old_values[lo:hi], returns[lo:hi],
                                    valid[lo:hi], args.value_clip_eps,
                                )
                                pg = torch.zeros((), device=values.device)
                                clipped = approx_kl = pg
                                accelerator.backward(args.vf_coef * vf / token_total)
                            else:
                                logits, hidden = forward_policy(rolled[lo:hi])
                                # Joint log-probability of the whole triplet: the
                                # ratio, the clip and the KL all act on the
                                # composite action rather than on its three tokens
                                # independently.
                                logprob = per_triplet(
                                    score_token_logprob(
                                        logits, rolled[lo:hi], positions,
                                        temperature=args.rollout_temperature,
                                        constrain=not args.unconstrained_rollout,
                                    )
                                )
                                if args.log_ratio_clip > 0:
                                    # Bound the ratio BEFORE the surrogate sees it.
                                    # max(unclipped, clipped) selects the exploding
                                    # branch when the advantage is negative, so an
                                    # unbounded log-ratio is a loss of ~e^|r|.
                                    delta = (logprob - old_logprob[lo:hi]).clamp(
                                        -args.log_ratio_clip, args.log_ratio_clip
                                    )
                                    logprob_for_ratio = old_logprob[lo:hi] + delta
                                else:
                                    logprob_for_ratio = logprob
                                values = value_head(
                                    hidden[:, action_positions - 1, :].float().detach()
                                )
                                pg, clipped, approx_kl = policy_loss(
                                    logprob_for_ratio, old_logprob[lo:hi],
                                    advantages[lo:hi], valid[lo:hi], args.clip_eps,
                                )
                                vf = value_loss(
                                    values, old_values[lo:hi], returns[lo:hi],
                                    valid[lo:hi], args.value_clip_eps,
                                )
                                loss = pg / token_total + args.vf_coef * vf / token_total
                                if is_last:
                                    penalty = trainer.anchor_penalty()
                                    if penalty is not None:
                                        loss = loss + args.original_weight_l2 * penalty
                                accelerator.backward(loss)

                        epoch_pg_sum += float(pg.item())
                        epoch_vf_sum += float(vf.item())
                        epoch_clip_sum += float(clipped.item())
                        epoch_kl_sum += float(approx_kl.item())

                    epoch_kl = trainer.reduce_mean(
                        epoch_kl_sum / float(token_total.item())
                    )
                    if not warming and args.target_kl > 0 and epoch_kl > args.target_kl:
                        trainer.optimizer.zero_grad(set_to_none=True)
                        value_optimizer.zero_grad(set_to_none=True)
                        trainer.log(
                            f"  step {trainer.completed_steps + 1}: pre-update "
                            f"approx_kl {epoch_kl:.4f} > target {args.target_kl}; "
                            f"discarding PPO epoch {epoch_index + 1}."
                        )
                        break

                    pg_sum, vf_sum, clip_sum = epoch_pg_sum, epoch_vf_sum, epoch_clip_sum
                    pre_update_kl = epoch_kl
                    epochs_run += 1
                    if warming:
                        trainer.optimizer.zero_grad(set_to_none=True)
                    else:
                        grad_norm = trainer.optimizer_step(step_scheduler=False)
                        if grad_norm is not False:
                            policy_steps_applied += 1
                    accelerator.clip_grad_norm_(value_head.parameters(), args.max_grad_norm)
                    value_optimizer.step()
                    value_optimizer.zero_grad(set_to_none=True)

                if args.dropout_in_update:
                    trainer.model.eval()
                # Cosine decay advances once per rollout batch, not once per inner
                # epoch, independent of how many survived the trust-region gate.
                if not warming and policy_steps_applied > 0:
                    trainer.scheduler.step()
                trainer.completed_steps += 1

                # train/REWARD is the task reward a rollout earned -- the F1 deltas
                # sum to exactly the rollout's F1, so this is directly comparable
                # with val/REWARD (which is the same quantity, decoded greedily).
                # train/REWARD_total additionally carries the KL penalty, i.e. the
                # quantity PPO actually maximises; both are on the F1 scale, so
                # their gap is what the KL term is costing.
                mean_f1 = trainer.reduce_mean(float(achieved_f1.mean().item()))
                mean_total = trainer.reduce_mean(
                    float((rewards.sum(dim=1).mean() / args.reward_scale).item())
                )
                mean_kl_ref = trainer.reduce_mean(
                    float((kl_to_ref.sum() / token_total).item())
                )
                # Two different questions, and at lambda<1 they have very
                # different ceilings. `ev` asks whether V predicts the FULL
                # undiscounted return-to-go -- which V is never trained to do at
                # lambda=0.95, so its ceiling is ~0.03 here (a perfect critic
                # scores 0.034; verified numerically). `ev_lambda` asks whether V
                # fits its ACTUAL regression target, the lambda-return, and is
                # the one to read for critic convergence. It is mildly
                # self-referential, since returns = A + V, so read it alongside
                # value_loss rather than alone.
                ev = explained_variance(old_values, mc_returns, valid)
                ev_lambda = explained_variance(old_values, returns, valid)

                if accelerator.is_main_process and trainer.use_wandb:
                    import wandb
                    wandb.log(
                        {
                            "train/REWARD": mean_f1,
                            "train/REWARD_total": mean_total,
                            "train/rollout_f1": mean_f1,
                            "train/kl_to_reference": mean_kl_ref,
                            "train/policy_loss": pg_sum / float(token_total.item()),
                            "train/value_loss": vf_sum / float(token_total.item()),
                            "train/clip_fraction": clip_sum / float(token_total.item()),
                            "train/approx_kl": pre_update_kl,
                            "train/explained_variance_lambda": ev_lambda,
                            "train/explained_variance": ev,
                            "train/value_target_std": float(
                                returns[valid.bool()].std().item()
                            ),
                            "train/epochs_applied": epochs_run,
                            "train/policy_steps": policy_steps_applied,
                            "train/lr": trainer.scheduler.get_last_lr()[0],
                            "train/step_seconds": time.perf_counter() - step_started,
                        },
                        step=int(trainer.completed_steps),
                    )

                if trainer.completed_steps % 10 == 0:
                    trainer.log(
                        f"step {trainer.completed_steps} | train REWARD "
                        f"{100 * mean_f1:.3f}% (total {100 * mean_total:.3f}%) | "
                        f"KL(ref) {mean_kl_ref:.4f} | "
                        f"approx_kl {pre_update_kl:.4f} | "
                        f"EV(lambda) {ev_lambda:.3f} EV(mc) {ev:.3f} | "
                        f"{time.perf_counter() - step_started:.1f}s"
                    )

                if trainer.due_for_validation():
                    validate_and_save_best()
                if trainer.due_for_checkpoint():
                    trainer.save_checkpoint()
    except Exception:
        training_failed = True
        raise
    finally:
        if not training_failed:
            validate_and_save_best(label="final")
        trainer.finish()


if __name__ == "__main__":
    main()
