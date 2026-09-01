#!/usr/bin/env python
"""Token-level PPO post-training of the best-val-loss checkpoint.

Same starting weights, data, rollout protocol and validation metric as the GRPO
arm; the difference is where the reward lives and who assigns credit.

  GRPO  one scalar per 414-token rollout, spread uniformly over every token,
        with a group of 8 samples supplying the baseline.
  PPO   +1 / -1 on each token as it is generated, a learned critic supplying the
        baseline, and GAE propagating the downstream cost of a bad token back to
        the token that caused it.

The reward is assigned to the action that earned it -- `rewards[:, t]` scores
`positions[t]`, the token just emitted -- so delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)
separates "this token was wrong" from "this token was fine but everything after it
drifted". That distinction is the entire point of the arm: in this format a wrong
onset does not just cost its own token, it desynchronises the rollout and costs
the pitch copies that follow.

The critic is a small MLP on *detached* final hidden states, so the value loss
trains the head alone and never reshapes the policy trunk.

Usage (3 GPUs):
    python -m torch.distributed.run --standalone --nproc_per_node=3 \\
        train_ppo.py --output_dir ./run_ppo
"""

from __future__ import annotations

import contextlib
import json
import time
import traceback

import torch
from torch.optim import AdamW

import posttrain_common
from onpolicy_rollout import (
    per_rollout_accuracy,
    per_type_means,
    rollout_score_slots,
    score_token_logprob,
)
from ppo import (
    ValueHead,
    approximate_kl,
    compute_gae,
    discounted_returns,
    explained_variance,
    policy_loss,
    token_rewards,
    value_loss,
    whiten,
)


def parse_args():
    parser = posttrain_common.build_parser(__doc__)
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument("--value_clip_eps", type=float, default=0.2)
    parser.add_argument("--vf_coef", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Discount. 1.0 = undiscounted episode (414 steps).")
    parser.add_argument("--gae_lambda", type=float, default=0.95)
    parser.add_argument("--ppo_epochs", type=int, default=4,
                        help="Passes over each rollout batch. Rollouts dominate wall "
                             "clock, so reusing them is most of PPO's efficiency; the "
                             "clip is what keeps the reuse honest.")
    parser.add_argument("--value_lr", type=float, default=1e-4,
                        help="Critic learning rate. Higher than the policy's: the head "
                             "is randomly initialised and has to catch up.")
    parser.add_argument("--value_head_hidden", type=int, default=256)
    parser.add_argument("--value_warmup_steps", type=int, default=25,
                        help="Steps where only the critic is updated. A zero-init head "
                             "would otherwise put pure noise in every advantage.")
    parser.add_argument("--advantage_clip", type=float, default=5.0,
                        help="Clamp whitened advantages to +/- this. 0 disables.")
    parser.add_argument("--target_kl", type=float, default=0.05,
                        help="Stop the remaining PPO epochs for this batch once the "
                             "policy has moved this far from the sampling policy. "
                             "0 disables.")
    parser.add_argument("--reward_correct", type=float, default=1.0)
    parser.add_argument("--reward_wrong", type=float, default=-1.0)
    parser.add_argument(
        "--no_save_best_val",
        dest="save_best_val",
        action="store_false",
        help="Do not maintain output_dir/best-val-reward (useful for short probes).",
    )
    parser.set_defaults(save_best_val=True)
    parser.add_argument(
        "--no_save_final",
        dest="save_final",
        action="store_false",
        help="Do not write output_dir/final (useful for short probes).",
    )
    parser.set_defaults(save_final=True)
    parser.set_defaults(prompts_per_step=8, micro_batch=4, learning_rate=1e-6)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rollout_temperature <= 0:
        raise SystemExit("--rollout_temperature must be > 0: PPO needs stochastic actions.")

    trainer = posttrain_common.PostTrainer(args, method="PPO")
    accelerator = trainer.accelerator

    hidden_size = trainer.base_model.config.n_embd
    value_head = ValueHead(hidden_size, args.value_head_hidden)
    value_optimizer = AdamW(value_head.parameters(), lr=args.value_lr, eps=1e-6)
    value_head, value_optimizer = accelerator.prepare(value_head, value_optimizer)

    trainer.log(
        f"Global batch: {args.prompts_per_step} windows/rank x "
        f"{accelerator.num_processes} ranks = "
        f"{args.prompts_per_step * accelerator.num_processes} rollouts/step "
        f"(one per window -- PPO needs no group). "
        f"gamma={args.gamma}, lambda={args.gae_lambda}, {args.ppo_epochs} PPO epochs, "
        f"critic {hidden_size}->{args.value_head_hidden}->1 on detached features, "
        f"reward {args.reward_correct:+g}/{args.reward_wrong:+g} per token."
    )

    best_val_reward = -float("inf")

    def validate_and_save_best(*, label=None, skip_if_current=False):
        """Validate and keep the exact best policy checkpoint for downstream viz."""
        nonlocal best_val_reward
        results = trainer.validate(label=label, skip_if_current=skip_if_current)
        if results is None:
            return None
        reward = float(results["reward"])
        if args.save_best_val and reward > best_val_reward:
            best_val_reward = reward
            trainer.save_checkpoint(name="best-val-reward")
            if accelerator.is_main_process:
                metadata = {
                    "step": int(trainer.completed_steps),
                    "val_reward": reward,
                    "checkpoint": str(args.output_dir / "best-val-reward"),
                }
                path = args.output_dir / "best_val_reward.json"
                temporary = path.with_suffix(path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                )
                temporary.replace(path)
                trainer.log(
                    f"New best val REWARD {reward:.6f} at step "
                    f"{trainer.completed_steps}; saved "
                    f"{args.output_dir / 'best-val-reward'}"
                )
        return results

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

                # ---- collect: one on-policy rollout per window ----------
                rollout = rollout_score_slots(
                    trainer.base_model,
                    input_ids,
                    targets=labels,
                    temperature=args.rollout_temperature,
                    constrain=not args.unconstrained_rollout,
                    collect_logprobs=True,
                    collect_gt_ce=True,
                    autocast_ctx=trainer.autocast,
                )
                rolled = rollout["rolled"]
                positions = rollout["positions"]
                valid = rollout["valid"].float()
                behavior_logprob = rollout["logprob"]

                rewards = token_rewards(
                    rolled, labels, positions, valid,
                    correct=args.reward_correct, wrong=args.reward_wrong,
                )

                # ---- critic pass over the collected batch ---------------
                old_values = []
                old_logprobs = []
                with torch.no_grad():
                    for lo in range(0, rolled.shape[0], args.micro_batch):
                        hi = min(lo + args.micro_batch, rolled.shape[0])
                        with trainer.autocast():
                            outputs = trainer.base_model(
                                input_ids=rolled[lo:hi], use_cache=False,
                                output_hidden_states=True,
                            )
                        hidden = outputs.hidden_states[-1]
                        old_values.append(
                            value_head(hidden[:, positions - 1, :].float().detach())
                        )
                        old_logprobs.append(
                            score_token_logprob(
                                outputs.logits,
                                rolled[lo:hi],
                                positions,
                                temperature=args.rollout_temperature,
                                constrain=not args.unconstrained_rollout,
                            ).detach()
                        )
                old_values = torch.cat(old_values, dim=0)
                # PPO must compare like with like.  Incremental KV-cache BF16
                # decoding and a full BF16 forward can diverge numerically on a
                # long rollout; using the cached behavior log-probability made
                # ratios differ from 1 before any update.  The denominator below
                # uses the identical full-forward path/microbatch shape as the
                # optimizer, while the discrepancy remains logged explicitly.
                old_logprob = torch.cat(old_logprobs, dim=0)

                advantages, returns = compute_gae(
                    rewards, old_values, valid, gamma=args.gamma, lam=args.gae_lambda,
                )
                mc_returns = discounted_returns(rewards, valid, gamma=args.gamma)
                advantages = whiten(advantages, valid)
                if args.advantage_clip > 0:
                    # Bound the per-token update regardless of critic quality. With
                    # gamma=1 over 414 steps the returns span +/-414, so a critic
                    # that is still near-useless (explained_variance ~0.03) emits
                    # outlier predictions; after whitening a lone outlier among 3312
                    # tokens sits at ~sqrt(3312) = 57 sigma and gets slammed. That is
                    # exactly the observed failure: approx_kl 1.6e5 is a mean over
                    # 3312 tokens, i.e. ONE token whose log-prob moved ~20 nats.
                    advantages = advantages.clamp(-args.advantage_clip, args.advantage_clip)
                warming = trainer.completed_steps < args.value_warmup_steps

                # ---- optimise -------------------------------------------
                token_total = valid.sum().clamp(min=1)
                chunks = list(range(0, rolled.shape[0], args.micro_batch))
                with torch.no_grad():
                    cache_full_kl_sum = approximate_kl(
                        old_logprob, behavior_logprob, valid
                    )
                    cache_full_abs_sum = (
                        (behavior_logprob - old_logprob).abs() * valid
                    ).sum()
                    local_cache_full_max = (
                        (behavior_logprob - old_logprob).abs() * valid
                    ).max()
                    cache_full_max = float(
                        accelerator.reduce(
                            local_cache_full_max, reduction="max"
                        ).item()
                    )
                pg_sum = vf_sum = clip_sum = 0.0
                first_epoch_kl = 0.0
                pre_update_kl = 0.0
                rejected_kl = 0.0
                epochs_run = 0
                policy_steps_applied = 0
                grad_norm = None

                for epoch_index in range(args.ppo_epochs):
                    trainer.optimizer.zero_grad(set_to_none=True)
                    value_optimizer.zero_grad(set_to_none=True)
                    epoch_pg_sum = 0.0
                    epoch_vf_sum = 0.0
                    epoch_clip_sum = 0.0
                    epoch_kl_sum = 0.0

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
                                # Critic-only step. The DDP-wrapped policy must NOT
                                # be invoked here: its output would not reach the
                                # loss (the critic reads detached features), so no
                                # parameter of it would ever produce a gradient and
                                # DDP's reducer would never finalise that backward.
                                # Doing it anyway corrupted the accumulation --
                                # approx_kl went 0.003 -> 1.6e5 the instant the
                                # policy loss switched on, and got worse the longer
                                # the warmup ran. Take the features from the
                                # unwrapped model under no_grad instead.
                                with torch.no_grad(), trainer.autocast():
                                    hidden = trainer.base_model(
                                        input_ids=rolled[lo:hi], use_cache=False,
                                        output_hidden_states=True,
                                    ).hidden_states[-1]
                                values = value_head(hidden[:, positions - 1, :].float().detach())
                                vf = value_loss(
                                    values, old_values[lo:hi], returns[lo:hi],
                                    valid[lo:hi], args.value_clip_eps,
                                )
                                pg = torch.zeros((), device=values.device)
                                clipped = approx_kl = pg
                                accelerator.backward(args.vf_coef * vf / token_total)
                            else:
                                logits, hidden = forward_policy(rolled[lo:hi])
                                logprob = score_token_logprob(
                                    logits, rolled[lo:hi], positions,
                                    temperature=args.rollout_temperature,
                                    constrain=not args.unconstrained_rollout,
                                )
                                values = value_head(
                                    hidden[:, positions - 1, :].float().detach()
                                )
                                pg, clipped, approx_kl = policy_loss(
                                    logprob, old_logprob[lo:hi], advantages[lo:hi],
                                    valid[lo:hi], args.clip_eps,
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

                    # Inspect the current policy before applying this epoch's
                    # gradient.  Previously the optimizer stepped first and the
                    # pre-step KL was checked afterwards, granting every breach
                    # one additional update and leaving the last update unchecked.
                    epoch_kl = trainer.reduce_mean(
                        epoch_kl_sum / float(token_total.item())
                    )
                    if epoch_index == 0:
                        first_epoch_kl = epoch_kl
                    if (
                        not warming
                        and args.target_kl > 0
                        and epoch_kl > args.target_kl
                    ):
                        rejected_kl = epoch_kl
                        trainer.optimizer.zero_grad(set_to_none=True)
                        value_optimizer.zero_grad(set_to_none=True)
                        trainer.log(
                            f"  step {trainer.completed_steps + 1}: pre-update "
                            f"approx_kl {epoch_kl:.4f} > target {args.target_kl}; "
                            f"discarding PPO epoch {epoch_index + 1} "
                            f"({epochs_run} applied)."
                        )
                        break

                    pg_sum = epoch_pg_sum
                    vf_sum = epoch_vf_sum
                    clip_sum = epoch_clip_sum
                    pre_update_kl = epoch_kl
                    epochs_run += 1
                    if warming:
                        trainer.optimizer.zero_grad(set_to_none=True)
                        grad_norm = None
                    else:
                        grad_norm = trainer.optimizer_step(step_scheduler=False)
                        # ``optimizer_step`` returns exactly ``False`` when it
                        # rejects NaN/Inf gradients; a valid zero gradient norm
                        # may compare false-y, so test the sentinel explicitly.
                        if grad_norm is not False:
                            policy_steps_applied += 1
                    accelerator.clip_grad_norm_(value_head.parameters(), args.max_grad_norm)
                    value_optimizer.step()
                    value_optimizer.zero_grad(set_to_none=True)

                # Cosine decay advances once per rollout batch, independent of
                # how many inner epochs survived the trust-region gate.
                if not warming and policy_steps_applied > 0:
                    trainer.scheduler.step()

                # Measure the policy that actually resulted from the accepted
                # updates.  This catches a final-epoch overshoot, which a
                # pre-update early-stop cannot observe until after the step.
                if warming:
                    post_update_kl = 0.0
                else:
                    current_logprobs = []
                    with torch.no_grad():
                        for lo in chunks:
                            hi = min(lo + args.micro_batch, rolled.shape[0])
                            with trainer.autocast():
                                outputs = trainer.base_model(
                                    input_ids=rolled[lo:hi], use_cache=False
                                )
                            current_logprobs.append(
                                score_token_logprob(
                                    outputs.logits,
                                    rolled[lo:hi],
                                    positions,
                                    temperature=args.rollout_temperature,
                                    constrain=not args.unconstrained_rollout,
                                ).detach()
                            )
                    current_logprob = torch.cat(current_logprobs, dim=0)
                    post_update_kl = trainer.reduce_mean(
                        float(approximate_kl(
                            current_logprob, old_logprob, valid
                        ).item()) / float(token_total.item())
                    )

                trainer.completed_steps += 1
                step = trainer.completed_steps

                accuracy = per_rollout_accuracy(rolled, labels, positions, rollout["valid"])
                rollout_by_type = per_type_means(
                    rollout["gt_ce"], rollout["valid"], positions
                )
                tokens = float(token_total.item())
                payload = {
                    "train/REWARD": trainer.reduce_mean(float(accuracy.sum(dim=1).mean().item())),
                    "train/reward_per_token": trainer.reduce_mean(
                        float((rewards * valid).sum().item()) / tokens
                    ),
                    "train/reward_onset_acc": trainer.reduce_mean(float(accuracy[:, 0].mean().item())),
                    "train/reward_duration_acc": trainer.reduce_mean(float(accuracy[:, 1].mean().item())),
                    "train/reward_pitch_acc": trainer.reduce_mean(float(accuracy[:, 2].mean().item())),
                    "train/policy_loss": trainer.reduce_mean(pg_sum / tokens),
                    "train/value_loss": trainer.reduce_mean(vf_sum / tokens),
                    # Monte Carlo returns are independent of V.  The legacy GAE
                    # lambda-return includes V and can therefore report high EV
                    # partly by construction.
                    "train/explained_variance": trainer.reduce_mean(
                        explained_variance(old_values, mc_returns, valid)
                    ),
                    "train/gae_explained_variance": trainer.reduce_mean(
                        explained_variance(old_values, returns, valid)
                    ),
                    "train/value_mean": trainer.reduce_mean(float((old_values * valid).sum().item()) / tokens),
                    "train/return_mean": trainer.reduce_mean(float((returns * valid).sum().item()) / tokens),
                    "train/mc_return_mean": trainer.reduce_mean(
                        float((mc_returns * valid).sum().item()) / tokens
                    ),
                    "train/clip_fraction": trainer.reduce_mean(clip_sum / tokens),
                    "train/approx_kl": post_update_kl,
                    "train/first_epoch_approx_kl": first_epoch_kl,
                    "train/pre_update_approx_kl": pre_update_kl,
                    "train/rejected_approx_kl": rejected_kl,
                    "train/cache_full_approx_kl": trainer.reduce_mean(
                        float(cache_full_kl_sum.item()) / tokens
                    ),
                    "train/cache_full_mean_abs_logprob_delta": trainer.reduce_mean(
                        float(cache_full_abs_sum.item()) / tokens
                    ),
                    "train/cache_full_max_abs_logprob_delta": cache_full_max,
                    "train/ppo_epochs_run": float(epochs_run),
                    "train/policy_steps_applied": float(policy_steps_applied),
                    "train/rollout_ce": trainer.reduce_mean(
                        float((rollout["gt_ce"] * rollout["valid"]).sum().item()) / tokens
                    ),
                    "train/rollout_ce_onset": trainer.reduce_mean(rollout_by_type["onset"]),
                    "train/rollout_ce_duration": trainer.reduce_mean(rollout_by_type["duration"]),
                    "train/rollout_ce_pitch": trainer.reduce_mean(rollout_by_type["pitch"]),
                    "train/learning_rate": trainer.scheduler.get_last_lr()[0],
                    "train/step_seconds": time.perf_counter() - step_started,
                }
                if isinstance(grad_norm, float):
                    payload["train/grad_norm"] = grad_norm
                trainer.wandb_log(payload, step)
                trainer.log(
                    f"step {step}/{args.max_steps} | REWARD {payload['train/REWARD']:.4f} "
                    f"(onset {payload['train/reward_onset_acc']:.3f} "
                    f"dur {payload['train/reward_duration_acc']:.3f} "
                    f"pitch {payload['train/reward_pitch_acc']:.3f}) | "
                    f"EVmc {payload['train/explained_variance']:+.3f} "
                    f"EVgae {payload['train/gae_explained_variance']:+.3f} | "
                    f"clip {payload['train/clip_fraction']:.3f} "
                    f"KLpost {payload['train/approx_kl']:.4f} "
                    f"KLcache {payload['train/cache_full_approx_kl']:.4f}"
                    f"{'  [critic warmup]' if warming else ''} | "
                    f"{payload['train/step_seconds']:.1f}s"
                )

                checkpoint_due = trainer.due_for_checkpoint()
                if trainer.due_for_validation() or checkpoint_due:
                    validate_and_save_best()
                if checkpoint_due:
                    trainer.save_checkpoint()
    except Exception:
        training_failed = True
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        if not training_failed:
            validate_and_save_best(label="final", skip_if_current=True)
            if args.save_final:
                trainer.save_checkpoint(name="final")
        trainer.finish()


if __name__ == "__main__":
    main()
