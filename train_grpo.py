#!/usr/bin/env python
"""GRPO post-training of the best-val-loss checkpoint.

Same starting weights, same data, same rollout protocol and same headline metric
as `train_onpolicy_distill.py` -- only the update rule differs, which is the
point of running the two side by side.

Here the oracle grades the rollout **as a whole** instead of token by token:

  reward(rollout) = onset_acc + duration_acc + pitch_acc   in [0, 3]

where each accuracy is the exact-match rate of that token role against the
ground-truth score tokens (no 10 ms bin tolerance) -- the same three quantities
logged as `val/ar_*_accuracy`. The earlier `-(cross entropy across the rollout)`
reward was tail-dominated and flattened or sharpened the policy instead of
improving it.

For each training window the policy samples a *group* of `--group_size`
independent rollouts, the group's rewards are standardised into advantages
(GRPO's critic-free baseline), and every token of a rollout is pushed up or down
by its rollout's advantage:

  loss = - mean_t  min( r_t A,  clip(r_t, 1-eps, 1+eps) A ),
         r_t = pi_theta(y_t | s_t) / pi_old(y_t | s_t)

With the default `--inner_epochs 1` there is exactly one optimizer step per batch
of rollouts, so r_t == 1 when the gradient is taken and the clip is inert -- this
is canonical on-policy GRPO. Raising `--inner_epochs` reuses each (expensive)
rollout batch for several updates, and then the clipping does real work.

The contrast with the distillation run: identical objective, but the signal
arrives as one scalar per 414-token rollout rather than as a target at every
position. The policy has to discover *which* tokens made a rollout good, from
the variance within each group.

Usage (3 GPUs):
    python -m torch.distributed.run --standalone --nproc_per_node=3 \\
        train_grpo.py --output_dir ./run_grpo
"""

from __future__ import annotations

import contextlib
import time
import traceback

import torch
from transformers import AutoModelForCausalLM

import posttrain_common
from onpolicy_rollout import (
    per_rollout_accuracy,
    per_type_means,
    rollout_score_slots,
    score_token_logprob,
    sequence_accuracy_reward,
)


def parse_args():
    parser = posttrain_common.build_parser(__doc__)
    parser.add_argument(
        "--group_size",
        type=int,
        default=8,
        help="Rollouts sampled per training window (GRPO's group).",
    )
    parser.add_argument(
        "--reward",
        choices=["accuracy", "neg_ce"],
        default="accuracy",
        help="accuracy: exact-match onset + duration + pitch accuracy of the rollout "
             "(each token counts only if it equals the ground-truth token exactly). "
             "neg_ce: the earlier reward, -(cross entropy across the whole rollout).",
    )
    parser.add_argument("--clip_eps", type=float, default=0.2)
    parser.add_argument(
        "--inner_epochs",
        type=int,
        default=1,
        help="Optimizer steps taken per batch of rollouts. >1 makes the PPO clip active.",
    )
    parser.add_argument(
        "--advantage_std_norm",
        action="store_true",
        default=True,
        help="Divide centred group rewards by their std (standard GRPO).",
    )
    parser.add_argument(
        "--no_advantage_std_norm",
        dest="advantage_std_norm",
        action="store_false",
        help="Centre only, no std division (the Dr.GRPO correction).",
    )
    parser.add_argument(
        "--advantage_eps",
        type=float,
        default=1e-3,
        help="Floor on the group reward std when standardising advantages. "
             "Raised from 1e-4 because the accuracy-sum reward lives in [0, 3] "
             "and eight T=1 rollouts of one window can share nearly the same "
             "score, which would otherwise blow the advantages up.",
    )
    parser.add_argument(
        "--kl_coef",
        type=float,
        default=0.0,
        help="Coefficient on the k3 KL penalty to the initial policy. >0 loads a "
             "frozen reference copy of --init_checkpoint.",
    )
    # Rollouts are the wall-clock bottleneck and GRPO needs a group per window, so
    # fewer windows per step than the distillation run, with the same total
    # sequences rolled out.
    parser.set_defaults(prompts_per_step=2, learning_rate=1e-6)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.group_size < 2:
        raise SystemExit("--group_size must be at least 2 (advantages need within-group spread).")
    if args.rollout_temperature <= 0:
        raise SystemExit("--rollout_temperature must be > 0: GRPO needs stochastic rollouts.")

    trainer = posttrain_common.PostTrainer(args, method="GRPO")
    accelerator = trainer.accelerator

    rollouts_per_rank = args.prompts_per_step * args.group_size
    trainer.log(
        f"Global batch: {args.prompts_per_step} windows x {args.group_size} rollouts "
        f"= {rollouts_per_rank} rollouts/rank x {accelerator.num_processes} ranks = "
        f"{rollouts_per_rank * accelerator.num_processes} rollouts/step, "
        f"update chunked at {args.micro_batch} sequences, "
        f"{args.inner_epochs} inner epoch(s)."
    )

    reference_model = None
    if args.kl_coef > 0:
        trainer.log(f"Loading frozen reference policy from {args.init_checkpoint}...")
        reference_model = AutoModelForCausalLM.from_pretrained(
            args.init_checkpoint, trust_remote_code=True, use_cache=False
        )
        reference_model.to(accelerator.device)
        reference_model.eval()
        reference_model.requires_grad_(False)

    trainer.validate(label="init (before any update)")

    training_failed = False
    try:
        while trainer.completed_steps < args.max_steps:
            for batch in trainer.train_dataloader:
                if trainer.completed_steps >= args.max_steps:
                    break
                step_started = time.perf_counter()

                # One prompt becomes `group_size` identical rows; the sampler makes
                # them diverge.
                input_ids = batch["input_ids"].to(accelerator.device)
                labels = batch["labels"].to(accelerator.device)
                input_ids = input_ids.repeat_interleave(args.group_size, dim=0)
                labels = labels.repeat_interleave(args.group_size, dim=0)

                # --- sample a group of rollouts and score each one -------
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
                valid = rollout["valid"]
                old_logprob = rollout["logprob"]

                token_counts = valid.sum(dim=1).clamp(min=1)
                rollout_ce = (rollout["gt_ce"] * valid).sum(dim=1) / token_counts
                accuracy = per_rollout_accuracy(rolled, labels, positions, valid)
                rewards = (
                    sequence_accuracy_reward(accuracy)
                    if args.reward == "accuracy"
                    else -rollout_ce
                )
                rollout_by_type = per_type_means(rollout["gt_ce"], valid, positions)

                grouped = rewards.view(args.prompts_per_step, args.group_size)
                centred = grouped - grouped.mean(dim=1, keepdim=True)
                group_std = grouped.std(dim=1, keepdim=True)
                if args.advantage_std_norm:
                    advantages = centred / (group_std + args.advantage_eps)
                else:
                    advantages = centred
                advantages = advantages.reshape(-1)

                # --- policy-gradient updates on the sampled rollouts -----
                chunks = list(range(0, rolled.shape[0], args.micro_batch))
                step_valid_tokens = valid.sum().clamp(min=1)
                anchor_value = None
                surrogate_total = 0.0
                clipped_total = 0.0
                ratio_total = 0.0
                kl_total = 0.0

                for _ in range(args.inner_epochs):
                    trainer.optimizer.zero_grad(set_to_none=True)
                    surrogate_total = clipped_total = ratio_total = kl_total = 0.0

                    for index, start in enumerate(chunks):
                        stop = min(start + args.micro_batch, rolled.shape[0])
                        is_last = index == len(chunks) - 1
                        sync_ctx = (
                            contextlib.nullcontext()
                            if is_last or accelerator.num_processes == 1
                            else accelerator.no_sync(trainer.model)
                        )
                        chunk_valid = valid[start:stop]
                        chunk_adv = advantages[start:stop].unsqueeze(1)

                        with sync_ctx:
                            outputs = trainer.model(
                                input_ids=rolled[start:stop], use_cache=False
                            )
                            logprob = score_token_logprob(
                                outputs.logits,
                                rolled[start:stop],
                                positions,
                                temperature=args.rollout_temperature,
                                constrain=not args.unconstrained_rollout,
                            )
                            ratio = torch.exp(logprob - old_logprob[start:stop])
                            unclipped = ratio * chunk_adv
                            clipped = (
                                ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps) * chunk_adv
                            )
                            surrogate = -torch.min(unclipped, clipped)
                            objective_sum = (surrogate * chunk_valid).sum()

                            loss_sum = objective_sum
                            if reference_model is not None:
                                with torch.no_grad(), trainer.autocast():
                                    reference_logits = reference_model(
                                        input_ids=rolled[start:stop], use_cache=False
                                    ).logits
                                reference_logprob = score_token_logprob(
                                    reference_logits,
                                    rolled[start:stop],
                                    positions,
                                    temperature=args.rollout_temperature,
                                    constrain=not args.unconstrained_rollout,
                                )
                                delta = reference_logprob - logprob
                                kl = torch.exp(delta) - delta - 1.0
                                kl_total += float((kl * chunk_valid).sum().item())
                                loss_sum = loss_sum + args.kl_coef * (kl * chunk_valid).sum()

                            scaled = loss_sum / step_valid_tokens
                            if is_last:
                                penalty = trainer.anchor_penalty()
                                if penalty is not None:
                                    anchor_value = float(penalty.item())
                                    scaled = scaled + args.original_weight_l2 * penalty
                            accelerator.backward(scaled)

                        surrogate_total += float(objective_sum.item())
                        was_clipped = (unclipped > clipped) & chunk_valid
                        clipped_total += float(was_clipped.sum().item())
                        ratio_total += float((ratio * chunk_valid).sum().item())

                    grad_norm = trainer.optimizer_step()
                    trainer.completed_steps += 1

                step = trainer.completed_steps
                total_tokens = float(step_valid_tokens.item())

                payload = {
                    "train/policy_loss": trainer.reduce_mean(surrogate_total / total_tokens),
                    # ALL-CAPS: the sequence-level accuracy-sum reward in [0, 3].
                    "train/REWARD": trainer.reduce_mean(float(rewards.mean().item())),
                    "train/reward_mean": trainer.reduce_mean(float(rewards.mean().item())),
                    "train/reward_onset_acc": trainer.reduce_mean(float(accuracy[:, 0].mean().item())),
                    "train/reward_duration_acc": trainer.reduce_mean(float(accuracy[:, 1].mean().item())),
                    "train/reward_pitch_acc": trainer.reduce_mean(float(accuracy[:, 2].mean().item())),
                    "train/rollout_ce": trainer.reduce_mean(float(rollout_ce.mean().item())),
                    "train/rollout_ce_best_in_group": trainer.reduce_mean(
                        float(rollout_ce.view(args.prompts_per_step, args.group_size)
                              .min(dim=1).values.mean().item())
                    ),
                    "train/rollout_ce_onset": trainer.reduce_mean(rollout_by_type["onset"]),
                    "train/rollout_ce_duration": trainer.reduce_mean(rollout_by_type["duration"]),
                    "train/rollout_ce_pitch": trainer.reduce_mean(rollout_by_type["pitch"]),
                    "train/group_reward_std": trainer.reduce_mean(
                        float(group_std.mean().item())
                    ),
                    "train/advantage_abs_mean": trainer.reduce_mean(
                        float(advantages.abs().mean().item())
                    ),
                    "train/ratio_mean": trainer.reduce_mean(ratio_total / total_tokens),
                    "train/clip_fraction": trainer.reduce_mean(clipped_total / total_tokens),
                    "train/learning_rate": trainer.scheduler.get_last_lr()[0],
                    "train/step_seconds": time.perf_counter() - step_started,
                }
                if reference_model is not None:
                    payload["train/kl_to_init"] = trainer.reduce_mean(kl_total / total_tokens)
                if anchor_value is not None:
                    payload["train/anchor_l2"] = anchor_value
                    payload["train/anchor_term"] = args.original_weight_l2 * anchor_value
                if isinstance(grad_norm, float):
                    payload["train/grad_norm"] = grad_norm
                trainer.wandb_log(payload, step)
                trainer.log(
                    f"step {step}/{args.max_steps} | "
                    f"REWARD {payload['train/REWARD']:.4f} "
                    f"(onset {payload['train/reward_onset_acc']:.3f} "
                    f"dur {payload['train/reward_duration_acc']:.3f} "
                    f"pitch {payload['train/reward_pitch_acc']:.3f}) | "
                    f"rollout CE {payload['train/rollout_ce']:.4f} "
                    f"(best-of-{args.group_size} {payload['train/rollout_ce_best_in_group']:.4f}) | "
                    f"group std {payload['train/group_reward_std']:.4f} | "
                    f"lr {payload['train/learning_rate']:.3e} | "
                    f"{payload['train/step_seconds']:.1f}s"
                )

                checkpoint_due = trainer.due_for_checkpoint()
                if trainer.due_for_validation() or checkpoint_due:
                    trainer.validate()
                if checkpoint_due:
                    trainer.save_checkpoint()
    except Exception:
        training_failed = True
        print(traceback.format_exc(), flush=True)
        raise
    finally:
        if not training_failed:
            trainer.validate(label="final", skip_if_current=True)
            trainer.save_checkpoint(name="final")
        trainer.finish()


if __name__ == "__main__":
    main()
