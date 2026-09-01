#!/usr/bin/env python
"""CRPO post-training of the best-val-loss checkpoint.

Wu et al., *Contrastive Reinforced Policy Optimization via Privileged
Self-Distillation* (arXiv:2607.28026), applied to packed score infilling.

    L_CRPO*(theta) = L_GRPO(theta) + lambda * L_CRPO(theta)              (Eq. 13)

The paper's diagnosis is the one this repo ran into head-on: on-policy self
distillation gives dense logit-level supervision but inherits exposure bias from
the self-teacher's privileged information, and the optimisation direction goes
bad. CRPO keeps the dense signal but reads it contrastively -- predictive entropy
separates positions where the privileged teacher genuinely knows better
(reflective exploration, pulled together) from positions where its advantage is
just exposure bias (pushed apart), and the contrast is normalised group-wise so
the signal stays granular.

Concretely, per group of `--group_size` rollouts of one window:

  1. sample the group on-policy (identical rollout protocol to train_grpo.py)
  2. student distribution  pi(.|S_{i,t})  from the rolled-out sequence   (grad)
  3. teacher distribution  pi(.|T_{i,t})  from the ground-truth sequence (no grad)
  4. renormalise both on the teacher's top-K support                (App. C.4)
  5. dE = E^S - E^T with E = sum p log p; bottom `--positive_fraction` of dE
     over the pooled group is P, the rest is N                       (Eqs. 7-9)
  6. sim = -KL(student || teacher); InfoNCE over P against P u N       (Eq. 11)
  7. add it to the GRPO surrogate with weight lambda                   (Eq. 13)

The privileged context is the ground-truth prefix -- see crpo.py for why that is
the faithful mapping of the paper's reference-solution context onto a packed
window, and what follows from it.

Companion run: train_grpo.py is the lambda -> 0 limit of this objective, so the
two runs isolate exactly what the contrastive term contributes.

Usage (3 GPUs):
    python -m torch.distributed.run --standalone --nproc_per_node=3 \\
        train_crpo.py --output_dir ./run_crpo
"""

from __future__ import annotations

import contextlib
import time
import traceback

import torch
from transformers import AutoModelForCausalLM

import posttrain_common
from crpo import EMATeacher, crpo_group_loss
from onpolicy_rollout import (
    per_rollout_accuracy,
    per_type_means,
    rollout_score_slots,
    score_token_logprob,
    sequence_accuracy_reward,
)


def parse_args():
    parser = posttrain_common.build_parser(__doc__)
    parser.add_argument("--group_size", type=int, default=8)
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
        "--advantage_eps",
        type=float,
        default=1e-3,
        help="Floor on the group reward std when standardising advantages. "
             "Raised from 1e-4 because the accuracy-sum reward lives in [0, 3] "
             "and eight T=1 rollouts of one window can share nearly the same "
             "score, which would otherwise blow the advantages up.",
    )
    parser.add_argument(
        "--advantage_std_norm", action="store_true", default=True,
        help="Divide centred group rewards by their std (standard GRPO).",
    )
    parser.add_argument(
        "--no_advantage_std_norm", dest="advantage_std_norm", action="store_false",
    )
    # Rescaled from the paper (Appendix B.2 had lambda=5, tau=1). See
    # train_crpo.sbatch for the measurements that forced the change: tau=1
    # collapsed Eq. 11's softmax (KLs here are 6-14 nats, not <1) and
    # lambda=5 let the contrastive term dominate L_GRPO.
    parser.add_argument("--crpo_lambda", type=float, default=2.0, help="lambda in Eq. 13.")
    parser.add_argument("--crpo_temperature", type=float, default=10.0, help="tau in Eq. 11.")
    parser.add_argument("--positive_fraction", type=float, default=0.3, help="p in Eq. 8.")
    parser.add_argument("--top_k", type=int, default=100, help="K of Appendix C.4.")
    parser.add_argument(
        "--ema_teacher_alpha", type=float, default=0.1,
        help="Trust-region self-teacher EMA (Appendix C.3). 0 = teacher is the "
             "current policy under stop-gradient, and no extra model copy.",
    )
    parser.add_argument(
        "--grpo_weight", type=float, default=1.0,
        help="Weight on L_GRPO in Eq. 13. Set 0 for the contrastive term alone.",
    )
    parser.set_defaults(prompts_per_step=2, learning_rate=1e-6)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.group_size < 2:
        raise SystemExit("--group_size must be at least 2.")
    if args.rollout_temperature <= 0:
        raise SystemExit("--rollout_temperature must be > 0: the group needs spread.")

    trainer = posttrain_common.PostTrainer(args, method="CRPO")
    accelerator = trainer.accelerator

    trainer.log(
        f"Global batch: {args.prompts_per_step} windows x {args.group_size} rollouts "
        f"= {args.prompts_per_step * args.group_size} rollouts/rank x "
        f"{accelerator.num_processes} ranks. "
        f"lambda={args.crpo_lambda}, tau={args.crpo_temperature}, "
        f"p={args.positive_fraction}, K={args.top_k}, ema_alpha={args.ema_teacher_alpha}."
    )

    # One group is one contrastive problem: Eq. 11's softmax runs over every
    # position of every rollout in the group, so a group cannot be split across
    # backward passes. The group is therefore the microbatch.
    if args.ema_teacher_alpha > 0:
        trainer.log(f"Trust-region self-teacher: EMA copy from {args.init_checkpoint}.")
        teacher_module = AutoModelForCausalLM.from_pretrained(
            args.init_checkpoint, trust_remote_code=True, use_cache=False
        )
        teacher_module.to(accelerator.device)
        teacher_module.eval()
        teacher_module.requires_grad_(False)
    else:
        trainer.log("Self-teacher is the current policy under stop-gradient (no EMA).")
        teacher_module = trainer.base_model
    teacher = EMATeacher(teacher_module, args.ema_teacher_alpha)

    trainer.validate(label="init (before any update)")

    training_failed = False
    try:
        while trainer.completed_steps < args.max_steps:
            for batch in trainer.train_dataloader:
                if trainer.completed_steps >= args.max_steps:
                    break
                step_started = time.perf_counter()

                window_ids = batch["input_ids"].to(accelerator.device)
                window_labels = batch["labels"].to(accelerator.device)
                input_ids = window_ids.repeat_interleave(args.group_size, dim=0)
                labels = window_labels.repeat_interleave(args.group_size, dim=0)

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
                advantages = (
                    centred / (group_std + args.advantage_eps)
                    if args.advantage_std_norm else centred
                )
                advantages = advantages.reshape(-1)

                step_valid_tokens = valid.sum().clamp(min=1)
                trainer.optimizer.zero_grad(set_to_none=True)

                anchor_value = None
                policy_total = 0.0
                contrastive_total = 0.0
                clipped_total = 0.0
                crpo_stats_sum = {}

                for group_index in range(args.prompts_per_step):
                    lo = group_index * args.group_size
                    hi = lo + args.group_size
                    is_last = group_index == args.prompts_per_step - 1
                    sync_ctx = (
                        contextlib.nullcontext()
                        if is_last or accelerator.num_processes == 1
                        else accelerator.no_sync(trainer.model)
                    )
                    group_valid = valid[lo:hi]

                    # The teacher context is the ground-truth prefix, which every
                    # rollout in the group shares -- one teacher forward per group.
                    with torch.no_grad(), trainer.autocast():
                        teacher_logits = teacher.module(
                            input_ids=window_ids[group_index : group_index + 1],
                            use_cache=False,
                        ).logits.detach()

                    with sync_ctx:
                        outputs = trainer.model(input_ids=rolled[lo:hi], use_cache=False)

                        logprob = score_token_logprob(
                            outputs.logits, rolled[lo:hi], positions,
                            temperature=args.rollout_temperature,
                            constrain=not args.unconstrained_rollout,
                        )
                        ratio = torch.exp(logprob - old_logprob[lo:hi])
                        chunk_adv = advantages[lo:hi].unsqueeze(1)
                        unclipped = ratio * chunk_adv
                        clipped = ratio.clamp(1.0 - args.clip_eps, 1.0 + args.clip_eps) * chunk_adv
                        policy_sum = (-torch.min(unclipped, clipped) * group_valid).sum()

                        crpo_loss, crpo_stats = crpo_group_loss(
                            outputs.logits,
                            teacher_logits,
                            positions,
                            group_valid,
                            top_k=args.top_k,
                            temperature=args.crpo_temperature,
                            positive_fraction=args.positive_fraction,
                        )

                        loss = args.grpo_weight * (policy_sum / step_valid_tokens)
                        loss = loss + args.crpo_lambda * crpo_loss / args.prompts_per_step
                        if is_last:
                            penalty = trainer.anchor_penalty()
                            if penalty is not None:
                                anchor_value = float(penalty.item())
                                loss = loss + args.original_weight_l2 * penalty
                        accelerator.backward(loss)

                    policy_total += float(policy_sum.item())
                    contrastive_total += float(crpo_loss.item())
                    clipped_total += float(((unclipped > clipped) & group_valid).sum().item())
                    for key, value in crpo_stats.items():
                        crpo_stats_sum[key] = crpo_stats_sum.get(key, 0.0) + value

                grad_norm = trainer.optimizer_step()
                teacher.update(trainer.base_model)
                trainer.completed_steps += 1
                step = trainer.completed_steps

                groups = float(args.prompts_per_step)
                total_tokens = float(step_valid_tokens.item())
                payload = {
                    "train/policy_loss": trainer.reduce_mean(policy_total / total_tokens),
                    "train/contrastive_loss": trainer.reduce_mean(contrastive_total / groups),
                    "train/rollout_ce": trainer.reduce_mean(float(rollout_ce.mean().item())),
                    "train/rollout_ce_onset": trainer.reduce_mean(rollout_by_type["onset"]),
                    "train/rollout_ce_duration": trainer.reduce_mean(rollout_by_type["duration"]),
                    "train/rollout_ce_pitch": trainer.reduce_mean(rollout_by_type["pitch"]),
                    # ALL-CAPS: the sequence-level accuracy-sum reward in [0, 3].
                    "train/REWARD": trainer.reduce_mean(float(rewards.mean().item())),
                    "train/reward_mean": trainer.reduce_mean(float(rewards.mean().item())),
                    "train/reward_onset_acc": trainer.reduce_mean(float(accuracy[:, 0].mean().item())),
                    "train/reward_duration_acc": trainer.reduce_mean(float(accuracy[:, 1].mean().item())),
                    "train/reward_pitch_acc": trainer.reduce_mean(float(accuracy[:, 2].mean().item())),
                    "train/group_reward_std": trainer.reduce_mean(float(group_std.mean().item())),
                    "train/clip_fraction": trainer.reduce_mean(clipped_total / total_tokens),
                    "train/learning_rate": trainer.scheduler.get_last_lr()[0],
                    "train/step_seconds": time.perf_counter() - step_started,
                }
                for key, value in crpo_stats_sum.items():
                    payload[f"crpo/{key}"] = trainer.reduce_mean(value / groups)
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
                    f"rollout CE {payload['train/rollout_ce']:.4f} | "
                    f"L_CRPO {payload['train/contrastive_loss']:.4f} | "
                    f"KL+ {payload.get('crpo/kl_positive', float('nan')):.3f} "
                    f"KL- {payload.get('crpo/kl_negative', float('nan')):.3f} | "
                    f"H_student {payload.get('crpo/student_entropy', float('nan')):.3f} "
                    f"H_teacher {payload.get('crpo/teacher_entropy', float('nan')):.3f} | "
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
