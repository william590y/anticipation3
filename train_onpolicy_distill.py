#!/usr/bin/env python
"""On-policy distillation of the best-val-loss checkpoint.

The model is rolled out over the body score slots of a training window -- its
own sampled score triplets, with the ground-truth control (performance) triplets
teacher-forced after each one, exactly as at inference time -- and then graded
at *every* position by an oracle that knows the answer: the cross-entropy of the
ground-truth next token given the model's own prefix.

  loss = mean over rolled-out score tokens of  -log p_theta(gt_token | own prefix)

This is on-policy distillation with a ground-truth teacher: the state
distribution is the student's (so it sees the states its own mistakes lead to,
which teacher forcing never shows it), while the supervision stays dense and
per-token (so every position gets signal, unlike a sequence-level reward).

The oracle is well defined here only because of the packed format: score slot k
always pairs with control k, so the ground-truth score note for slot k is fixed
regardless of what the model generated for slots 0..k-1. A drifted rollout is
graded against the note it *should* have produced at that slot, which is what
teaches recovery rather than confabulation.

Companion run: `train_grpo.py` optimises the same quantity through a
sequence-level GRPO reward instead of dense per-token CE.

Usage (4 GPUs):
    python -m torch.distributed.run --standalone --nproc_per_node=4 \\
        train_onpolicy_distill.py --output_dir ./run_onpolicy_distill
"""

from __future__ import annotations

import contextlib
import time
import traceback

import torch

import posttrain_common
from onpolicy_rollout import (
    masked_mean,
    per_type_means,
    rollout_score_slots,
    score_token_ce,
)


def parse_args():
    parser = posttrain_common.build_parser(__doc__)
    parser.set_defaults(learning_rate=1e-5, wandb_run_name=None)
    return parser.parse_args()


def main():
    args = parse_args()
    trainer = posttrain_common.PostTrainer(args, method="on-policy distillation")
    accelerator = trainer.accelerator

    trainer.log(
        f"Global batch: {args.prompts_per_step} windows/rank x "
        f"{accelerator.num_processes} ranks = "
        f"{args.prompts_per_step * accelerator.num_processes} rollouts/step, "
        f"update chunked at {args.micro_batch} sequences."
    )

    trainer.validate(label="init (before any update)")

    training_failed = False
    try:
        while trainer.completed_steps < args.max_steps:
            for batch in trainer.train_dataloader:
                if trainer.completed_steps >= args.max_steps:
                    break
                step_started = time.perf_counter()

                input_ids = batch["input_ids"].to(accelerator.device)
                labels = batch["labels"].to(accelerator.device)

                # --- roll the current policy out (no grad) ---------------
                rollout = rollout_score_slots(
                    trainer.base_model,
                    input_ids,
                    targets=labels,
                    temperature=args.rollout_temperature,
                    constrain=not args.unconstrained_rollout,
                    collect_logprobs=False,
                    collect_gt_ce=True,
                    autocast_ctx=trainer.autocast,
                )
                rolled = rollout["rolled"]
                positions = rollout["positions"]
                valid = rollout["valid"]
                rollout_ce = float(masked_mean(rollout["gt_ce"], valid).item())
                rollout_by_type = per_type_means(rollout["gt_ce"], valid, positions)

                # --- dense CE against the oracle, on the policy's own states ---
                # Every microbatch contributes ce_sum / (all valid tokens this
                # step), so accumulating them is an exact token-mean regardless of
                # how the step is chunked; DDP then averages that across ranks.
                chunks = list(range(0, rolled.shape[0], args.micro_batch))
                step_valid_tokens = valid.sum().clamp(min=1)
                ce_total = 0.0
                anchor_value = None
                trainer.optimizer.zero_grad(set_to_none=True)

                for index, start in enumerate(chunks):
                    stop = min(start + args.micro_batch, rolled.shape[0])
                    is_last = index == len(chunks) - 1
                    sync_ctx = (
                        contextlib.nullcontext()
                        if is_last or accelerator.num_processes == 1
                        else accelerator.no_sync(trainer.model)
                    )
                    with sync_ctx:
                        outputs = trainer.model(
                            input_ids=rolled[start:stop], use_cache=False
                        )
                        ce = score_token_ce(outputs.logits, labels[start:stop], positions)
                        ce_sum = (ce * valid[start:stop]).sum()
                        ce_total += float(ce_sum.item())

                        scaled = ce_sum / step_valid_tokens
                        if is_last:
                            penalty = trainer.anchor_penalty()
                            if penalty is not None:
                                anchor_value = float(penalty.item())
                                scaled = scaled + args.original_weight_l2 * penalty
                        accelerator.backward(scaled)

                grad_norm = trainer.optimizer_step()
                trainer.completed_steps += 1
                step = trainer.completed_steps

                # --- logging --------------------------------------------
                mean_loss = trainer.reduce_mean(ce_total / float(step_valid_tokens.item()))
                mean_rollout_ce = trainer.reduce_mean(rollout_ce)
                payload = {
                    "train/loss": mean_loss,
                    "train/rollout_ce": mean_rollout_ce,
                    # The exposure gap lives almost entirely in the onset tokens
                    # (checkpoint-2500: teacher-forced 0.030 -> greedy AR 5.285),
                    # so the per-type split is the interesting series here.
                    "train/rollout_ce_onset": trainer.reduce_mean(rollout_by_type["onset"]),
                    "train/rollout_ce_duration": trainer.reduce_mean(rollout_by_type["duration"]),
                    "train/rollout_ce_pitch": trainer.reduce_mean(rollout_by_type["pitch"]),
                    "train/learning_rate": trainer.scheduler.get_last_lr()[0],
                    "train/step_seconds": time.perf_counter() - step_started,
                }
                if anchor_value is not None:
                    payload["train/anchor_l2"] = anchor_value
                    payload["train/anchor_term"] = args.original_weight_l2 * anchor_value
                if isinstance(grad_norm, float):
                    payload["train/grad_norm"] = grad_norm
                trainer.wandb_log(payload, step)
                trainer.log(
                    f"step {step}/{args.max_steps} | distill CE {mean_loss:.4f} | "
                    f"rollout CE {mean_rollout_ce:.4f} | "
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
