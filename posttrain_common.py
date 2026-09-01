"""Shared setup for the two on-policy post-training runs.

`train_onpolicy_distill.py` and `train_grpo.py` start from the same checkpoint,
see the same data with the same augmentation, and are measured with the same
autoregressive validation loss -- only the update rule differs. Everything that
must be identical for the two curves to be comparable lives here; each script
supplies just its own `training_step`.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import random
from pathlib import Path

import numpy as np
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM

import train as base_train
from anticipation.vocab import VOCAB_SIZE
from onpolicy_rollout import (
    append_metrics_row,
    evaluate_autoregressive,
    fixed_eval_indices,
    plot_ar_loss_curve,
)

try:
    import wandb
except ImportError:  # pragma: no cover - optional dependency
    wandb = None

# The non-LoRA checkpoint with the lowest teacher-forced validation loss from the
# paper-split fine-tune (val loss 1.2180) -- the one the visualizer's "best val
# loss" checkpoint set uses. Both post-training runs start here.
DEFAULT_INIT_CHECKPOINT = "run_paper_split_v2/checkpoint-2500"

METRIC_FIELDS = [
    "step",
    "ar_loss",
    "ar_loss_onset",
    "ar_loss_duration",
    "ar_loss_pitch",
    "tf_loss",
    "tf_loss_onset",
    "tf_loss_duration",
    "tf_loss_pitch",
    "ar_pitch_accuracy",
    "ar_onset_accuracy",
    "ar_duration_accuracy",
    "REWARD",
]


def add_common_args(parser):
    parser.add_argument("--data_file", type=Path, default=Path("./data/train_paper.txt"))
    parser.add_argument("--val_file", type=Path, default=Path("./data/val_paper.txt"))
    parser.add_argument(
        "--init_checkpoint",
        type=str,
        default=DEFAULT_INIT_CHECKPOINT,
        help="Weights to post-train (a plain HF checkpoint directory).",
    )
    parser.add_argument("--output_dir", type=Path, required=True)

    parser.add_argument(
        "--prompts_per_step",
        type=int,
        default=8,
        help="Training windows rolled out per rank per optimizer step.",
    )
    parser.add_argument(
        "--micro_batch",
        type=int,
        default=4,
        help="Sequences per grad-enabled forward/backward chunk. Rollouts are "
             "KV-cached and launch-bound so they batch cheaply; the update pass "
             "is what runs out of memory, so it is chunked separately.",
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument(
        "--final_lr_fraction",
        type=float,
        default=0.1,
        help="Cosine-decay floor as a fraction of --learning_rate.",
    )
    parser.add_argument("--max_steps", type=int, default=2000)
    parser.add_argument("--save_steps", type=int, default=250)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--max_grad_norm", type=float, default=2.0)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--rollout_temperature",
        type=float,
        default=1.0,
        help="Sampling temperature for on-policy rollouts (0 = greedy).",
    )
    parser.add_argument(
        "--unconstrained_rollout",
        action="store_true",
        help="Sample rollouts from the full vocabulary instead of the legal "
             "score-token range used by the decoder.",
    )

    parser.add_argument(
        "--eval_windows",
        type=int,
        default=96,
        help="Fixed validation windows for the AR-loss metric. 96 at "
             "--val_batch_size 8 is 12 batches, which divides evenly across both "
             "the 4-GPU and the 3-GPU run.",
    )
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--eval_seed", type=int, default=1234)

    # Augmentation: the musical transforms from the original fine-tune are kept,
    # the noise injections (per-onset/duration jitter, score-embedding masking)
    # are off -- on-policy training already supplies its own input perturbation,
    # and jitter would desynchronise the rolled-out context from the labels.
    parser.add_argument("--transpose_range_semitones", type=int, default=12)
    parser.add_argument("--tempo_scale_range", type=float, default=0.2)
    parser.add_argument("--onset_jitter_std", type=float, default=0.0)
    parser.add_argument("--dur_jitter_range", type=float, default=0.0)

    parser.add_argument(
        "--original_weight_l2",
        type=float,
        default=1e5,
        help="L2 anchor to --init_checkpoint's weights (train.py's default "
             "regulariser, here anchored to the checkpoint being post-trained). "
             "0 disables.",
    )

    parser.add_argument("--wandb_project", type=str, default="anticipation-asap")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument(
        "--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"]
    )
    return parser


class PostTrainer:
    """Model/data/optimizer setup plus the shared validation and checkpoint loop."""

    def __init__(self, args, method):
        self.args = args
        self.method = method

        if args.micro_batch <= 0:
            raise ValueError("--micro_batch must be positive.")
        if args.prompts_per_step <= 0:
            raise ValueError("--prompts_per_step must be positive.")
        if args.original_weight_l2 < 0:
            raise ValueError("--original_weight_l2 must be non-negative.")

        base_train.validate_selected_cuda_device_or_raise()
        base_train.report_runtime_device()

        mixed_precision = "bf16" if torch.cuda.is_available() else "no"
        self.accelerator = Accelerator(mixed_precision=mixed_precision)
        accelerator = self.accelerator

        random.seed(args.seed + accelerator.process_index)
        np.random.seed(args.seed + accelerator.process_index)
        torch.manual_seed(args.seed + accelerator.process_index)

        if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        self.use_wandb = (
            accelerator.is_main_process and args.wandb_mode != "disabled" and wandb is not None
        )
        if self.use_wandb:
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name or args.output_dir.name,
                mode=args.wandb_mode,
                dir=str(args.output_dir),
                config={**vars(args), "method": method},
            )

        self.log(
            f"Distributed: world_size={accelerator.num_processes}, "
            f"rank={accelerator.process_index}, device={accelerator.device}"
        )

        self.train_dataset = base_train.TokenizedDataset(
            args.data_file,
            onset_jitter_std=args.onset_jitter_std,
            dur_jitter_range=args.dur_jitter_range,
            mask_prob=0.0,  # never zero out score embeddings: the rollout *is* the context
            transpose_range_semitones=args.transpose_range_semitones,
            tempo_scale_range=args.tempo_scale_range,
            is_training=True,
        )
        self.val_dataset = base_train.TokenizedDataset(args.val_file, is_training=False)
        if len(self.train_dataset) == 0 or len(self.val_dataset) == 0:
            raise ValueError("Empty training or validation dataset.")

        self.eval_indices = fixed_eval_indices(
            len(self.val_dataset), args.eval_windows, seed=args.eval_seed
        )
        self.log(
            f"Autoregressive validation subset: {len(self.eval_indices)} fixed windows "
            f"of {len(self.val_dataset)} (seed {args.eval_seed})."
        )

        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=args.prompts_per_step,
            shuffle=True,
            collate_fn=_collate,
            pin_memory=torch.cuda.is_available(),
            num_workers=0,
            drop_last=True,
        )

        self.log(f"Loading initial weights from {args.init_checkpoint}...")
        model = AutoModelForCausalLM.from_pretrained(
            args.init_checkpoint, trust_remote_code=True, use_cache=False
        )
        if model.config.vocab_size != VOCAB_SIZE:
            model.resize_token_embeddings(VOCAB_SIZE)

        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            eps=1e-6,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )

        self.model, self.optimizer, self.train_dataloader = accelerator.prepare(
            model, optimizer, train_dataloader
        )
        self.base_model = accelerator.unwrap_model(self.model)

        self.reference_parameters = {}
        if args.original_weight_l2 > 0:
            self.reference_parameters, tensors, params, nbytes = (
                base_train._capture_reference_parameters(self.base_model)
            )
            self.log(
                f"Weight anchor to {args.init_checkpoint}: lambda={args.original_weight_l2}, "
                f"{tensors} tensors, {params:,} params, {nbytes / 1024 ** 2:.0f} MiB."
            )

        floor = max(0.0, args.final_lr_fraction)

        def lr_lambda(step):
            progress = float(step) / float(max(1, args.max_steps))
            cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
            return floor + (1.0 - floor) * cosine

        self.scheduler = LambdaLR(self.optimizer, lr_lambda)

        # Dropout stays off. For GRPO it keeps pi_theta identical to the policy the
        # rollout sampled from (so the importance ratio really is 1 on the first
        # inner epoch); keeping distillation on the same footing makes the two
        # runs differ only in their update rule.
        self.model.eval()

        self.completed_steps = 0
        self.last_validated_step = None
        self.last_checkpoint_step = None
        self.history_steps = []
        self.history_ar = []
        self.history_tf = []

    # -- cadence ----------------------------------------------------------
    # Threshold rather than modulo: GRPO's --inner_epochs takes several optimizer
    # steps per rollout batch, so the step counter can jump past a multiple.

    def due_for_validation(self):
        if self.last_validated_step is None:
            return True
        return self.completed_steps - self.last_validated_step >= self.args.eval_steps

    def due_for_checkpoint(self):
        if self.last_checkpoint_step is None:
            return self.completed_steps >= self.args.save_steps
        return self.completed_steps - self.last_checkpoint_step >= self.args.save_steps

    # -- helpers ----------------------------------------------------------

    def log(self, message):
        if self.accelerator.is_main_process:
            print(message, flush=True)

    def autocast(self):
        return self.accelerator.autocast()

    def anchor_penalty(self):
        if not self.reference_parameters:
            return None
        return base_train._compute_original_weight_l2_penalty(
            self.base_model, self.reference_parameters
        )

    def reduce_mean(self, value):
        tensor = torch.as_tensor(value, dtype=torch.float64, device=self.accelerator.device)
        return float(self.accelerator.reduce(tensor, reduction="mean").item())

    def optimizer_step(self, step_scheduler=True):
        """Clip, step, and report whether the step was applied.

        `step_scheduler=False` is for inner-loop optimisers (PPO takes several
        optimiser steps per training step); the LR schedule must advance once per
        training step, not once per inner epoch, or a cosine decay sized to
        --max_steps reaches its floor several times too early.
        """
        invalid = base_train._find_invalid_gradient_parameter(self.model)
        flagged = base_train._count_flagged_processes(self.accelerator, invalid is not None)
        if flagged > 0:
            self.log(
                f"WARNING: NaN/Inf gradients on {flagged}/{self.accelerator.num_processes} "
                f"rank(s); skipping optimizer step. ({invalid})"
            )
            self.optimizer.zero_grad(set_to_none=True)
            return False
        grad_norm = self.accelerator.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
        self.optimizer.step()
        if step_scheduler:
            self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        return float(grad_norm) if grad_norm is not None else True

    def wandb_log(self, payload, step):
        if self.use_wandb:
            wandb.log(payload, step=step)

    # -- validation / checkpoints -----------------------------------------

    def validate(self, label=None, skip_if_current=False):
        args = self.args
        step = int(self.completed_steps)
        if skip_if_current and self.last_validated_step == step:
            return None
        self.last_validated_step = step
        self.accelerator.wait_for_everyone()
        self.log(f"\nAutoregressive validation at {label or f'step {step}'}...")

        results = evaluate_autoregressive(
            self.base_model,
            self.accelerator,
            self.val_dataset,
            self.eval_indices,
            batch_size=args.val_batch_size,
            autocast_ctx=self.autocast,
        )

        if self.accelerator.is_main_process:
            payload = {
                "val/ar_loss": results["ar_loss"],
                "val/ar_loss_onset": results["ar_loss_by_type"]["onset"],
                "val/ar_loss_duration": results["ar_loss_by_type"]["duration"],
                "val/ar_loss_pitch": results["ar_loss_by_type"]["pitch"],
                "val/tf_loss_score": results["tf_loss"],
                "val/tf_loss_onset": results["tf_loss_by_type"]["onset"],
                "val/tf_loss_duration": results["tf_loss_by_type"]["duration"],
                "val/tf_loss_pitch": results["tf_loss_by_type"]["pitch"],
                "val/exposure_gap": results["ar_loss"] - results["tf_loss"],
                "val/ar_pitch_accuracy": results["ar_accuracy"]["pitch"] * 100,
                "val/ar_onset_accuracy": results["ar_accuracy"]["onset"] * 100,
                "val/ar_duration_accuracy": results["ar_accuracy"]["duration"] * 100,
                # ALL-CAPS so the sequence-level accuracy-sum reward is trivial
                # to find on wandb. Same [0, 3] scale as train/REWARD.
                "val/REWARD": results["reward"],
            }
            self.wandb_log(payload, step)

            self.history_steps.append(step)
            self.history_ar.append(results["ar_loss"])
            self.history_tf.append(results["tf_loss"])

            append_metrics_row(
                args.output_dir,
                {
                    "step": step,
                    "ar_loss": results["ar_loss"],
                    "ar_loss_onset": results["ar_loss_by_type"]["onset"],
                    "ar_loss_duration": results["ar_loss_by_type"]["duration"],
                    "ar_loss_pitch": results["ar_loss_by_type"]["pitch"],
                    "tf_loss": results["tf_loss"],
                    "tf_loss_onset": results["tf_loss_by_type"]["onset"],
                    "tf_loss_duration": results["tf_loss_by_type"]["duration"],
                    "tf_loss_pitch": results["tf_loss_by_type"]["pitch"],
                    "ar_pitch_accuracy": results["ar_accuracy"]["pitch"] * 100,
                    "ar_onset_accuracy": results["ar_accuracy"]["onset"] * 100,
                    "ar_duration_accuracy": results["ar_accuracy"]["duration"] * 100,
                    "REWARD": results["reward"],
                },
                METRIC_FIELDS,
            )
            plot_ar_loss_curve(
                args.output_dir,
                self.history_steps,
                self.history_ar,
                self.history_tf,
                f"{self.method}: validation loss along the rollout",
            )

            print(
                f"  AR loss {results['ar_loss']:.4f} "
                f"(onset {results['ar_loss_by_type']['onset']:.4f}, "
                f"dur {results['ar_loss_by_type']['duration']:.4f}, "
                f"pitch {results['ar_loss_by_type']['pitch']:.4f}) | "
                f"TF loss {results['tf_loss']:.4f} | "
                f"gap {results['ar_loss'] - results['tf_loss']:.4f} | "
                f"AR acc pitch {results['ar_accuracy']['pitch'] * 100:.2f}% "
                f"onset {results['ar_accuracy']['onset'] * 100:.2f}% "
                f"dur {results['ar_accuracy']['duration'] * 100:.2f}% | "
                f"REWARD {results['reward']:.4f} "
                f"({results['windows']} windows)",
                flush=True,
            )

        self.accelerator.wait_for_everyone()
        self.model.eval()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        return results

    def save_checkpoint(self, name=None):
        # Named artifacts (for example PPO's rolling best-val-reward) are
        # independent of the numbered recovery-checkpoint cadence.
        if name is None:
            self.last_checkpoint_step = int(self.completed_steps)
        directory = self.args.output_dir / (name or f"checkpoint-{self.completed_steps}")
        if self.accelerator.is_main_process:
            os.makedirs(directory, exist_ok=True)
        self.accelerator.wait_for_everyone()
        unwrapped = self.accelerator.unwrap_model(self.model)
        unwrapped.save_pretrained(
            directory,
            is_main_process=self.accelerator.is_main_process,
            save_function=self.accelerator.save,
        )
        self.accelerator.wait_for_everyone()
        self.log(f"Saved {directory}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def finish(self):
        if self.use_wandb:
            wandb.finish()


def _collate(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def build_parser(description):
    parser = argparse.ArgumentParser(description=description)
    return add_common_args(parser)


def load_reference_model(args, accelerator):
    """A frozen second copy of `--init_checkpoint`, for a KL-to-reference reward.

    Deliberately NOT wrapped by `accelerator.prepare`: it takes no gradient and
    must not join the DDP reducer. It is placed on the local device and left in
    eval mode, so its log-probabilities are a fixed function of the rollout.
    """
    model = AutoModelForCausalLM.from_pretrained(
        args.init_checkpoint, trust_remote_code=True, use_cache=False
    )
    if model.config.vocab_size != VOCAB_SIZE:
        model.resize_token_embeddings(VOCAB_SIZE)
    model.to(accelerator.device)
    model.eval()
    model.requires_grad_(False)
    return model
