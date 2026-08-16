"""Train an LTLM where the oracle posterior is regularized toward p(z|P).

Loss
----
    L = E_{q(z|P,S)} [ -log p_theta(S | P, z) ] + beta * KL(q(z|P,S) || p_phi(z|P))

Fast inner loop (AdamVI) fits q with p_phi held fixed. The slow step updates
the decoder and the performance planner. beta=0 recovers an unconstrained
oracle; larger beta forces thoughts to be recoverable from performance.

This is the intended replacement for the previous isotropic KL(q || N(0,I))
plus auxiliary diffusion loss, which let the oracle encode the score.

Launch (same data flags as the Cornell LTLM jobs)::

    python train_ltlm.py \\
        --data_file data/train_paper.txt --val_file data/val_paper.txt \\
        --output_dir ./run_ltlm_planner --wandb_run_name ltlm_planner \\
        --kl_weight 5 --thoughts_per_layer 4 --mcmc_steps 16 \\
        --batch_size 4 --gradient_accumulation_steps 4 --max_steps 20000
"""

from __future__ import annotations

import argparse
import math
import os
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM

try:
    import wandb
except ImportError:
    wandb = None

from accelerate import Accelerator

from anticipation.ltlm_model import LTLMCausalLM, control_token_mask
from anticipation.ltlm_objective import latent_alignment_stats, planner_regularized_loss
from anticipation.ltlm_posterior import PosteriorOptimizer
from anticipation.vocab import VOCAB_SIZE
from train import (
    TokenizedDataset,
    report_runtime_device,
    validate_selected_cuda_device_or_raise,
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_file", type=Path, default=Path("./data/train_paper.txt"))
    parser.add_argument("--val_file", type=Path, default=Path("./data/val_paper.txt"))
    parser.add_argument("--model_name", type=str, default="stanford-crfm/music-medium-800k")
    parser.add_argument("--output_dir", type=Path, default=Path("./run_ltlm_planner"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--final_learning_rate", type=float, default=3e-6)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=2500)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--eval_max_samples", type=int, default=100)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--max_grad_norm", type=float, default=2.0)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--thoughts_per_layer", type=int, default=4)
    parser.add_argument("--mcmc_steps", type=int, default=16)
    parser.add_argument("--inference_method", type=str, default="adamVI")
    parser.add_argument("--fast_lr", type=float, default=0.3)
    parser.add_argument("--final_fast_lr", type=float, default=0.34)
    parser.add_argument(
        "--kl_weight",
        type=float,
        default=5.0,
        help="beta in KL(q(z|P,S) || p_phi(z|P)). 0 = unconstrained oracle.",
    )
    parser.add_argument(
        "--isotropic_kl_weight",
        type=float,
        default=0.0,
        help="Optional extra KL(q || N(0,I)). Leave 0; the planner KL replaces it.",
    )
    parser.add_argument("--elbo_reduction", type=str, default="mean", choices=["mean", "batchmean", "sum"])
    parser.add_argument("--onset_jitter_std", type=float, default=0.05)
    parser.add_argument("--dur_jitter_range", type=float, default=0.05)
    parser.add_argument("--mask_prob", type=float, default=0.0)
    parser.add_argument("--transpose_range_semitones", type=int, default=12)
    parser.add_argument("--tempo_scale_range", type=float, default=0.2)
    parser.add_argument("--loss_mask_performance_tokens", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--wandb_project", type=str, default="anticipation-asap")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online", choices=["online", "offline", "disabled"])
    return parser.parse_args()


def collate_fn(batch):
    return {
        "input_ids": torch.stack([item["input_ids"] for item in batch]),
        "attention_mask": torch.stack([item["attention_mask"] for item in batch]),
        "labels": torch.stack([item["labels"] for item in batch]),
    }


def performance_only_labels(input_ids: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Keep NLL only on control tokens so VI / eval sees q(z|P) rather than q(z|P,S)."""
    masked = labels.clone()
    score_positions = ~control_token_mask(input_ids)
    masked = masked.masked_fill(score_positions, -100)
    return masked


def cosine_lr_lambda(step: int, max_steps: int, initial_lr: float, final_lr: float) -> float:
    progress = float(step) / float(max(1, max_steps))
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return (final_lr / initial_lr) + (1.0 - final_lr / initial_lr) * cosine_decay


@torch.no_grad()
def evaluate_paths(model: LTLMCausalLM, posterior: PosteriorOptimizer, dataloader, accelerator, max_batches: int):
    """Teacher-forced NLL on oracle q(z|P,S), prefix/performance q(z|P), and planner p(z|P)."""
    model.eval()
    totals = {
        "oracle_nll": 0.0, "oracle_kl": 0.0, "oracle_cos": 0.0, "oracle_n": 0,
        "prefix_nll": 0.0, "prefix_kl": 0.0, "prefix_cos": 0.0, "prefix_n": 0,
        "planner_nll": 0.0, "planner_n": 0,
    }
    n_batches = 0
    for batch in dataloader:
        if max_batches > 0 and n_batches >= max_batches:
            break
        input_ids = batch["input_ids"].to(accelerator.device)
        labels = batch["labels"].to(accelerator.device)
        attention_mask = batch["attention_mask"].to(accelerator.device)

        oracle = posterior.infer(input_ids, labels, attention_mask=attention_mask)
        nll_o, _ = model.score_nll(input_ids, labels, oracle["z"], attention_mask=attention_mask)
        mu_p, log_var_p = model.planner_params(input_ids)
        _, o_stats = planner_regularized_loss(
            nll_o, oracle["mu_q"], oracle["log_var_q"], mu_p, log_var_p, beta=model.beta
        )
        align_o = latent_alignment_stats(oracle["mu_q"], mu_p)

        prefix_labels = performance_only_labels(input_ids, labels)
        prefix = posterior.infer(input_ids, prefix_labels, attention_mask=attention_mask)
        nll_px, _ = model.score_nll(input_ids, labels, prefix["z"], attention_mask=attention_mask)
        _, p_stats = planner_regularized_loss(
            nll_px, prefix["mu_q"], prefix["log_var_q"], mu_p, log_var_p, beta=model.beta
        )
        align_p = latent_alignment_stats(prefix["mu_q"], mu_p)

        nll_pl, _ = model.score_nll(input_ids, labels, mu_p, attention_mask=attention_mask)

        bs = input_ids.shape[0]
        totals["oracle_nll"] += float(nll_o) * bs
        totals["oracle_kl"] += float(o_stats["planner_kl"]) * bs
        totals["oracle_cos"] += float(align_o["z_cosine"]) * bs
        totals["oracle_n"] += bs
        totals["prefix_nll"] += float(nll_px) * bs
        totals["prefix_kl"] += float(p_stats["planner_kl"]) * bs
        totals["prefix_cos"] += float(align_p["z_cosine"]) * bs
        totals["prefix_n"] += bs
        totals["planner_nll"] += float(nll_pl) * bs
        totals["planner_n"] += bs
        n_batches += 1

    def avg(key, count_key):
        n = max(totals[count_key], 1)
        return totals[key] / n

    model.train()
    return {
        "val/oracle/loss": avg("oracle_nll", "oracle_n"),
        "val/oracle/kl": avg("oracle_kl", "oracle_n"),
        "val/oracle/z_cosine_vs_planner": avg("oracle_cos", "oracle_n"),
        "val/prefix/loss": avg("prefix_nll", "prefix_n"),
        "val/prefix/kl": avg("prefix_kl", "prefix_n"),
        "val/prefix/z_cosine_vs_planner": avg("prefix_cos", "prefix_n"),
        "val/planner/loss": avg("planner_nll", "planner_n"),
        "val/loss": avg("oracle_nll", "oracle_n"),
    }


def main():
    args = parse_args()
    if args.inference_method != "adamVI":
        raise ValueError("Only --inference_method adamVI is implemented")
    if wandb is None and args.wandb_mode != "disabled":
        args.wandb_mode = "disabled"

    validate_selected_cuda_device_or_raise(force_cpu=args.force_cpu)
    report_runtime_device(force_cpu=args.force_cpu)

    mixed_precision = "bf16" if torch.cuda.is_available() and not args.force_cpu else "no"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        cpu=args.force_cpu,
        mixed_precision=mixed_precision,
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    use_wandb = accelerator.is_main_process and args.wandb_mode != "disabled" and wandb is not None
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.output_dir.name,
            mode=args.wandb_mode,
            dir=str(args.output_dir),
            config=vars(args),
        )

    train_dataset = TokenizedDataset(
        args.data_file,
        onset_jitter_std=args.onset_jitter_std,
        dur_jitter_range=args.dur_jitter_range,
        mask_prob=args.mask_prob,
        transpose_range_semitones=args.transpose_range_semitones,
        tempo_scale_range=args.tempo_scale_range,
        loss_mask_performance_tokens=args.loss_mask_performance_tokens,
        is_training=True,
    )
    val_dataset = TokenizedDataset(args.val_file, is_training=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.force_cpu,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available() and not args.force_cpu,
    )

    base = AutoModelForCausalLM.from_pretrained(args.model_name, trust_remote_code=True, use_cache=False)
    if base.config.vocab_size != VOCAB_SIZE:
        base.resize_token_embeddings(VOCAB_SIZE)
    model = LTLMCausalLM(
        base,
        thoughts_per_layer=args.thoughts_per_layer,
        beta=args.kl_weight,
        isotropic_kl_weight=args.isotropic_kl_weight,
        elbo_reduction=args.elbo_reduction,
    )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=1e-6, weight_decay=0.01)
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)

    from torch.optim.lr_scheduler import LambdaLR

    scheduler = LambdaLR(
        optimizer,
        lambda step: cosine_lr_lambda(step, args.max_steps, args.learning_rate, args.final_learning_rate),
    )

    raw_model = accelerator.unwrap_model(model)
    posterior = PosteriorOptimizer(
        raw_model,
        num_steps=args.mcmc_steps,
        lr=args.fast_lr,
        final_lr=args.final_fast_lr,
    )

    completed_steps = 0
    progress = tqdm(total=args.max_steps, disable=not accelerator.is_main_process, desc="LTLM")

    def run_validation():
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            max_batches = max(1, args.eval_max_samples // max(args.val_batch_size, 1))
            metrics = evaluate_paths(raw_model, posterior, val_loader, accelerator, max_batches)
            print(
                f"step {completed_steps}: "
                f"oracle NLL {metrics['val/oracle/loss']:.4f}  "
                f"planner NLL {metrics['val/planner/loss']:.4f}  "
                f"prefix NLL {metrics['val/prefix/loss']:.4f}  "
                f"cos(q,p) {metrics['val/oracle/z_cosine_vs_planner']:.3f}"
            )
            if use_wandb:
                wandb.log(metrics, step=completed_steps)
        accelerator.wait_for_everyone()
        model.train()

    try:
        while completed_steps < args.max_steps:
            for batch in train_loader:
                with accelerator.accumulate(model):
                    input_ids = batch["input_ids"]
                    labels = batch["labels"]
                    attention_mask = batch["attention_mask"]

                    # Fast: fit q(z|P,S) against a frozen planner.
                    inferred = posterior.infer(input_ids, labels, attention_mask=attention_mask)
                    mu_q = inferred["mu_q"]
                    log_var_q = inferred["log_var_q"]
                    eps = inferred["eps"]

                    # Slow: update decoder + planner. q is treated as data.
                    mu_q = mu_q.detach()
                    log_var_q = log_var_q.detach()
                    loss, stats, _ = raw_model.elbo(
                        input_ids,
                        labels,
                        mu_q,
                        log_var_q,
                        eps,
                        attention_mask=attention_mask,
                        detach_planner=False,
                    )
                    accelerator.backward(loss)
                    if accelerator.sync_gradients:
                        accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()
                        completed_steps += 1
                        progress.update(1)
                        if accelerator.is_main_process and use_wandb:
                            wandb.log(
                                {
                                    "train/loss": float(stats["loss"]),
                                    "train/mcmc_nll": float(stats["nll"]),
                                    "train/planner_kl": float(stats["planner_kl"]),
                                    "train/z_cosine_vs_planner": float(stats["z_cosine"]),
                                    "train/z_rmse_vs_planner": float(stats["z_rmse"]),
                                    "train/learning_rate": scheduler.get_last_lr()[0],
                                },
                                step=completed_steps,
                            )
                        if completed_steps % args.eval_steps == 0:
                            run_validation()
                        if (
                            accelerator.is_main_process
                            and completed_steps % args.save_steps == 0
                        ):
                            ckpt = args.output_dir / f"checkpoint-{completed_steps}"
                            ckpt.mkdir(parents=True, exist_ok=True)
                            accelerator.unwrap_model(model).base_model.save_pretrained(ckpt)
                            extra_state = {
                                k: v.detach().cpu()
                                for k, v in raw_model.state_dict().items()
                                if not k.startswith("base_model.")
                            }
                            torch.save(
                                {"ltlm": extra_state, "step": completed_steps, "args": vars(args)},
                                ckpt / "ltlm_extra.pt",
                            )
                        if completed_steps >= args.max_steps:
                            break
            if completed_steps >= args.max_steps:
                break
    finally:
        progress.close()
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
