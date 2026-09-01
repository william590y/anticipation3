#!/usr/bin/env python
"""Stage 1 trainer: quantize frozen PianoBART features into a score "plan".

The score encoder is pretrained PianoBART, frozen (``pianobart_encoder.py``).
Trainable: the performance encoder, the cross-attention queries that pool the
frozen score states, the codebook, and a performance-conditioned decoder that
has to rebuild the score from the plan alone.

Reads the same packed windows and applies the same augmentation as ``train.py``
so the codes stage 2 sees are drawn from the distribution the pooling head was
fitted on.

Validation reports reconstruction accuracy twice: once from each window's own
plan and once from a *neighbouring* window's plan. The gap is how much
information the plan actually carries -- if it is ~0 the codes are decorative
and stage 2 has nothing to condition on. Piano rolls of the performance, the
ground-truth score and the reconstruction go to wandb every evaluation.

    torchrun --standalone --nproc_per_node=4 train_plan_vq.py \
        --data_file data/train_paper.txt --val_file data/val_paper.txt \
        --output_dir ./run_plan_vq
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.utils import DistributedDataParallelKwargs
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from packed_dataset import TokenizedDataset
from pianobart_encoder import load_pianobart, score_features
from plan_vq import PlanVQ, PlanVQConfig, save_plan_vq, split_packed_batch
from plan_vq_viz import piano_roll_figure  # also pins matplotlib to Agg

import matplotlib.pyplot as plt  # noqa: E402


def collate_fn(batch):
    return {"input_ids": torch.stack([item["input_ids"] for item in batch])}


def build_eval_dataloader(dataset, accelerator, batch_size, requested_size):
    """A fixed random validation subset, identical across ranks.

    Mirrors ``train._build_random_eval_dataloader`` but without importing
    ``train``, which would pull in the whole GPT-2 fine-tuning stack.
    """
    world = accelerator.num_processes
    count = len(dataset) if requested_size <= 0 else min(requested_size, len(dataset))
    if world > 1 and count >= world:
        count = (count // world) * world

    seed = torch.tensor(
        random.randrange(1, 2**31) if accelerator.is_main_process else 0,
        device=accelerator.device,
        dtype=torch.int64,
    )
    seed = int(accelerator.reduce(seed, reduction="sum").item())
    indices = (
        list(range(len(dataset)))
        if count >= len(dataset)
        else random.Random(seed).sample(range(len(dataset)), count)
    )
    loader = DataLoader(
        Subset(dataset, indices), batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    return accelerator.prepare_data_loader(loader, device_placement=False)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_file", type=Path, default=Path("data/train_paper.txt"))
    parser.add_argument("--val_file", type=Path, default=Path("data/val_paper.txt"))
    parser.add_argument("--output_dir", type=Path, default=Path("./run_plan_vq"))

    parser.add_argument("--pianobart_checkpoint", type=str, default=None,
                        help="Local model.safetensors; downloaded from the Hub when omitted.")

    # plan geometry -- the knobs the two-stage scheme is actually about
    parser.add_argument("--num_codes", type=int, default=8,
                        help="Plan length in tokens (the spec asks for 4-16).")
    parser.add_argument("--codebook_size", type=int, default=512)
    parser.add_argument("--code_dim", type=int, default=32,
                        help="Low dimension is what keeps the codebook fully used.")
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_head", type=int, default=8)
    parser.add_argument("--perf_layers", type=int, default=2,
                        help="Depth of the trainable performance encoder.")
    parser.add_argument("--encoder_layers", type=int, default=2,
                        help="Cross-attention pooling over frozen features; it does not need depth.")
    parser.add_argument("--decoder_layers", type=int, default=4)
    parser.add_argument("--fourier_bands", type=int, default=32,
                        help="Sinusoidal bands per time field in the performance encoder.")
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--commitment", type=float, default=0.25)
    parser.add_argument("--ema_decay", type=float, default=0.99)
    parser.add_argument("--no_l2_normalize", action="store_true")

    # optimisation
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--val_batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=3e-4)
    parser.add_argument("--final_learning_rate", type=float, default=3e-5)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument("--save_steps", type=int, default=2000)
    parser.add_argument("--eval_samples", type=int, default=512)
    parser.add_argument("--piano_roll_examples", type=int, default=4,
                        help="Validation windows drawn as piano rolls each eval. 0 disables.")
    parser.add_argument("--num_workers", type=int, default=6,
                        help="The dataset seeks into a multi-GB text file per item; workers matter here.")
    parser.add_argument("--max_grad_norm", type=float, default=2.0)

    # augmentation -- defaults copied from train.py so stage 1 and stage 2 agree
    parser.add_argument("--onset_jitter_std", type=float, default=0.05)
    parser.add_argument("--dur_jitter_range", type=float, default=0.05)
    parser.add_argument("--mask_prob", type=float, default=0.0)
    parser.add_argument("--transpose_range_semitones", type=int, default=12)
    parser.add_argument("--tempo_scale_range", type=float, default=0.2)

    parser.add_argument("--wandb_project", type=str, default="anticipation-asap")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    return parser


def features_for(encoder, meta, input_ids):
    """Raw performance notes, frozen score features, and the score targets."""
    perf, score, valid = split_packed_batch(input_ids)
    return perf, score_features(encoder, meta, score, valid), score, valid


@torch.no_grad()
def evaluate(model, encoder, meta, accelerator, dataset, args):
    """Reconstruction accuracy with the true plan and with a neighbour's plan."""
    model.eval()
    loader = build_eval_dataloader(dataset, accelerator, args.val_batch_size, args.eval_samples)

    totals = torch.zeros(10, device=accelerator.device, dtype=torch.float64)
    for batch in tqdm(loader, desc="Plan-VQ eval", leave=False,
                      disable=not accelerator.is_local_main_process):
        input_ids = batch["input_ids"].to(accelerator.device)
        perf, features, score, valid = features_for(encoder, meta, input_ids)
        true = model(perf, features, score, valid)
        shuffled = model(perf, features, score, valid, shuffle_codes=True)
        totals += torch.tensor(
            [
                true["accuracies"]["onset"].item(),
                true["accuracies"]["duration"].item(),
                true["accuracies"]["pitch"].item(),
                shuffled["accuracies"]["onset"].item(),
                shuffled["accuracies"]["duration"].item(),
                shuffled["accuracies"]["pitch"].item(),
                true["reconstruction_loss"].item(),
                true["stats"]["perplexity"].item(),
                true["stats"]["usage"].item(),
                1.0,
            ],
            device=accelerator.device,
            dtype=torch.float64,
        )

    totals = accelerator.reduce(totals, reduction="sum")
    batches = max(1.0, totals[9].item())
    keys = [
        "onset_accuracy", "duration_accuracy", "pitch_accuracy",
        "shuffled_onset_accuracy", "shuffled_duration_accuracy", "shuffled_pitch_accuracy",
        "reconstruction_loss", "codebook_perplexity", "codebook_usage",
    ]
    results = {key: totals[i].item() / batches for i, key in enumerate(keys)}
    # The number that says whether the plan is worth having.
    results["onset_plan_gain"] = results["onset_accuracy"] - results["shuffled_onset_accuracy"]
    return results


def main():
    args = build_parser().parse_args()

    accelerator = Accelerator(
        mixed_precision="bf16" if torch.cuda.is_available() else "no",
        # The EMA codebook is kept in sync by an explicit all-reduce inside the
        # quantizer; broadcasting buffers from rank 0 would silently discard the
        # other ranks' statistics.
        kwargs_handlers=[DistributedDataParallelKwargs(broadcast_buffers=False)],
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
    accelerator.wait_for_everyone()

    use_wandb = (
        accelerator.is_main_process and args.wandb_mode != "disabled" and wandb is not None
    )
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or args.output_dir.name,
            mode=args.wandb_mode,
            dir=str(args.output_dir),
            config=vars(args),
        )

    if accelerator.is_main_process:
        print("Loading frozen PianoBART...")
    encoder, meta = load_pianobart(
        device=accelerator.device,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        checkpoint_path=args.pianobart_checkpoint,
    )
    if accelerator.is_main_process:
        frozen = sum(p.numel() for p in encoder.parameters())
        print(f"  {frozen / 1e6:.0f}M frozen parameters, hidden {meta.hidden_size}")

    train_dataset = TokenizedDataset(
        args.data_file,
        onset_jitter_std=args.onset_jitter_std,
        dur_jitter_range=args.dur_jitter_range,
        mask_prob=args.mask_prob,
        transpose_range_semitones=args.transpose_range_semitones,
        tempo_scale_range=args.tempo_scale_range,
        is_training=True,
    )
    val_dataset = TokenizedDataset(args.val_file, is_training=False)
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Empty training or validation set; check the token files.")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
        drop_last=True,
    )

    config = PlanVQConfig(
        num_codes=args.num_codes,
        codebook_size=args.codebook_size,
        code_dim=args.code_dim,
        d_model=args.d_model,
        n_head=args.n_head,
        perf_layers=args.perf_layers,
        encoder_layers=args.encoder_layers,
        decoder_layers=args.decoder_layers,
        fourier_bands=args.fourier_bands,
        dropout=args.dropout,
        commitment=args.commitment,
        ema_decay=args.ema_decay,
        l2_normalize=not args.no_l2_normalize,
        pianobart_dim=meta.hidden_size,
    )
    model = PlanVQ(config)
    if accelerator.is_main_process:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"PlanVQ head: {trainable / 1e6:.1f}M trainable parameters, config={config}")
        (args.output_dir / "plan_vq_config.json").write_text(
            json.dumps(config.to_dict(), indent=2)
        )

    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01, eps=1e-6)
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, args.max_steps - args.warmup_steps)
        cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
        floor = args.final_learning_rate / args.learning_rate
        return floor + (1.0 - floor) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda)

    completed_steps = 0
    best_gain = -float("inf")
    progress = tqdm(total=args.max_steps, desc="PlanVQ", disable=not accelerator.is_main_process)
    model.train()

    def save(tag):
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            save_plan_vq(accelerator.unwrap_model(model), args.output_dir / f"{tag}.pt")
            print(f"  saved {args.output_dir / f'{tag}.pt'}")
        accelerator.wait_for_everyone()

    def log_piano_rolls():
        """Fixed validation windows every eval, rank 0 only.

        Uses the unwrapped module: a DDP forward from a single rank is exactly
        the kind of thing that deadlocks a multi-GPU run.
        """
        if args.piano_roll_examples <= 0 or not accelerator.is_main_process:
            return
        raw = accelerator.unwrap_model(model)
        was_training = raw.training
        raw.eval()
        try:
            count = min(args.piano_roll_examples, len(val_dataset))
            windows = torch.stack(
                [val_dataset[i]["input_ids"] for i in range(count)]
            ).to(accelerator.device)
            perf, features, score, valid = features_for(encoder, meta, windows)
            predicted, _ = raw.reconstruct(perf, features, valid)
            figure = piano_roll_figure(
                perf, score, valid, predicted,
                step=completed_steps,
                max_examples=args.piano_roll_examples,
            )
            roll_dir = args.output_dir / "piano_rolls"
            roll_dir.mkdir(parents=True, exist_ok=True)
            figure.savefig(roll_dir / f"step{completed_steps:06d}.png", dpi=110)
            if use_wandb:
                wandb.log(
                    {"piano_rolls/reconstruction": wandb.Image(figure)}, step=completed_steps
                )
        finally:
            plt.close("all")
            if was_training:
                raw.train()

    def run_validation():
        nonlocal best_gain
        results = evaluate(model, encoder, meta, accelerator, val_dataset, args)
        log_piano_rolls()
        model.train()
        if accelerator.is_main_process:
            print(
                f"\n[val step {completed_steps}] recon onset {100 * results['onset_accuracy']:.2f}% "
                f"dur {100 * results['duration_accuracy']:.2f}% "
                f"pitch {100 * results['pitch_accuracy']:.2f}% | "
                f"onset with a WRONG plan {100 * results['shuffled_onset_accuracy']:.2f}% "
                f"(plan gain {100 * results['onset_plan_gain']:+.2f} pts) | "
                f"codebook perplexity {results['codebook_perplexity']:.1f}, "
                f"codes used/batch {results['codebook_usage']:.0f}"
            )
            if use_wandb:
                wandb.log({f"val/{k}": v for k, v in results.items()}, step=completed_steps)
        if results["onset_plan_gain"] > best_gain:
            best_gain = results["onset_plan_gain"]
            save("plan_vq_best")
        return results

    while completed_steps < args.max_steps:
        for batch in train_dataloader:
            perf, features, score, valid = features_for(encoder, meta, batch["input_ids"])
            outputs = model(perf, features, score, valid)
            loss = outputs["loss"]

            accelerator.backward(loss)
            accelerator.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            completed_steps += 1
            progress.update(1)

            if completed_steps % 25 == 0 and accelerator.is_main_process:
                stats = outputs["stats"]
                message = {
                    "train/loss": loss.item(),
                    "train/reconstruction_loss": outputs["reconstruction_loss"].item(),
                    "train/vq_loss": outputs["vq_loss"].item(),
                    "train/onset_accuracy": outputs["accuracies"]["onset"].item(),
                    "train/duration_accuracy": outputs["accuracies"]["duration"].item(),
                    "train/pitch_accuracy": outputs["accuracies"]["pitch"].item(),
                    "train/codebook_perplexity": stats["perplexity"].item(),
                    "train/codebook_usage": stats["usage"].item(),
                    "train/codes_restarted": stats["restarted"].item(),
                    "train/learning_rate": scheduler.get_last_lr()[0],
                }
                if use_wandb:
                    wandb.log(message, step=completed_steps)
                if completed_steps % 100 == 0:
                    print(
                        f"step {completed_steps}/{args.max_steps} | loss {loss.item():.4f} | "
                        f"onset {100 * message['train/onset_accuracy']:.1f}% "
                        f"dur {100 * message['train/duration_accuracy']:.1f}% "
                        f"pitch {100 * message['train/pitch_accuracy']:.1f}% | "
                        f"perplexity {message['train/codebook_perplexity']:.1f}"
                    )

            if completed_steps % args.eval_steps == 0:
                run_validation()
            if completed_steps % args.save_steps == 0:
                save(f"plan_vq_step{completed_steps}")
            if completed_steps >= args.max_steps:
                break

    progress.close()
    run_validation()
    save("plan_vq_final")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
