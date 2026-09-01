#!/usr/bin/env python
"""Stage 2: fine-tune the LM with a VQ "plan" prefixed to every packed window.

Identical to ``train.py`` -- same data, augmentation, optimizer, cosine
schedule, weight anchor, gradient clipping, effective batch and step count --
except that each window is preceded by ``[PLAN_BOS] c_1 ... c_K``, the plan
codes for that window. The model predicts the codes like any other token (so it
learns a prior over plans) and then conditions on them.

Codes are computed **online**, by running the frozen stage-1 encoder
(``--vq_checkpoint`` plus PianoBART) over each batch as it arrives. That is not
just convenience: the LM sees a freshly transposed and tempo-scaled window every
epoch, and a plan describing the un-augmented score does not describe what the
model is looking at. Both stages run on stock transformers, so they can share a
process.

Validation reports the plan from both ends:

  ``val/ar_*_accuracy``            oracle plan (the encoder's codes)   -- ceiling
  ``val/pred_ar_*_accuracy``       the model's own greedy plan         -- reality
  ``val/plan_code_accuracy``       how often it recovers a true code

If the second and third are far below the first, the plan is carrying
information the model cannot reconstruct at inference time; see the placement
discussion in ``plan_lm.py``.

    torchrun --standalone --nproc_per_node=4 train_plan_lm.py \
        --vq_checkpoint run_plan_vq/plan_vq_best.pt --output_dir ./run_plan_lm
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import traceback
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

from anticipation.config import EVENT_SIZE
from anticipation.packed_sequence import ALTERNATING_START, iter_score_slot_positions
from anticipation.vocab import REST
from plan_lm import (
    PACKED_LENGTH,
    PlanLayout,
    assemble_inputs,
    assemble_labels,
    plan_autoregressive_generate_score,
    prepare_plan_model,
)
from packed_dataset import TokenizedDataset
from pianobart_encoder import load_pianobart, score_features
from plan_vq import load_plan_vq, split_packed_batch
from train import (
    TOKEN_TYPE_NAMES,
    _build_random_eval_dataloader,
    _capture_reference_parameters,
    _compute_original_weight_l2_penalty,
    _count_flagged_processes,
    _find_invalid_gradient_parameter,
    _get_input_embedding_layer,
    _move_batch_to_device,
    build_error_heatmap_chart,
    num_score_slots,
)


def collate_fn(batch):
    return {
        key: torch.stack([item[key] for item in batch])
        for key in ("input_ids", "attention_mask", "labels", "score_token_mask", "score_mask")
    }


def insert_plan_block(tensor, block, layout):
    """Splice a plan-shaped block into a packed-length tensor at the plan slot."""
    cut = layout.insert_at
    return torch.cat([tensor[:, :cut], block, tensor[:, cut:]], dim=1)


def packed_index_map(layout, device):
    """Assembled index -> packed index, with ``-1`` at the plan tokens."""
    cut = layout.insert_at
    mapping = torch.empty(layout.total_length, dtype=torch.long, device=device)
    mapping[:cut] = torch.arange(cut, device=device)
    mapping[cut : cut + layout.plan_length] = -1
    mapping[cut + layout.plan_length :] = torch.arange(cut, PACKED_LENGTH, device=device)
    return mapping


class OnlinePlanEncoder:
    """The frozen stage-1 encoder, run inline on the batch the LM will see.

    Holds PianoBART and the stage-1 VQ head; neither ever receives a gradient.
    The codes therefore describe the *augmented* window, which is the whole
    point of not precomputing them.
    """

    def __init__(self, vq_checkpoint, device, dtype, pianobart_checkpoint=None):
        self.plan_vq = load_plan_vq(vq_checkpoint, device=device)
        self.plan_vq.requires_grad_(False)
        self.encoder, self.meta = load_pianobart(
            device=device, dtype=dtype, checkpoint_path=pianobart_checkpoint
        )
        self.num_codes = self.plan_vq.config.num_codes
        self.codebook_size = self.plan_vq.config.codebook_size

    @torch.no_grad()
    def __call__(self, input_ids):
        perf, score, valid = split_packed_batch(input_ids)
        features = score_features(self.encoder, self.meta, score, valid)
        return self.plan_vq.encode_codes(perf, features, valid).to(input_ids.device)


class PlanBatcher:
    """Turns a packed batch into a plan-prefixed batch."""

    def __init__(self, plan_encoder, layout):
        self.plan_encoder = plan_encoder
        self.layout = layout

    def codes(self, batch):
        return self.plan_encoder(batch["input_ids"])

    def build(self, batch, codes=None):
        input_ids = batch["input_ids"]
        if codes is None:
            codes = self.codes(batch)

        tokens, position_ids = assemble_inputs(input_ids, codes, self.layout)
        labels = assemble_labels(batch["labels"], codes, self.layout)
        # An explicit all-ones attention mask is REQUIRED, not cosmetic: with
        # attention_mask=None transformers inspects position_ids for "packed
        # sequence" format, and the plan's dedicated position rows look like a
        # document boundary, which would block the window from attending to the
        # plan at all.
        attention_mask = torch.ones_like(tokens)

        score_mask = batch.get("score_mask")
        if score_mask is not None and torch.any(score_mask).item():
            plan_pad = torch.zeros(
                (tokens.shape[0], self.layout.plan_length),
                dtype=score_mask.dtype,
                device=score_mask.device,
            )
            score_mask = insert_plan_block(score_mask, plan_pad, self.layout)
        else:
            score_mask = None

        return {
            "input_ids": tokens,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "score_mask": score_mask,
            "codes": codes,
        }


def forward_plan_batch(model, prepared):
    kwargs = {
        "attention_mask": prepared["attention_mask"],
        "position_ids": prepared["position_ids"],
        "labels": prepared["labels"],
    }
    score_mask = prepared["score_mask"]
    if score_mask is None:
        return model(input_ids=prepared["input_ids"], **kwargs)

    inputs_embeds = _get_input_embedding_layer(model)(prepared["input_ids"])
    inputs_embeds = inputs_embeds.masked_fill(score_mask.unsqueeze(-1), 0.0)
    return model(inputs_embeds=inputs_embeds, **kwargs)


# ---------------------------------------------------------------------------
# validation
# ---------------------------------------------------------------------------

def _score_slot_stats(generated, reference, heatmap_slots, accumulators):
    """Accumulate per-slot autoregressive errors (both tensors packed-shaped)."""
    pitch_err, onset_err, dur_err, onset_abs, dur_abs, slot_total = accumulators
    for slot_idx, pos in enumerate(iter_score_slot_positions(PACKED_LENGTH, ALTERNATING_START)):
        if slot_idx >= heatmap_slots:
            break
        real = reference[:, pos + 2] != REST
        if not torch.any(real):
            continue
        slot_total[slot_idx] += int(real.sum().item())
        onset_err[slot_idx] += int(((generated[:, pos] != reference[:, pos]) & real).sum().item())
        dur_err[slot_idx] += int(
            ((generated[:, pos + 1] != reference[:, pos + 1]) & real).sum().item()
        )
        pitch_err[slot_idx] += int(
            ((generated[:, pos + 2] != reference[:, pos + 2]) & real).sum().item()
        )
        onset_abs[slot_idx] += float(
            (torch.abs(generated[:, pos] - reference[:, pos]) * real).sum().item()
        )
        dur_abs[slot_idx] += float(
            (torch.abs(generated[:, pos + 1] - reference[:, pos + 1]) * real).sum().item()
        )


@torch.no_grad()
def evaluate_plan_model(
    model,
    batcher,
    accelerator,
    dataset,
    args,
    heatmap_slots,
):
    layout = batcher.layout
    model.eval()

    # ---- teacher forced ----------------------------------------------------
    loader = _build_random_eval_dataloader(
        dataset=dataset,
        accelerator=accelerator,
        batch_size=args.val_batch_size,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.eval_num_workers,
        requested_size=args.eval_max_samples,
        description="Teacher-forced eval",
    )

    mapping = packed_index_map(layout, accelerator.device)
    roles = mapping % EVENT_SIZE

    loss_sum = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    sample_count = torch.zeros((), device=accelerator.device, dtype=torch.float64)
    type_ce = torch.zeros(EVENT_SIZE, device=accelerator.device, dtype=torch.float64)
    type_count = torch.zeros(EVENT_SIZE, device=accelerator.device, dtype=torch.float64)
    pitch_hits = torch.zeros(2, device=accelerator.device, dtype=torch.float64)
    plan_stats = torch.zeros(3, device=accelerator.device, dtype=torch.float64)  # ce, hits, count
    # `loss` is what the model optimizes and includes the plan tokens; this is
    # the packed-tokens-only number, which IS comparable to train.py's val/loss.
    packed_ce = torch.zeros(2, device=accelerator.device, dtype=torch.float64)

    for batch in tqdm(loader, desc="Evaluating", leave=False,
                      disable=not accelerator.is_local_main_process):
        batch = _move_batch_to_device(batch, accelerator.device)
        prepared = batcher.build(batch)
        outputs = forward_plan_batch(model, prepared)

        logits = outputs.logits[:, :-1, :].float()
        labels = prepared["labels"][:, 1:]
        target_role = roles[1:]
        target_packed = mapping[1:]

        losses = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
            ignore_index=-100,
        ).view(labels.shape)
        valid = labels != -100

        loss_sum += float(outputs.loss.item()) * batch["input_ids"].shape[0]
        sample_count += batch["input_ids"].shape[0]

        packed_valid = valid & (target_packed >= 0).unsqueeze(0)
        packed_ce[0] += losses[packed_valid].sum().double()
        packed_ce[1] += packed_valid.sum().double()
        for token_type in range(EVENT_SIZE):
            selected = packed_valid & (target_role == token_type).unsqueeze(0)
            type_ce[token_type] += losses[selected].sum().double()
            type_count[token_type] += selected.sum().double()

        # Teacher-forced pitch accuracy over real body score slots.
        predictions = logits.argmax(dim=-1)
        pitch_slots = (
            (target_packed >= ALTERNATING_START)
            & (target_packed % 6 == 2)
        ).unsqueeze(0) & valid
        pitch_slots = pitch_slots & (labels != REST)
        pitch_hits[0] += ((predictions == labels) & pitch_slots).sum().double()
        pitch_hits[1] += pitch_slots.sum().double()

        # Plan prediction: can the model recover the codes it is handed?
        plan_slots = (target_packed < 0).unsqueeze(0) & valid
        plan_stats[0] += losses[plan_slots].sum().double()
        plan_stats[1] += ((predictions == labels) & plan_slots).sum().double()
        plan_stats[2] += plan_slots.sum().double()

    reduced = accelerator.reduce(
        torch.cat(
            [
                loss_sum.view(1), sample_count.view(1), type_ce, type_count,
                pitch_hits, plan_stats, packed_ce,
            ]
        ),
        reduction="sum",
    )
    loss_sum, sample_count = reduced[0].item(), reduced[1].item()
    type_ce = reduced[2 : 2 + EVENT_SIZE]
    type_count = reduced[2 + EVENT_SIZE : 2 + 2 * EVENT_SIZE]
    pitch_hits = reduced[2 + 2 * EVENT_SIZE : 4 + 2 * EVENT_SIZE]
    plan_stats = reduced[4 + 2 * EVENT_SIZE : 7 + 2 * EVENT_SIZE]
    packed_ce = reduced[7 + 2 * EVENT_SIZE : 9 + 2 * EVENT_SIZE]

    if sample_count == 0:
        raise ValueError("Validation produced zero samples.")

    results = {
        "loss": loss_sum / sample_count,
        "loss_packed": (
            packed_ce[0].item() / packed_ce[1].item() if packed_ce[1] > 0 else float("nan")
        ),
        "loss_by_type": {
            TOKEN_TYPE_NAMES[i]: (
                type_ce[i].item() / type_count[i].item() if type_count[i] > 0 else float("nan")
            )
            for i in range(EVENT_SIZE)
        },
        "teacher_forced_pitch_accuracy": (
            pitch_hits[0].item() / pitch_hits[1].item() if pitch_hits[1] > 0 else 0.0
        ),
        "plan_code_loss": (
            plan_stats[0].item() / plan_stats[2].item() if plan_stats[2] > 0 else float("nan")
        ),
        "plan_code_accuracy": (
            plan_stats[1].item() / plan_stats[2].item() if plan_stats[2] > 0 else 0.0
        ),
    }

    if args.eval_autoregressive_samples <= 0 or args.disable_autoregressive_pitch_eval:
        results["autoregressive"] = None
        results["predicted_plan"] = None
        accelerator.wait_for_everyone()
        model.train()
        return results

    # ---- autoregressive, oracle plan vs the model's own plan ---------------
    ar_loader = _build_random_eval_dataloader(
        dataset=dataset,
        accelerator=accelerator,
        batch_size=args.val_batch_size,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.eval_num_workers,
        requested_size=args.eval_autoregressive_samples,
        description="Autoregressive eval",
    )

    def new_accumulators():
        return [
            torch.zeros(heatmap_slots, dtype=torch.float64, device=accelerator.device)
            for _ in range(6)
        ]

    oracle = new_accumulators()
    predicted = new_accumulators()
    plan_match = torch.zeros(2, device=accelerator.device, dtype=torch.float64)

    for batch in tqdm(ar_loader, desc="Autoregressive eval", leave=False,
                      disable=not accelerator.is_local_main_process):
        batch = _move_batch_to_device(batch, accelerator.device)
        input_ids = batch["input_ids"]
        if input_ids.shape[0] == 0:
            continue
        codes = batcher.codes(batch)

        generated, _ = plan_autoregressive_generate_score(
            model, input_ids, layout, accelerator.device, codes=codes
        )
        _score_slot_stats(generated, input_ids, heatmap_slots, oracle)

        generated_pred, own_codes = plan_autoregressive_generate_score(
            model, input_ids, layout, accelerator.device, generate_plan=True
        )
        _score_slot_stats(generated_pred, input_ids, heatmap_slots, predicted)
        plan_match[0] += (own_codes == codes).sum().double()
        plan_match[1] += codes.numel()

    stacked = accelerator.reduce(
        torch.cat([torch.stack(oracle), torch.stack(predicted), plan_match.view(2, 1).expand(2, heatmap_slots)]),
        reduction="sum",
    )
    oracle_vecs = stacked[:6].cpu().numpy()
    predicted_vecs = stacked[6:12].cpu().numpy()
    plan_match_total = stacked[12, 0].item(), stacked[13, 0].item()

    def summarize(vectors, with_heatmaps):
        pitch_err, onset_err, dur_err, onset_abs, dur_abs, slot_total = vectors
        total = float(slot_total.sum())
        if total <= 0:
            return None

        summary = {
            "pitch_accuracy": 1.0 - float(pitch_err.sum()) / total,
            "onset_accuracy": 1.0 - float(onset_err.sum()) / total,
            "duration_accuracy": 1.0 - float(dur_err.sum()) / total,
            "total_notes": int(total),
        }
        if with_heatmaps:
            with np.errstate(invalid="ignore", divide="ignore"):
                summary["slot_pitch_error_freq"] = np.where(
                    slot_total > 0, pitch_err / slot_total, np.nan
                )
                summary["slot_onset_mae"] = np.where(
                    slot_total > 0, onset_abs / slot_total, np.nan
                )
                summary["slot_duration_mae"] = np.where(
                    slot_total > 0, dur_abs / slot_total, np.nan
                )
        return summary

    results["autoregressive"] = summarize(oracle_vecs, with_heatmaps=True)
    results["predicted_plan"] = summarize(predicted_vecs, with_heatmaps=False)
    results["greedy_plan_code_accuracy"] = (
        plan_match_total[0] / plan_match_total[1] if plan_match_total[1] > 0 else 0.0
    )

    accelerator.wait_for_everyone()
    model.train()
    return results


# ---------------------------------------------------------------------------

def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_file", type=Path, default=Path("data/train_paper.txt"))
    parser.add_argument("--val_file", type=Path, default=Path("data/val_paper.txt"))
    parser.add_argument("--model_name", type=str, default="stanford-crfm/music-medium-800k")
    parser.add_argument("--output_dir", type=Path, default=Path("./run_plan_lm"))

    parser.add_argument("--vq_checkpoint", type=Path, required=True,
                        help="Stage-1 checkpoint, e.g. run_plan_vq/plan_vq_best.pt.")
    parser.add_argument("--pianobart_checkpoint", type=str, default=None,
                        help="Local model.safetensors; downloaded from the Hub when omitted.")
    parser.add_argument("--plan_placement", type=str, default="front",
                        choices=["front", "after_prefix"],
                        help="'front' is the literal spec; 'after_prefix' lets the plan "
                             "condition on the 32 lookahead controls (see plan_lm.py).")

    # everything below matches train.py's defaults
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--final_learning_rate", type=float, default=3e-6)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=2500)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--eval_max_samples", type=int, default=500)
    parser.add_argument("--eval_autoregressive_samples", type=int, default=100)
    parser.add_argument("--disable-autoregressive-pitch-eval", action="store_true", default=False)
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--onset_jitter_std", type=float, default=0.05)
    parser.add_argument("--dur_jitter_range", type=float, default=0.05)
    parser.add_argument("--mask_prob", type=float, default=0.0)
    parser.add_argument("--loss_mask_performance_tokens", action="store_true")
    parser.add_argument("--transpose_range_semitones", type=int, default=12)
    parser.add_argument("--tempo_scale_range", type=float, default=0.2)
    parser.add_argument("--original_weight_l2", type=float, default=1e5)
    parser.add_argument("--max_grad_norm", type=float, default=2.0)

    parser.add_argument("--wandb_project", type=str, default="anticipation-asap")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])
    return parser


def main():
    args = build_parser().parse_args()
    if wandb is None and args.wandb_mode != "disabled":
        print("WARNING: wandb is not installed; falling back to --wandb_mode disabled.")
        args.wandb_mode = "disabled"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision="bf16" if torch.cuda.is_available() else "no",
    )
    if accelerator.is_main_process:
        os.makedirs(args.output_dir, exist_ok=True)
        print(
            "Global effective batch size: "
            f"{args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}"
        )
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

    # ---- the frozen stage-1 encoder ----------------------------------------
    plan_encoder = OnlinePlanEncoder(
        args.vq_checkpoint,
        device=accelerator.device,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        pianobart_checkpoint=args.pianobart_checkpoint,
    )
    layout = PlanLayout(
        num_codes=plan_encoder.num_codes,
        codebook_size=plan_encoder.codebook_size,
        placement=args.plan_placement,
    )
    if accelerator.is_main_process:
        print(
            f"Plan: {layout.num_codes} codes from a {layout.codebook_size}-entry codebook "
            f"({layout.num_codes * math.log2(layout.codebook_size):.0f} bits), "
            f"placement={layout.placement}, sequence length {layout.total_length}"
        )

    # ---- data --------------------------------------------------------------
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
    val_dataset = TokenizedDataset(
        args.val_file,
        loss_mask_performance_tokens=args.loss_mask_performance_tokens,
        is_training=False,
    )
    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Empty training or validation set; check the token files.")

    batcher = PlanBatcher(plan_encoder, layout)
    val_batcher = batcher

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available(),
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    # ---- model -------------------------------------------------------------
    print(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, trust_remote_code=True, use_cache=False
    )
    prepare_plan_model(model, layout, verbose=accelerator.is_main_process)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=1e-6,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)

    base_model = accelerator.unwrap_model(model)
    original_weight_references = {}
    if args.original_weight_l2 > 0:
        original_weight_references, tensors, params, _ = _capture_reference_parameters(base_model)
        if accelerator.is_main_process:
            print(
                f"Original-weight L2: lambda={args.original_weight_l2}, "
                f"{tensors} tensors, {params:,} parameters."
            )

    def lr_lambda(step):
        progress = float(step) / float(max(1, args.max_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = args.final_learning_rate / args.learning_rate
        return floor + (1.0 - floor) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda)

    completed_steps = 0
    heatmap_slots = num_score_slots(PACKED_LENGTH)
    validation_steps, pitch_history, onset_history, duration_history = [], [], [], []
    progress_bar = tqdm(total=args.max_steps, desc="Training", disable=not accelerator.is_main_process)
    training_failed = False

    def run_validation(label):
        validation_step = int(completed_steps)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(f"\nRunning validation at {label}...")

        results = evaluate_plan_model(
            model, val_batcher, accelerator, val_dataset, args, heatmap_slots
        )

        if accelerator.is_main_process:
            log_data = {
                "val/loss": results["loss"],
                # Comparable to run_paper_split_v2's val/loss; val/loss above
                # also counts the plan tokens.
                "val/loss_packed": results["loss_packed"],
                "val/loss_onset": results["loss_by_type"]["onset"],
                "val/loss_duration": results["loss_by_type"]["duration"],
                "val/loss_pitch": results["loss_by_type"]["pitch"],
                "val/teacher_forced_pitch_accuracy": results["teacher_forced_pitch_accuracy"] * 100,
                "val/plan_code_loss": results["plan_code_loss"],
                "val/plan_code_accuracy": results["plan_code_accuracy"] * 100,
            }
            autoregressive = results["autoregressive"]
            message = "(skipped)"
            if autoregressive is not None:
                log_data.update({
                    "val/ar_pitch_accuracy": autoregressive["pitch_accuracy"] * 100,
                    "val/ar_onset_accuracy": autoregressive["onset_accuracy"] * 100,
                    "val/ar_duration_accuracy": autoregressive["duration_accuracy"] * 100,
                })
                predicted = results["predicted_plan"]
                log_data.update({
                    "val/pred_ar_pitch_accuracy": predicted["pitch_accuracy"] * 100,
                    "val/pred_ar_onset_accuracy": predicted["onset_accuracy"] * 100,
                    "val/pred_ar_duration_accuracy": predicted["duration_accuracy"] * 100,
                    "val/greedy_plan_code_accuracy": results["greedy_plan_code_accuracy"] * 100,
                    # The number the whole scheme lives or dies by.
                    "val/plan_onset_gap": (
                        autoregressive["onset_accuracy"] - predicted["onset_accuracy"]
                    ) * 100,
                })
                message = (
                    f"oracle plan: pitch {autoregressive['pitch_accuracy'] * 100:.2f}%, "
                    f"onset {autoregressive['onset_accuracy'] * 100:.2f}%, "
                    f"duration {autoregressive['duration_accuracy'] * 100:.2f}% | "
                    f"own plan: pitch {predicted['pitch_accuracy'] * 100:.2f}%, "
                    f"onset {predicted['onset_accuracy'] * 100:.2f}%, "
                    f"duration {predicted['duration_accuracy'] * 100:.2f}% "
                    f"(codes matched {results['greedy_plan_code_accuracy'] * 100:.1f}%)"
                )

                validation_steps.append(validation_step)
                pitch_history.append(autoregressive["slot_pitch_error_freq"])
                onset_history.append(autoregressive["slot_onset_mae"])
                duration_history.append(autoregressive["slot_duration_mae"])
                for history, key, title, value_label, cmap in (
                    (pitch_history, "pitch_error_freq",
                     "Autoregressive pitch error frequency (oracle plan)",
                     "pitch error frequency", "magma"),
                    (onset_history, "onset_mae",
                     "Autoregressive onset MAE (oracle plan, 10 ms bins)",
                     "onset MAE (bins)", "viridis"),
                    (duration_history, "duration_mae",
                     "Autoregressive duration MAE (oracle plan, 10 ms bins)",
                     "duration MAE (bins)", "viridis"),
                ):
                    image = build_error_heatmap_chart(
                        history, validation_steps, output_dir=args.output_dir,
                        metric_key=key, title=title, value_label=value_label, cmap=cmap,
                    )
                    if use_wandb and image is not None:
                        log_data[f"heatmaps/{key}"] = image

            if use_wandb:
                wandb.log(log_data, step=validation_step)
            print(
                f"Validation Loss: {results['loss']:.4f} "
                f"(onset {results['loss_by_type']['onset']:.4f}, "
                f"duration {results['loss_by_type']['duration']:.4f}, "
                f"pitch {results['loss_by_type']['pitch']:.4f}), "
                f"Teacher-Forced Pitch Accuracy: "
                f"{results['teacher_forced_pitch_accuracy'] * 100:.2f}%, "
                f"plan-code accuracy (teacher forced) "
                f"{results['plan_code_accuracy'] * 100:.2f}%\n  {message}"
            )

        accelerator.wait_for_everyone()
        model.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    def save(directory):
        if accelerator.is_main_process:
            os.makedirs(directory, exist_ok=True)
        accelerator.wait_for_everyone()
        accelerator.unwrap_model(model).save_pretrained(
            directory,
            is_main_process=accelerator.is_main_process,
            save_function=accelerator.save,
        )
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(f"Saved checkpoint to {directory}")
        accelerator.wait_for_everyone()

    model.train()
    try:
        while completed_steps < args.max_steps:
            for batch in train_dataloader:
                with accelerator.accumulate(model):
                    prepared = batcher.build(batch)
                    outputs = forward_plan_batch(model, prepared)
                    loss = outputs.loss
                    l2_penalty = None
                    if original_weight_references:
                        l2_penalty = _compute_original_weight_l2_penalty(
                            base_model, original_weight_references
                        )
                        loss = loss + args.original_weight_l2 * l2_penalty

                    invalid = _count_flagged_processes(
                        accelerator, bool(torch.isnan(loss).any() or torch.isinf(loss).any())
                    )
                    if invalid > 0:
                        if accelerator.is_main_process:
                            print(
                                f"WARNING: NaN/Inf loss on {invalid}/{accelerator.num_processes} "
                                "rank(s); skipping this synchronized step."
                            )
                        optimizer.zero_grad()
                        continue

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        bad_grad = _find_invalid_gradient_parameter(model)
                        bad_ranks = _count_flagged_processes(accelerator, bad_grad is not None)
                        if bad_ranks > 0:
                            if accelerator.is_main_process:
                                print(
                                    f"WARNING: NaN/Inf gradients on {bad_ranks}/"
                                    f"{accelerator.num_processes} rank(s); skipping step."
                                )
                            optimizer.zero_grad()
                            continue

                        accelerator.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                        optimizer.step()
                        scheduler.step()
                        optimizer.zero_grad()

                        reduced_loss = accelerator.reduce(
                            loss.detach().to(accelerator.device, torch.float64), reduction="mean"
                        ).item()
                        completed_steps += 1
                        progress_bar.update(1)

                        if completed_steps % 10 == 0 and accelerator.is_main_process:
                            current_lr = scheduler.get_last_lr()[0]
                            if use_wandb:
                                train_log = {
                                    "train/loss": reduced_loss,
                                    "train/learning_rate": current_lr,
                                }
                                if l2_penalty is not None:
                                    train_log["train/anchor_l2"] = l2_penalty.item()
                                    train_log["train/anchor_term"] = (
                                        args.original_weight_l2 * l2_penalty.item()
                                    )
                                wandb.log(train_log, step=completed_steps)
                            print(
                                f"Step: {completed_steps}/{args.max_steps}, "
                                f"Loss: {reduced_loss:.4f}, LR: {current_lr:.8e}"
                            )

                        is_checkpoint_step = completed_steps % args.save_steps == 0
                        if completed_steps % args.eval_steps == 0 and not is_checkpoint_step:
                            run_validation(f"step {completed_steps}")
                        if is_checkpoint_step:
                            run_validation(f"checkpoint step {completed_steps}")
                            save(args.output_dir / f"checkpoint-{completed_steps}")
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                gc.collect()

                if completed_steps >= args.max_steps:
                    break

    except Exception as error:
        training_failed = True
        print(f"Error during training: {error}")
        print(traceback.format_exc())
        raise
    finally:
        progress_bar.close()
        if training_failed:
            if accelerator.is_main_process:
                print("Skipping final validation/save because training exited with an error.")
        else:
            run_validation(f"final step {completed_steps}")
            save(args.output_dir / "final")
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
