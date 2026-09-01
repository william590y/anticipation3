#!/usr/bin/env python
"""Fine-tune the anticipatory music transformer as a Latent Thought Model.

Each packed window is paired with a continuous plan ``z`` inferred by the
ICML 2025 LTM dual-rate procedure (Kong et al.): fast AdamVI on local
variational parameters, then a slow decoder update of ``p(tokens | z)``. A
performance-conditioned diffusion model is trained on the same batches to
amortise the oracle posterior, which is the only plan available at real
inference time (the score is unknown).

Validation reports three ways of getting ``z``:

  val/oracle/*      q(z | the whole window)     -- Bayesian ceiling
  val/prefix/*      q(z | tokens[:192])         -- 32 lookahead controls
  val/diffusion/*   z ~ diffusion(performance)  -- what inference can run

All three feed the LM a *sample* of a plan, which is what the slow decoder
update is conditioned on.

Launch (4 GPUs, effective batch 64)::

    sbatch train_ltlm.sbatch
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
from packed_dataset import PACKED_SEQUENCE_LENGTH, TokenizedDataset
from plan_vq import split_packed_batch
from train import (
    TOKEN_TYPE_NAMES,
    _accumulate_per_type_teacher_forced,
    _build_random_eval_dataloader,
    _capture_reference_parameters,
    _compute_original_weight_l2_penalty,
    _count_flagged_processes,
    _find_invalid_gradient_parameter,
    _move_batch_to_device,
    build_error_heatmap_chart,
    check_model_for_nans,
    num_score_slots,
    print_gpu_memory_stats,
    report_runtime_device,
    validate_selected_cuda_device_or_raise,
)
from ltlm import (
    LTLMConfig,
    attach_latent_thoughts,
    infer_posterior,
    ltlm_autoregressive_generate_score,
    ltlm_forward,
    ltlm_module_names,
    ones_attention_mask,
    prefix_observation_labels,
    resolve_z_dim,
    save_ltlm_sidecar,
    score_only_labels,
    z_length,
)
from ltlm_diffusion import DiffusionConfig, PlanDiffusion, load_diffusion, save_diffusion

MODES = ("oracle", "prefix", "diffusion")


def collate_fn(batch):
    return {
        key: torch.stack([item[key] for item in batch])
        for key in ("input_ids", "attention_mask", "labels", "score_token_mask", "score_mask")
    }


def _default_token_file(kind):
    """Prefer the paper split when those files exist (the current research path)."""
    paper = {
        "train": Path("data/train_paper.txt"),
        "val": Path("data/val_paper.txt"),
    }[kind]
    fallback = {
        "train": Path("data/train_normalized.txt"),
        "val": Path("data/test_normalized.txt"),
    }[kind]
    return paper if paper.exists() else fallback


def _fast_lr_for_step(step, max_steps, initial, final):
    if max_steps <= 1:
        return final
    frac = min(1.0, max(0.0, step / float(max_steps - 1)))
    return initial + frac * (final - initial)


def posterior_labels(batch, score_only):
    """The observation set the variational posterior explains.

    Controls are conditioning, not output, so with ``score_only`` the ELBO's
    likelihood term covers only the real score triplets.
    """
    if not score_only:
        return batch["labels"]
    return score_only_labels(batch["labels"], batch["score_token_mask"])


def _infer_z(raw_model, diffusion, batch, mode, ltlm_config, steps, fast_lr,
             score_only=False):
    """Return a detached plan ``z`` for this validation/training mode.

    Teacher-forced metrics always score ``p(full window | z)``; only the
    *inference* of ``z`` changes with ``mode``.

    Every mode returns a **sample** from its plan distribution, not a posterior
    mean. The slow decoder update is conditioned on ``z = mu + eps*sigma``, so a
    mean is off-distribution by exactly ``q``'s spread, and until ``log_var``
    collapses that spread is order 1 -- large enough that the "oracle" stopped
    being an upper bound at all (measured at smoke-test step 3: oracle TF loss
    19.12 against 15.39 for a prefix plan that had barely left the prior).
    Sampling everywhere makes the oracle the quantity the inner loop actually
    maximised, and makes the diffusion model's output distribution match what
    the LM was trained to consume.

    Returns ``(z, mu, stats)``. ``mu`` is the *noise-free* plan and exists only
    for the fidelity metrics: comparing samples goes blind once ``log_var``
    approaches 0, because then ``z`` is dominated by ``eps`` and the cosine
    between two arms is the cosine between two independent noise draws --
    ~0 however well the means agree. That is what hid the kl_weight=96
    posterior collapse behind a "diffusion cos 0.001" that looked like the
    kl_weight=1 failure but was not. The diffusion arm has no posterior, so it
    reports its sample for both.
    """
    input_ids = batch["input_ids"]
    attention_mask = batch.get("attention_mask")
    if attention_mask is None:
        attention_mask = ones_attention_mask(input_ids)
    score_mask = batch.get("score_mask")
    if score_mask is not None and not torch.any(score_mask):
        score_mask = None

    if mode == "oracle":
        z, mu, log_var, stats = infer_posterior(
            raw_model, input_ids, posterior_labels(batch, score_only), ltlm_config,
            attention_mask=attention_mask, score_mask=score_mask,
            steps=steps, fast_lr=fast_lr,
        )
        return z, mu, stats
    if mode == "prefix":
        # Observation built from input_ids, not from the training labels:
        # the packed prefix [0, 192) holds 32 controls and 32 dummy rests and
        # no real score triplet, so anything that masks the controls (either
        # --posterior_score_only or --loss_mask_performance_tokens) would leave
        # this arm with nothing to explain and the posterior on the prior.
        # See ltlm.prefix_observation_labels.
        z, mu, log_var, stats = infer_posterior(
            raw_model, input_ids,
            prefix_observation_labels(input_ids, ltlm_config.prefix_length),
            ltlm_config, attention_mask=attention_mask, score_mask=score_mask,
            steps=steps, fast_lr=fast_lr,
        )
        return z, mu, stats
    if mode == "diffusion":
        # One DDIM pass: mu_hat is the bare prediction (what the fidelity
        # metrics compare against the oracle's mu), z is that same draw with q's
        # spread re-added (what the LM consumes). Sampling twice would both
        # double the cost and make the two disagree.
        mu_hat = diffusion.sample(
            input_ids, steps=diffusion.config.sample_steps, add_posterior_noise=False
        )
        z = mu_hat + torch.randn_like(mu_hat) * diffusion.posterior_sigma
        return z, mu_hat, {}
    raise ValueError(f"unknown mode {mode!r}")


def _empty_slot_accums(heatmap_slots, device):
    return [
        torch.zeros(heatmap_slots, dtype=torch.float64, device=device)
        for _ in range(6)
    ]


def _score_slot_stats(generated, reference, heatmap_slots, accums):
    pitch_err, onset_err, dur_err, onset_abs, dur_abs, slot_total = accums
    for slot_idx, pos in enumerate(iter_score_slot_positions(reference.shape[1], ALTERNATING_START)):
        if slot_idx >= heatmap_slots or pos + 2 >= generated.shape[1]:
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
            (torch.abs(generated[:, pos] - reference[:, pos]).float() * real.float()).sum().item()
        )
        dur_abs[slot_idx] += float(
            (torch.abs(generated[:, pos + 1] - reference[:, pos + 1]).float() * real.float()).sum().item()
        )


def _finalize_ar(accums, accelerator):
    stacked = torch.stack(accums)
    stacked = accelerator.reduce(stacked, reduction="sum")
    pitch_err, onset_err, dur_err, onset_abs, dur_abs, slot_total = stacked.cpu().numpy()
    total = float(slot_total.sum())

    def acc(err):
        return 1.0 - (float(err.sum()) / total) if total > 0 else 0.0

    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "pitch_accuracy": acc(pitch_err),
            "onset_accuracy": acc(onset_err),
            "duration_accuracy": acc(dur_err),
            "total_notes": int(total),
            "slot_pitch_error_freq": np.where(slot_total > 0, pitch_err / slot_total, np.nan),
            "slot_onset_mae": np.where(slot_total > 0, onset_abs / slot_total, np.nan),
            "slot_duration_mae": np.where(slot_total > 0, dur_abs / slot_total, np.nan),
        }


@torch.no_grad()
def evaluate_ltlm(
    model,
    diffusion,
    accelerator,
    dataset,
    args,
    ltlm_config,
    heatmap_slots,
):
    """Teacher-forced + AR metrics for oracle / prefix / diffusion plans."""
    model.eval()
    diffusion.eval()
    raw_model = accelerator.unwrap_model(model)
    raw_diff = accelerator.unwrap_model(diffusion)
    device = accelerator.device
    eval_steps = args.eval_mcmc_steps if args.eval_mcmc_steps is not None else ltlm_config.mcmc_steps
    fast_lr = ltlm_config.fast_lr

    pin_memory = torch.cuda.is_available() and not args.force_cpu
    tf_loader = _build_random_eval_dataloader(
        dataset=dataset,
        accelerator=accelerator,
        batch_size=args.val_batch_size,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=args.eval_num_workers,
        requested_size=args.eval_max_samples,
        description="Teacher-forced eval",
    )

    tf_stats = {
        mode: {
            "loss_sum": 0.0,
            "n": 0,
            "pitch_hits": 0,
            "pitch_n": 0,
            "ce_sum": [0.0] * EVENT_SIZE,
            "ce_n": [0] * EVENT_SIZE,
            "kl_sum": 0.0,
            "kl_n": 0,
        }
        for mode in MODES
    }

    # How close each non-oracle plan lands to the oracle plan it approximates.
    # Downstream AR accuracy can hide a broken amortiser (the LM may just be
    # ignoring z), so measure the plan itself as well.
    plan_stats = {
        mode: {"sq_sum": 0.0, "cos_sum": 0.0, "n": 0} for mode in ("prefix", "diffusion")
    }
    oracle_sq_sum = 0.0
    oracle_mu_sq_sum = 0.0

    for batch in tqdm(
        tf_loader, desc="Evaluating", leave=False,
        disable=not accelerator.is_local_main_process,
    ):
        batch = _move_batch_to_device(batch, device)
        z_by_mode = {}
        mu_by_mode = {}
        for mode in MODES:
            z, mu_mode, stats = _infer_z(
                raw_model, raw_diff, batch, mode, ltlm_config, eval_steps, fast_lr,
                score_only=args.posterior_score_only,
            )
            z_by_mode[mode] = z
            mu_by_mode[mode] = mu_mode
            outputs = ltlm_forward(
                raw_model,
                batch["input_ids"],
                z,
                labels=batch["labels"],
                attention_mask=ones_attention_mask(batch["input_ids"]),
                score_mask=batch.get("score_mask"),
            )
            local_bs = batch["input_ids"].size(0)
            bucket = tf_stats[mode]
            bucket["loss_sum"] += float(outputs.loss.item()) * local_bs
            bucket["n"] += local_bs
            if "kl" in stats:
                bucket["kl_sum"] += float(stats["kl"]) * local_bs
                bucket["kl_n"] += local_bs
            for b in range(local_bs):
                _accumulate_per_type_teacher_forced(
                    outputs.logits[b], batch["labels"][b], bucket["ce_sum"], bucket["ce_n"]
                )
                seq = batch["input_ids"][b]
                logits = outputs.logits[b]
                labels = batch["labels"][b]
                for pos in iter_score_slot_positions(seq.size(0), ALTERNATING_START):
                    note_pos = pos + 2
                    if labels[note_pos] == -100 or seq[note_pos] == REST:
                        continue
                    bucket["pitch_n"] += 1
                    if int(logits[note_pos - 1].argmax()) == int(labels[note_pos]):
                        bucket["pitch_hits"] += 1

        # Fidelity is measured on posterior MEANS, not samples. With samples the
        # cosine is between two independent eps draws once log_var -> 0, which
        # reads ~0 no matter how well the plans agree. mu_rms vs z_rms is also
        # the direct collapse indicator: mu_rms -> 0 with z_rms -> 1 is q sitting
        # on the prior.
        oracle_z = z_by_mode["oracle"].float()
        oracle_mu = mu_by_mode["oracle"].float()
        oracle_sq_sum += float(oracle_z.pow(2).mean(dim=(1, 2)).sum())
        oracle_mu_sq_sum += float(oracle_mu.pow(2).mean(dim=(1, 2)).sum())
        for mode in plan_stats:
            other = mu_by_mode[mode].float()
            plan_stats[mode]["sq_sum"] += float((other - oracle_mu).pow(2).mean(dim=(1, 2)).sum())
            plan_stats[mode]["cos_sum"] += float(
                torch.nn.functional.cosine_similarity(
                    other.flatten(1), oracle_mu.flatten(1), dim=1
                ).sum()
            )
            plan_stats[mode]["n"] += oracle_mu.shape[0]

    for mode in list(plan_stats):
        packed = torch.tensor(
            [plan_stats[mode]["sq_sum"], plan_stats[mode]["cos_sum"],
             plan_stats[mode]["n"], oracle_sq_sum, oracle_mu_sq_sum],
            device=device, dtype=torch.float64,
        )
        sq_sum, cos_sum, n, oracle_sq, oracle_mu_sq = [
            float(x) for x in accelerator.reduce(packed, reduction="sum")
        ]
        plan_stats[mode] = {
            # RMSE is on the same per-element scale as oracle_mu_rms, so read
            # them together: rmse >= mu_rms means no oracle signal recovered.
            "z_rmse_vs_oracle": math.sqrt(sq_sum / max(n, 1.0)),
            "z_cosine_vs_oracle": cos_sum / max(n, 1.0),
            "oracle_z_rms": math.sqrt(oracle_sq / max(n, 1.0)),
            # mu_rms ~ 0 while z_rms ~ 1 is posterior collapse: q is the prior.
            "oracle_mu_rms": math.sqrt(oracle_mu_sq / max(n, 1.0)),
        }

    results = {"plan_fidelity": plan_stats}
    for mode in MODES:
        bucket = tf_stats[mode]
        packed = torch.tensor(
            [bucket["loss_sum"], bucket["n"], bucket["pitch_hits"], bucket["pitch_n"],
             bucket["kl_sum"], bucket["kl_n"]]
            + bucket["ce_sum"]
            + [float(c) for c in bucket["ce_n"]],
            device=device, dtype=torch.float64,
        )
        packed = accelerator.reduce(packed, reduction="sum")
        loss_sum, n, hits, pitch_n, kl_sum, kl_n = [float(x) for x in packed[:6]]
        ce_sum = [float(packed[6 + i]) for i in range(EVENT_SIZE)]
        ce_n = [int(packed[6 + EVENT_SIZE + i]) for i in range(EVENT_SIZE)]
        results[mode] = {
            "loss": loss_sum / max(n, 1.0),
            "teacher_forced_pitch_accuracy": hits / max(pitch_n, 1.0),
            "kl": kl_sum / max(kl_n, 1.0) if kl_n > 0 else float("nan"),
            "loss_by_type": {
                TOKEN_TYPE_NAMES[i]: (ce_sum[i] / ce_n[i] if ce_n[i] > 0 else float("nan"))
                for i in range(EVENT_SIZE)
            },
            "autoregressive": None,
        }

    if args.disable_autoregressive_pitch_eval or args.eval_autoregressive_samples <= 0:
        accelerator.wait_for_everyone()
        return results

    ar_loader = _build_random_eval_dataloader(
        dataset=dataset,
        accelerator=accelerator,
        batch_size=args.val_batch_size,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        num_workers=args.eval_num_workers,
        requested_size=args.eval_autoregressive_samples,
        description="Autoregressive eval",
    )
    ar_accums = {mode: _empty_slot_accums(heatmap_slots, device) for mode in MODES}

    for batch in tqdm(
        ar_loader, desc="Autoregressive eval", leave=False,
        disable=not accelerator.is_local_main_process,
    ):
        batch = _move_batch_to_device(batch, device)
        input_ids = batch["input_ids"]
        if input_ids.shape[0] == 0 or input_ids.shape[1] <= ALTERNATING_START:
            continue
        for mode in MODES:
            z, _, _ = _infer_z(
                raw_model, raw_diff, batch, mode, ltlm_config, eval_steps, fast_lr,
                score_only=args.posterior_score_only,
            )
            generated = ltlm_autoregressive_generate_score(
                raw_model, input_ids, z, device,
                constrain_score_tokens=True,
                ground_truth_score_tokens_to_feed=0,
            )
            _score_slot_stats(generated, input_ids, heatmap_slots, ar_accums[mode])

    for mode in MODES:
        results[mode]["autoregressive"] = _finalize_ar(ar_accums[mode], accelerator)

    accelerator.wait_for_everyone()
    return results


def _fixed_piano_roll_indices(dataset, count):
    """First ``count`` validation windows, in file order — same every eval.

    Matches ``train_plan_vq.log_piano_rolls`` (``val_dataset[0..N)``), so the
    wandb images are a time series of the *same* music, not a new random
    sample each checkpoint.
    """
    if count is None or count <= 0 or len(dataset) == 0:
        return []
    return list(range(min(int(count), len(dataset))))


def _stack_val_windows(dataset, indices, device):
    keys = ("input_ids", "attention_mask", "labels", "score_token_mask", "score_mask")
    batch = {
        key: torch.stack([dataset[i][key] for i in indices])
        for key in keys
    }
    return _move_batch_to_device(batch, device)


def render_ltlm_piano_rolls(
    raw_model,
    raw_diff,
    windows_batch,
    indices,
    ltlm_config,
    args,
    step,
):
    """One (3 windows x 3 columns) figure per plan mode. Rank 0 only.

    Generated score is the same greedy AR decode as ``val/<mode>/ar_*``:
    ``ltlm_autoregressive_generate_score`` with ground-truth controls
    teacher-forced, conditioned on that mode's ``z``. Drawing is delegated
    to ``plan_vq_viz.piano_roll_figure`` (green exact, red mismatch, GT
    ghosted grey).
    """
    from plan_vq_viz import piano_roll_figure

    eval_steps = args.eval_mcmc_steps if args.eval_mcmc_steps is not None else ltlm_config.mcmc_steps
    fast_lr = ltlm_config.fast_lr
    device = windows_batch["input_ids"].device
    figures = {}
    for mode in MODES:
        z, _, _ = _infer_z(
            raw_model, raw_diff, windows_batch, mode, ltlm_config, eval_steps, fast_lr,
            score_only=args.posterior_score_only,
        )
        generated = ltlm_autoregressive_generate_score(
            raw_model,
            windows_batch["input_ids"],
            z,
            device,
            constrain_score_tokens=True,
            ground_truth_score_tokens_to_feed=0,
        )
        perf, score, valid = split_packed_batch(windows_batch["input_ids"])
        _, predicted, _ = split_packed_batch(generated)
        figures[mode] = piano_roll_figure(
            perf,
            score,
            valid,
            predicted,
            step=step,
            max_examples=len(indices),
            suptitle=f"LTLM AR decode ({mode})",
            pred_heading=f"generated score ({mode})",
            window_ids=indices,
        )
    return figures


def _save_checkpoint(accelerator, model, diffusion, directory):
    directory = Path(directory)
    if accelerator.is_main_process:
        directory.mkdir(parents=True, exist_ok=True)
    accelerator.wait_for_everyone()
    raw_model = accelerator.unwrap_model(model)
    raw_diff = accelerator.unwrap_model(diffusion)
    raw_model.save_pretrained(
        directory / "lm",
        is_main_process=accelerator.is_main_process,
        save_function=accelerator.save,
    )
    if accelerator.is_main_process:
        save_ltlm_sidecar(raw_model, directory / "lm")
        torch.save(raw_model.state_dict(), directory / "ltlm_state.pt")
        save_diffusion(raw_diff, directory / "diffusion.pt")
        print(f"Saved checkpoint to {directory}")
    accelerator.wait_for_everyone()


def parse_args():
    parser = argparse.ArgumentParser(description="LTLM fine-tune of the packed ASAP transformer")
    parser.add_argument("--data_file", type=Path, default=None)
    parser.add_argument("--val_file", type=Path, default=None)
    parser.add_argument("--model_name", type=str, default="stanford-crfm/music-medium-800k")
    parser.add_argument("--output_dir", type=Path, default=Path("./run_ltlm"))
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--final_learning_rate", type=float, default=3e-6)
    parser.add_argument("--diffusion_lr", type=float, default=1e-4,
                        help="Denoiser LR. Not the LM's 3e-5: that is a fine-tune "
                             "LR for a pretrained 355M model, while this one trains "
                             "from scratch. 1e-4 is ADM/DiT's value for a "
                             "transformer of this width (d_model 1024); the old "
                             "3e-4 default was calibrated when it was 256 wide.")
    parser.add_argument("--diffusion_warmup_steps", type=int, default=500,
                        help="Linear warmup for the denoiser, then constant. The "
                             "LM needs no warmup (it is pretrained) and should "
                             "not be constant (it converges); the denoiser is the "
                             "other way round on both counts.")
    parser.add_argument("--diffusion_weight", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=20000)
    parser.add_argument("--save_steps", type=int, default=2500)
    parser.add_argument("--eval_steps", type=int, default=1000)
    parser.add_argument("--eval_max_samples", type=int, default=500)
    parser.add_argument("--eval_autoregressive_samples", type=int, default=100)
    parser.add_argument(
        "--piano_roll_examples",
        type=int,
        default=3,
        help="Fixed validation windows drawn as piano rolls each eval "
             "(first N of val_file, same every step). 0 disables.",
    )
    parser.add_argument("--disable-autoregressive-pitch-eval", action="store_true", default=False)
    parser.add_argument("--eval_num_workers", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--reduce_memory", action="store_true")
    parser.add_argument("--onset_jitter_std", type=float, default=0.05)
    parser.add_argument("--dur_jitter_range", type=float, default=0.05)
    parser.add_argument("--mask_prob", type=float, default=0.00)
    parser.add_argument("--loss_mask_performance_tokens", action="store_true")
    parser.add_argument("--transpose_range_semitones", type=int, default=12)
    parser.add_argument("--tempo_scale_range", type=float, default=0.2)
    parser.add_argument("--original_weight_l2", type=float, default=1e5)
    parser.add_argument("--max_grad_norm", type=float, default=2.0)
    parser.add_argument("--wandb_project", type=str, default="anticipation-asap")
    parser.add_argument("--wandb_run_name", type=str, default=None)
    parser.add_argument("--wandb_mode", type=str, default="online",
                        choices=["online", "offline", "disabled"])

    parser.add_argument("--thoughts_per_layer", type=int, default=4,
                        help="Latent thought vectors per GPT-2 layer (paper: 8).")
    parser.add_argument("--z_dim", type=int, default=0,
                        help="Thought vector dim; 0 uses the GPT-2 hidden size.")
    parser.add_argument("--mcmc_steps", type=int, default=16,
                        help="Fast-learning / AdamVI steps T_fast (paper: 16).")
    parser.add_argument("--eval_mcmc_steps", type=int, default=None,
                        help="Override T_fast at validation. Default: --mcmc_steps.")
    parser.add_argument("--inference_method", type=str, default="adamVI",
                        choices=["adamVI", "langevin"])
    parser.add_argument("--fast_lr", type=float, default=0.3)
    parser.add_argument("--final_fast_lr", type=float, default=0.34)
    parser.add_argument("--kl_weight", type=float, default=1.0)
    parser.add_argument("--elbo_reduction", type=str, default="mean", choices=["mean", "sum"])
    parser.add_argument("--langevin_step_size", type=float, default=0.1)
    parser.add_argument("--posterior_score_only", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Restrict the variational likelihood to the real score "
                             "triplets, on the argument that controls are "
                             "conditioning rather than something the model "
                             "generates. Off by default, matching train.py's "
                             "--loss_mask_performance_tokens: the ELBO then covers "
                             "the whole window, as in the paper. Applies to "
                             "training and the oracle mode; the prefix mode is "
                             "exempt either way (its prefix holds no real score "
                             "triplet -- see ltlm.prefix_observation_labels).")
    parser.add_argument("--prefix_length", type=int, default=ALTERNATING_START,
                        help="Packed-token prefix used for val/prefix posterior. "
                             "Default ALTERNATING_START=192 (32 lookahead controls).")

    parser.add_argument("--diffusion_hidden", type=int, default=1024,
                        help="Denoiser width. Must not be below --z_dim (the LM "
                             "hidden size when 0): z_out is one linear, so a "
                             "narrower model cannot represent the full epsilon.")
    parser.add_argument("--diffusion_layers", type=int, default=4)
    parser.add_argument("--diffusion_heads", type=int, default=8)
    parser.add_argument("--diffusion_perf_layers", type=int, default=2)
    parser.add_argument("--diffusion_steps", type=int, default=1000)
    parser.add_argument("--diffusion_sample_steps", type=int, default=50)
    parser.add_argument("--diffusion_dropout", type=float, default=0.0)
    parser.add_argument("--init_checkpoint", type=Path, default=None,
                        help="Optional LTLM checkpoint dir (ltlm_state.pt + diffusion.pt) to resume.")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.data_file is None:
        args.data_file = _default_token_file("train")
    if args.val_file is None:
        args.val_file = _default_token_file("val")
    if args.original_weight_l2 < 0:
        raise ValueError("--original_weight_l2 must be non-negative.")
    if wandb is None and args.wandb_mode != "disabled":
        print("WARNING: wandb is not installed; falling back to --wandb_mode disabled.")
        args.wandb_mode = "disabled"

    validate_selected_cuda_device_or_raise(force_cpu=args.force_cpu)
    report_runtime_device(force_cpu=args.force_cpu)
    print(
        f"Per-rank effective batch size: {args.batch_size * args.gradient_accumulation_steps}"
    )
    print(f"Data: train={args.data_file} val={args.val_file}")
    oracle_scope = (
        "the window's real score triplets (--posterior_score_only)"
        if args.posterior_score_only else "the full window"
    )
    decoder_scope = (
        "score triplets only (--loss_mask_performance_tokens)"
        if args.loss_mask_performance_tokens else "the full window"
    )
    print(
        f"Oracle posterior likelihood: {oracle_scope}.\n"
        f"Decoder train/val loss: {decoder_scope}.\n"
        f"Prefix posterior: packed tokens [0, {args.prefix_length}) = the 32 "
        "lookahead (control, dummy-rest) pairs, read from input_ids so neither "
        "mask can empty it.\n"
        "Diffusion: maps the 170 performance notes to the oracle plan sample z."
    )

    mixed_precision = "bf16" if torch.cuda.is_available() and not args.force_cpu else "no"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        cpu=args.force_cpu,
        mixed_precision=mixed_precision,
    )
    print(
        "Distributed setup: "
        f"type={accelerator.distributed_type}, world_size={accelerator.num_processes}, "
        f"rank={accelerator.process_index}, device={accelerator.device}"
    )
    if accelerator.is_main_process:
        print(
            "Global effective batch size: "
            f"{args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}"
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

    print("Initial GPU memory stats:")
    print_gpu_memory_stats()

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
    if train_dataset.sequence_length and train_dataset.sequence_length != PACKED_SEQUENCE_LENGTH:
        print(
            f"Warning: training sequence length {train_dataset.sequence_length} "
            f"!= {PACKED_SEQUENCE_LENGTH}"
        )
    piano_roll_indices = _fixed_piano_roll_indices(val_dataset, args.piano_roll_examples)
    if accelerator.is_main_process:
        if piano_roll_indices:
            print(
                f"Piano rolls: fixed val windows {piano_roll_indices} "
                f"(first {len(piano_roll_indices)} of {args.val_file}, every eval)."
            )
        else:
            print("Piano rolls disabled (--piano_roll_examples 0).")

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available() and not args.force_cpu,
        num_workers=args.num_workers,
        persistent_workers=args.num_workers > 0,
    )

    print(f"Loading model {args.model_name}...")
    model_kwargs = {"trust_remote_code": True, "use_cache": False}
    if args.reduce_memory and torch.cuda.is_available():
        model_kwargs["torch_dtype"] = torch.bfloat16
        model_kwargs["low_cpu_mem_usage"] = True
    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)

    ltlm_config = LTLMConfig(
        thoughts_per_layer=args.thoughts_per_layer,
        z_dim=args.z_dim,
        mcmc_steps=args.mcmc_steps,
        inference_method=args.inference_method,
        fast_lr=args.fast_lr,
        final_fast_lr=args.final_fast_lr,
        kl_weight=args.kl_weight,
        elbo_reduction=args.elbo_reduction,
        langevin_step_size=args.langevin_step_size,
        prefix_length=args.prefix_length,
    )
    attach_latent_thoughts(model, ltlm_config, verbose=accelerator.is_main_process)
    z_dim = resolve_z_dim(model, ltlm_config)
    n_thoughts = z_length(model, ltlm_config)
    if args.diffusion_hidden < z_dim:
        raise ValueError(
            f"--diffusion_hidden {args.diffusion_hidden} < z_dim {z_dim}: the "
            "denoiser's single z_out linear would confine the epsilon prediction "
            "to a lower-dimensional subspace and DDIM would carry the rest of "
            "the noise into the sampled plan."
        )

    diffusion = PlanDiffusion(DiffusionConfig(
        z_dim=z_dim,
        n_thoughts=n_thoughts,
        d_model=args.diffusion_hidden,
        n_head=args.diffusion_heads,
        n_layer=args.diffusion_layers,
        perf_layers=args.diffusion_perf_layers,
        dropout=args.diffusion_dropout,
        timesteps=args.diffusion_steps,
        sample_steps=args.diffusion_sample_steps,
    ))

    if args.init_checkpoint is not None:
        ckpt = Path(args.init_checkpoint)
        state_path = ckpt / "ltlm_state.pt"
        diff_path = ckpt / "diffusion.pt"
        if state_path.is_file():
            payload = torch.load(state_path, map_location="cpu", weights_only=False)
            model.load_state_dict(payload)
            print(f"Loaded LTLM weights from {state_path}")
        if diff_path.is_file():
            loaded = load_diffusion(diff_path, device="cpu")
            diffusion.load_state_dict(loaded.state_dict())
            print(f"Loaded diffusion weights from {diff_path}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        eps=1e-6,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )
    diff_optimizer = AdamW(
        diffusion.parameters(),
        lr=args.diffusion_lr,
        eps=1e-6,
        weight_decay=0.01,
        betas=(0.9, 0.999),
    )

    model, diffusion, optimizer, diff_optimizer, train_dataloader = accelerator.prepare(
        model, diffusion, optimizer, diff_optimizer, train_dataloader
    )
    base_model = accelerator.unwrap_model(model)

    original_weight_references = {}
    if args.original_weight_l2 > 0:
        refs, tensors, params, _ = _capture_reference_parameters(base_model)
        skip = set(ltlm_module_names(base_model))
        original_weight_references = {k: v for k, v in refs.items() if k not in skip}
        if accelerator.is_main_process:
            print(
                f"Original-weight L2: lambda={args.original_weight_l2}, "
                f"{len(original_weight_references)} pretrained tensors "
                f"(cross-attn / thought embeddings are NOT anchored)."
            )
    elif accelerator.is_main_process:
        print("Original-weight L2 regularization disabled.")

    def lr_lambda(step):
        progress = float(step) / float(max(1, args.max_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = args.final_learning_rate / args.learning_rate
        return floor + (1.0 - floor) * cosine

    scheduler = LambdaLR(optimizer, lr_lambda)

    def diff_lr_lambda(step):
        """Linear warmup, then constant -- deliberately not the LM's cosine.

        Two reasons the decoder's schedule is wrong for this model:

        * It is trained **from scratch**, not fine-tuned, so it needs a warmup
          the pretrained LM does not (AdamW at full LR on step 0 of a fresh
          transformer is a standard source of early divergence).
        * Its target is **non-stationary**: the oracle posterior moves every
          time the decoder does, so there is no point at which the diffusion has
          converged and should anneal. Cosining it to 10% would leave it unable
          to track a target that is still drifting at step 19000. Constant-after-
          warmup is the usual diffusion-training recipe (ADM, DiT) for the same
          reason.
        """
        warmup = max(0, int(args.diffusion_warmup_steps))
        if warmup and step < warmup:
            return float(step + 1) / float(warmup)
        return 1.0

    diff_scheduler = LambdaLR(diff_optimizer, diff_lr_lambda)

    completed_steps = 0
    heatmap_slots = num_score_slots(val_dataset.sequence_length or PACKED_SEQUENCE_LENGTH)
    validation_steps, pitch_history, onset_history, duration_history = [], [], [], []
    progress_bar = tqdm(total=args.max_steps, desc="Training", disable=not accelerator.is_main_process)
    training_failed = False

    def run_validation(label):
        validation_step = int(completed_steps)
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            print(f"\nRunning validation at {label}...")

        results = evaluate_ltlm(
            model, diffusion, accelerator, val_dataset, args, ltlm_config, heatmap_slots
        )

        if accelerator.is_main_process:
            log_data = {}
            parts = []
            for mode in MODES:
                block = results[mode]
                log_data[f"val/{mode}/loss"] = block["loss"]
                log_data[f"val/{mode}/teacher_forced_pitch_accuracy"] = (
                    block["teacher_forced_pitch_accuracy"] * 100
                )
                log_data[f"val/{mode}/loss_onset"] = block["loss_by_type"]["onset"]
                log_data[f"val/{mode}/loss_duration"] = block["loss_by_type"]["duration"]
                log_data[f"val/{mode}/loss_pitch"] = block["loss_by_type"]["pitch"]
                if block["kl"] == block["kl"]:  # not NaN
                    log_data[f"val/{mode}/kl"] = block["kl"]
                ar = block["autoregressive"]
                if ar is not None:
                    log_data[f"val/{mode}/ar_pitch_accuracy"] = ar["pitch_accuracy"] * 100
                    log_data[f"val/{mode}/ar_onset_accuracy"] = ar["onset_accuracy"] * 100
                    log_data[f"val/{mode}/ar_duration_accuracy"] = ar["duration_accuracy"] * 100
                    parts.append(
                        f"{mode}: pitch {ar['pitch_accuracy'] * 100:.2f}%, "
                        f"onset {ar['onset_accuracy'] * 100:.2f}%, "
                        f"duration {ar['duration_accuracy'] * 100:.2f}%"
                    )
            for mode, fidelity in results.get("plan_fidelity", {}).items():
                log_data[f"val/{mode}/z_rmse_vs_oracle"] = fidelity["z_rmse_vs_oracle"]
                log_data[f"val/{mode}/z_cosine_vs_oracle"] = fidelity["z_cosine_vs_oracle"]
            if "plan_fidelity" in results:
                log_data["val/oracle/z_rms"] = (
                    results["plan_fidelity"]["diffusion"]["oracle_z_rms"]
                )
                log_data["val/oracle/mu_rms"] = (
                    results["plan_fidelity"]["diffusion"]["oracle_mu_rms"]
                )

            # train.py-shaped aliases for the oracle ceiling, so existing
            # dashboards still have val/loss and val/ar_*.
            log_data["val/loss"] = results["oracle"]["loss"]
            oracle_ar = results["oracle"]["autoregressive"]
            if oracle_ar is not None:
                log_data["val/ar_pitch_accuracy"] = oracle_ar["pitch_accuracy"] * 100
                log_data["val/ar_onset_accuracy"] = oracle_ar["onset_accuracy"] * 100
                log_data["val/ar_duration_accuracy"] = oracle_ar["duration_accuracy"] * 100
                validation_steps.append(validation_step)
                pitch_history.append(oracle_ar["slot_pitch_error_freq"])
                onset_history.append(oracle_ar["slot_onset_mae"])
                duration_history.append(oracle_ar["slot_duration_mae"])
                for history, key, title, value_label, cmap in (
                    (pitch_history, "pitch_error_freq",
                     "Autoregressive pitch error frequency (oracle z)",
                     "pitch error frequency", "magma"),
                    (onset_history, "onset_mae",
                     "Autoregressive onset MAE (oracle z, 10 ms bins)",
                     "onset MAE (bins)", "viridis"),
                    (duration_history, "duration_mae",
                     "Autoregressive duration MAE (oracle z, 10 ms bins)",
                     "duration MAE (bins)", "viridis"),
                ):
                    image = build_error_heatmap_chart(
                        history, validation_steps, output_dir=args.output_dir,
                        metric_key=key, title=title, value_label=value_label, cmap=cmap,
                    )
                    if use_wandb and image is not None:
                        log_data[f"heatmaps/{key}"] = image

            if piano_roll_indices:
                try:
                    import matplotlib.pyplot as plt

                    windows = _stack_val_windows(
                        val_dataset, piano_roll_indices, accelerator.device
                    )
                    figures = render_ltlm_piano_rolls(
                        accelerator.unwrap_model(model),
                        accelerator.unwrap_model(diffusion),
                        windows,
                        piano_roll_indices,
                        ltlm_config,
                        args,
                        validation_step,
                    )
                    roll_dir = args.output_dir / "piano_rolls"
                    roll_dir.mkdir(parents=True, exist_ok=True)
                    for mode, figure in figures.items():
                        figure.savefig(
                            roll_dir / f"{mode}_step{validation_step:06d}.png", dpi=110
                        )
                        if use_wandb:
                            log_data[f"val/{mode}/piano_roll"] = wandb.Image(figure)
                    plt.close("all")
                except Exception as viz_error:
                    print(f"WARNING: piano-roll logging failed: {viz_error}")
                    print(traceback.format_exc())

            if use_wandb:
                wandb.log(log_data, step=validation_step)
            ar_msg = " | ".join(parts) if parts else "(AR skipped)"
            print(
                f"Validation (oracle TF loss {results['oracle']['loss']:.4f}, "
                f"prefix {results['prefix']['loss']:.4f}, "
                f"diffusion {results['diffusion']['loss']:.4f})\n  {ar_msg}"
            )
            fidelity = results.get("plan_fidelity")
            if fidelity:
                print(
                    f"  plan vs oracle (mu_rms {fidelity['diffusion']['oracle_mu_rms']:.4f}, "
                    f"z_rms {fidelity['diffusion']['oracle_z_rms']:.4f}): "
                    + ", ".join(
                        f"{mode} rmse {block['z_rmse_vs_oracle']:.4f} "
                        f"cos {block['z_cosine_vs_oracle']:.3f}"
                        for mode, block in fidelity.items()
                    )
                )

        accelerator.wait_for_everyone()
        model.train()
        diffusion.train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    model.train()
    diffusion.train()
    try:
        while completed_steps < args.max_steps:
            for batch in train_dataloader:
                # Both models accumulate together: the diffusion loss is part of
                # the same backward, so its no_sync window has to match the LM's.
                with accelerator.accumulate(model, diffusion):
                    batch = _move_batch_to_device(batch, accelerator.device)
                    raw_model = accelerator.unwrap_model(model)
                    raw_diff = accelerator.unwrap_model(diffusion)
                    fast_lr = _fast_lr_for_step(
                        completed_steps, args.max_steps, args.fast_lr, args.final_fast_lr
                    )
                    z, mu, log_var, mcmc_stats = infer_posterior(
                        raw_model,
                        batch["input_ids"],
                        posterior_labels(batch, args.posterior_score_only),
                        ltlm_config,
                        attention_mask=ones_attention_mask(batch["input_ids"]),
                        score_mask=batch.get("score_mask") if torch.any(batch["score_mask"]) else None,
                        steps=args.mcmc_steps,
                        fast_lr=fast_lr,
                    )

                    outputs = ltlm_forward(
                        model,
                        batch["input_ids"],
                        z,
                        labels=batch["labels"],
                        attention_mask=ones_attention_mask(batch["input_ids"]),
                        score_mask=batch.get("score_mask") if torch.any(batch["score_mask"]) else None,
                    )
                    lm_loss = outputs.loss
                    l2_penalty = None
                    if original_weight_references:
                        l2_penalty = _compute_original_weight_l2_penalty(
                            base_model, original_weight_references
                        )
                        lm_loss = lm_loss + args.original_weight_l2 * l2_penalty

                    # Through the DDP wrapper, not raw_diff.loss: see
                    # PlanDiffusion.forward for why bypassing it desynchronises
                    # the ranks. mu is already detached, so no gradient crosses
                    # back into the LM.
                    #
                    # The target is mu, NOT the sample z the LM is conditioned
                    # on. eps is independent of the window and of the
                    # performance, so a denoiser trained on mu + eps*sigma
                    # spends nearly all its gradient on noise it cannot predict:
                    # measured, only 8.2% of the sample target's variance was
                    # signal at kl_weight=10 and 1.35% at kl_weight=30.
                    # PlanDiffusion.sample re-adds eps*posterior_sigma so the LM
                    # still receives a z-shaped object.
                    raw_diff.update_posterior_sigma(log_var)
                    diff_loss = diffusion(mu, batch["input_ids"])
                    loss = lm_loss + args.diffusion_weight * diff_loss

                    invalid = _count_flagged_processes(
                        accelerator,
                        bool(torch.isnan(loss).any() or torch.isinf(loss).any()),
                    )
                    if invalid > 0:
                        if accelerator.is_main_process:
                            print(
                                f"WARNING: NaN/Inf loss on {invalid}/{accelerator.num_processes} "
                                "rank(s); skipping this synchronized step."
                            )
                        optimizer.zero_grad()
                        diff_optimizer.zero_grad()
                        continue

                    accelerator.backward(loss)

                    if accelerator.sync_gradients:
                        bad_grad = _find_invalid_gradient_parameter(model)
                        bad_diff = _find_invalid_gradient_parameter(diffusion)
                        bad_ranks = _count_flagged_processes(
                            accelerator, bad_grad is not None or bad_diff is not None
                        )
                        if bad_ranks > 0:
                            if accelerator.is_main_process:
                                print(
                                    f"WARNING: NaN/Inf gradients on {bad_ranks}/"
                                    f"{accelerator.num_processes} rank(s); skipping step."
                                )
                            optimizer.zero_grad()
                            diff_optimizer.zero_grad()
                            continue

                        accelerator.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
                        accelerator.clip_grad_norm_(
                            diffusion.parameters(), max_norm=args.max_grad_norm
                        )
                        optimizer.step()
                        diff_optimizer.step()
                        scheduler.step()
                        diff_scheduler.step()
                        optimizer.zero_grad()
                        diff_optimizer.zero_grad()

                        reduced_loss = accelerator.reduce(
                            lm_loss.detach().to(accelerator.device, torch.float64),
                            reduction="mean",
                        ).item()
                        reduced_diff = accelerator.reduce(
                            diff_loss.detach().to(accelerator.device, torch.float64),
                            reduction="mean",
                        ).item()
                        completed_steps += 1
                        progress_bar.update(1)

                        if completed_steps % 10 == 0 and accelerator.is_main_process:
                            current_lr = scheduler.get_last_lr()[0]
                            if use_wandb:
                                train_log = {
                                    "train/loss": reduced_loss,
                                    "train/diffusion_loss": reduced_diff,
                                    "train/learning_rate": current_lr,
                                    "train/diffusion_lr": diff_scheduler.get_last_lr()[0],
                                    "train/fast_lr": fast_lr,
                                    "train/mcmc_nll": mcmc_stats["nll"],
                                    "train/mcmc_kl": mcmc_stats["kl"],
                                    "train/z_norm": mcmc_stats["z_norm"],
                                    "train/mu_norm": mcmc_stats["mu_norm"],
                                    "train/log_var_mean": mcmc_stats["log_var_mean"],
                                }
                                if l2_penalty is not None:
                                    train_log["train/anchor_l2"] = l2_penalty.item()
                                    train_log["train/anchor_term"] = (
                                        args.original_weight_l2 * l2_penalty.item()
                                    )
                                wandb.log(train_log, step=completed_steps)
                            print(
                                f"Step: {completed_steps}/{args.max_steps}, "
                                f"Loss: {reduced_loss:.4f}, Diff: {reduced_diff:.4f}, "
                                f"KL: {mcmc_stats['kl']:.4f}, LR: {current_lr:.8e}"
                            )
                            if check_model_for_nans(model):
                                print("NaN parameters detected in LM.")
                            if completed_steps % 100 == 0:
                                print_gpu_memory_stats()

                        is_checkpoint_step = completed_steps % args.save_steps == 0
                        if completed_steps % args.eval_steps == 0 and not is_checkpoint_step:
                            run_validation(f"step {completed_steps}")
                        if is_checkpoint_step:
                            run_validation(f"checkpoint step {completed_steps}")
                            _save_checkpoint(
                                accelerator, model, diffusion,
                                args.output_dir / f"checkpoint-{completed_steps}",
                            )
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
            _save_checkpoint(accelerator, model, diffusion, args.output_dir / "final")
        if use_wandb:
            wandb.finish()


if __name__ == "__main__":
    main()
