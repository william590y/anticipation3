"""Train an LTLM where the oracle posterior is regularized toward p(z|P).

Loss
----
    L = E_{q(z|P,S)} [ -log p_theta(S | P, z) ] + beta * D(q(z|P,S), p_phi(z|P))

p_phi is a performance-conditional diffusion by default (D = DDPM bound on
-log p_phi(z|P)). --planner gaussian keeps the analytic KL parameterization.

Fast inner loop (AdamVI) fits q with p_phi held fixed. The slow step updates
the decoder and the planner. beta=0 recovers an unconstrained oracle; larger
beta forces thoughts to be recoverable from performance.

Speed: torch.compile + SDPA flash kernels on the teacher-forced window.
TensorRT and speculative decoding are not applicable (TensorRT has no
training backward through this graph; speculative decoding is for token-by-token
AR generation, not 1020-token teacher-forced F+B).

Launch::

    python train_ltlm.py \\
        --data_file data/train_paper.txt --val_file data/val_paper.txt \\
        --output_dir ./run_ltlm_diffusion --wandb_run_name ltlm_diffusion \\
        --planner diffusion --kl_weight 5 --thoughts_per_layer 4 --mcmc_steps 16 \\
        --batch_size 4 --gradient_accumulation_steps 4 --max_steps 20000
"""

from __future__ import annotations

import argparse
import math
import os
from datetime import timedelta
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm
from transformers import AutoModelForCausalLM

try:
    import wandb
except ImportError:
    wandb = None

from accelerate import Accelerator, InitProcessGroupKwargs

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
    parser.add_argument("--output_dir", type=Path, default=Path("./run_ltlm_diffusion"))
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
        help="beta in D(q(z|P,S), p_phi(z|P)). 0 = unconstrained oracle.",
    )
    parser.add_argument(
        "--planner",
        type=str,
        default="diffusion",
        choices=["diffusion", "gaussian"],
        help="Parameterization of p_phi(z|P). diffusion uses the DDPM bound as D.",
    )
    parser.add_argument("--diffusion_timesteps", type=int, default=1000)
    parser.add_argument("--diffusion_layers", type=int, default=3)
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="DDIM steps when sampling z ~ p_phi(z|P) for the planner eval path.",
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
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="torch.compile the LTLM. TensorRT / speculative decoding do not apply "
        "(no training backward through TensorRT; this loop is teacher-forced, not AR decode).",
    )
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune"],
        help="Inductor mode. default fuses kernels without CUDA-graph address constraints.",
    )
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        help="GPT-2 attention kernel. sdpa uses PyTorch flash/mem-efficient kernels on Ada.",
    )
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


# Job 69260: 10 min default NCCL timeout fired while rank 0 was still in
# validation (and, before that, ranks had drifted because the slow step
# bypassed DDP.forward). Validation can exceed 10 min; keep the watchdog above it.
NCCL_TIMEOUT = timedelta(minutes=60)

EVAL_TOTAL_KEYS = (
    "oracle_nll",
    "oracle_kl",
    "oracle_cos",
    "oracle_n",
    "prefix_nll",
    "prefix_kl",
    "prefix_cos",
    "prefix_n",
    "planner_nll",
    "planner_n",
)


def slow_elbo(
    model,
    input_ids,
    labels,
    mu_q,
    log_var_q,
    eps,
    attention_mask=None,
    detach_planner=False,
):
    """Slow-step ELBO through ``model.forward`` so DDP's reducer is armed.

    ``unwrap_model(model).elbo(...)`` skips ``DistributedDataParallel.forward``
    / ``prepare_for_backward``. Gradients never all-reduce, ranks train
    independent copies, and the next ``wait_for_everyone`` NCCL-timeouts
    (job 69260: rank 0 still at step 955 while ranks 1-2 were already in
    ``run_validation``). Calling ``.elbo`` also skips ``torch.compile``.
    """
    return model(
        input_ids,
        labels=labels,
        attention_mask=attention_mask,
        mu_q=mu_q,
        log_var_q=log_var_q,
        eps=eps,
        detach_planner=detach_planner,
    )


def enable_fast_kernels():
    """TF32 + SDPA flash/mem-efficient. No-op on CPU."""
    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)


def load_base_lm(model_name: str, attn_implementation: str):
    kwargs = dict(trust_remote_code=True, use_cache=False)
    if attn_implementation and attn_implementation != "eager":
        try:
            return AutoModelForCausalLM.from_pretrained(
                model_name, attn_implementation=attn_implementation, **kwargs
            )
        except (TypeError, ValueError) as exc:
            print(f"attn_implementation={attn_implementation!r} rejected ({exc}); falling back")
    return AutoModelForCausalLM.from_pretrained(model_name, **kwargs)


def compiled_root(module):
    """Walk ``_orig_mod`` so checkpoints save the real ``LTLMCausalLM``."""
    root = module
    while hasattr(root, "_orig_mod"):
        root = root._orig_mod
    return root


def compile_ltlm(model, enabled: bool, mode: str):
    if not enabled:
        return model
    try:
        # dynamic=True: last val/train batches can be shorter than batch_size;
        # static shapes would recompile (minutes) every time that happens.
        compiled = torch.compile(model, mode=mode, dynamic=True)
        print(f"torch.compile enabled (mode={mode}, dynamic=True); first step compiles")
        return compiled
    except Exception as exc:
        print(f"torch.compile failed, running eager: {exc}")
        return model


def dataloader_kwargs(args, pin_memory: bool) -> dict:
    kwargs = dict(
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def broadcast_offsets(accelerator, offsets, seqlen):
    """Share rank-0 line offsets so the other ranks do not rescan GB-scale token files."""
    if accelerator.num_processes == 1:
        return offsets, seqlen
    device = accelerator.device
    meta = torch.tensor(
        [0 if offsets is None else len(offsets), 0 if seqlen is None else int(seqlen)],
        device=device,
        dtype=torch.long,
    )
    torch.distributed.broadcast(meta, src=0)
    n_off, seqlen_b = int(meta[0].item()), int(meta[1].item())
    buf = torch.empty(n_off, device=device, dtype=torch.long)
    if accelerator.is_main_process:
        buf.copy_(torch.tensor(offsets, device=device, dtype=torch.long))
    torch.distributed.broadcast(buf, src=0)
    return buf.cpu().tolist(), seqlen_b


def build_dataset(args, *, is_training: bool, offsets=None, sequence_length=None):
    return TokenizedDataset(
        args.data_file if is_training else args.val_file,
        onset_jitter_std=args.onset_jitter_std if is_training else 0.0,
        dur_jitter_range=args.dur_jitter_range if is_training else 0.0,
        mask_prob=args.mask_prob if is_training else 0.0,
        transpose_range_semitones=args.transpose_range_semitones if is_training else 0,
        tempo_scale_range=args.tempo_scale_range if is_training else 0.0,
        loss_mask_performance_tokens=args.loss_mask_performance_tokens,
        is_training=is_training,
        offsets=offsets,
        sequence_length=sequence_length,
    )


@torch.no_grad()
def evaluate_paths(model: LTLMCausalLM, posterior: PosteriorOptimizer, dataloader, accelerator, max_batches: int):
    """Teacher-forced NLL on oracle / prefix / planner paths.

    Every rank must enter this and participate in the final ``reduce``. A
    rank-0-only eval with ``wait_for_everyone`` around it is how job 69260
    died: non-main ranks sat on an NCCL barrier until the 10 min watchdog.

    The dataloader must already be rank-sharded (Subset of eval_max_samples
    then ``accelerator.prepare``). Oracle and prefix AdamVI are stacked into
    one doubled batch so those two independent posteriors share one kernel
    stream instead of running back-to-back.
    """
    model.eval()
    device = accelerator.device
    totals = {
        key: torch.zeros((), device=device, dtype=torch.float64) for key in EVAL_TOTAL_KEYS
    }
    for batch_idx, batch in enumerate(dataloader):
        if max_batches > 0 and batch_idx >= max_batches:
            break
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        planner_z = model.sample_planner_z(input_ids)

        prefix_labels = performance_only_labels(input_ids, labels)
        oracle, prefix = posterior.infer_paired(
            input_ids, labels, prefix_labels, attention_mask=attention_mask
        )
        nll_o, _ = model.score_nll(input_ids, labels, oracle["z"], attention_mask=attention_mask)
        div_o, mu_p, log_var_p = model.planner_divergence(
            oracle["z"], oracle["mu_q"], oracle["log_var_q"], input_ids, detach_planner=True
        )
        _, o_stats = planner_regularized_loss(
            nll_o,
            oracle["mu_q"],
            oracle["log_var_q"],
            mu_p if mu_p is not None else planner_z,
            log_var_p,
            beta=model.beta,
            divergence=div_o,
        )
        align_o = latent_alignment_stats(oracle["mu_q"], planner_z)

        nll_px, _ = model.score_nll(input_ids, labels, prefix["z"], attention_mask=attention_mask)
        div_p, _, _ = model.planner_divergence(
            prefix["z"], prefix["mu_q"], prefix["log_var_q"], input_ids, detach_planner=True
        )
        _, p_stats = planner_regularized_loss(
            nll_px,
            prefix["mu_q"],
            prefix["log_var_q"],
            planner_z,
            log_var_p,
            beta=model.beta,
            divergence=div_p,
        )
        align_p = latent_alignment_stats(prefix["mu_q"], planner_z)

        nll_pl, _ = model.score_nll(input_ids, labels, planner_z, attention_mask=attention_mask)

        bs = float(input_ids.shape[0])
        totals["oracle_nll"] += nll_o.detach().double() * bs
        totals["oracle_kl"] += o_stats["planner_kl"].detach().double() * bs
        totals["oracle_cos"] += align_o["z_cosine"].detach().double() * bs
        totals["oracle_n"] += bs
        totals["prefix_nll"] += nll_px.detach().double() * bs
        totals["prefix_kl"] += p_stats["planner_kl"].detach().double() * bs
        totals["prefix_cos"] += align_p["z_cosine"].detach().double() * bs
        totals["prefix_n"] += bs
        totals["planner_nll"] += nll_pl.detach().double() * bs
        totals["planner_n"] += bs

    packed = torch.stack([totals[key] for key in EVAL_TOTAL_KEYS])
    packed = accelerator.reduce(packed, reduction="sum")
    reduced = {key: packed[i] for i, key in enumerate(EVAL_TOTAL_KEYS)}

    def avg(key, count_key):
        n = reduced[count_key].clamp(min=1.0)
        return float(reduced[key] / n)

    model.train()
    metrics = {
        "val/oracle/loss": avg("oracle_nll", "oracle_n"),
        "val/oracle/kl": avg("oracle_kl", "oracle_n"),
        "val/oracle/z_cosine_vs_planner": avg("oracle_cos", "oracle_n"),
        "val/prefix/loss": avg("prefix_nll", "prefix_n"),
        "val/prefix/kl": avg("prefix_kl", "prefix_n"),
        "val/prefix/z_cosine_vs_planner": avg("prefix_cos", "prefix_n"),
        "val/planner/loss": avg("planner_nll", "planner_n"),
        "val/loss": avg("oracle_nll", "oracle_n"),
    }
    if model.planner_type == "diffusion":
        metrics["val/oracle/diffusion_loss"] = metrics["val/oracle/kl"]
        metrics["val/prefix/diffusion_loss"] = metrics["val/prefix/kl"]
    return metrics


def main():
    args = parse_args()
    if args.inference_method != "adamVI":
        raise ValueError("Only --inference_method adamVI is implemented")
    if wandb is None and args.wandb_mode != "disabled":
        args.wandb_mode = "disabled"

    if args.num_workers > 0:
        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

    validate_selected_cuda_device_or_raise(force_cpu=args.force_cpu)
    report_runtime_device(force_cpu=args.force_cpu)
    enable_fast_kernels()

    mixed_precision = "bf16" if torch.cuda.is_available() and not args.force_cpu else "no"
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        cpu=args.force_cpu,
        mixed_precision=mixed_precision,
        kwargs_handlers=[InitProcessGroupKwargs(timeout=NCCL_TIMEOUT)],
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

    if accelerator.is_main_process:
        train_dataset = build_dataset(args, is_training=True)
        t_off, t_len = train_dataset.offsets, train_dataset.sequence_length
    else:
        t_off, t_len = None, 0
    t_off, t_len = broadcast_offsets(accelerator, t_off, t_len)
    if not accelerator.is_main_process:
        train_dataset = build_dataset(args, is_training=True, offsets=t_off, sequence_length=t_len)

    if accelerator.is_main_process:
        val_full = build_dataset(args, is_training=False)
        v_off, v_len = val_full.offsets, val_full.sequence_length
    else:
        v_off, v_len = None, 0
    v_off, v_len = broadcast_offsets(accelerator, v_off, v_len)
    if not accelerator.is_main_process:
        val_full = build_dataset(args, is_training=False, offsets=v_off, sequence_length=v_len)

    n_val = min(args.eval_max_samples, len(val_full))
    val_dataset = Subset(val_full, range(n_val))
    pin = torch.cuda.is_available() and not args.force_cpu
    dl_kw = dataloader_kwargs(args, pin)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **dl_kw)
    val_loader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False, **dl_kw)

    base = load_base_lm(args.model_name, args.attn_implementation)
    if base.config.vocab_size != VOCAB_SIZE:
        base.resize_token_embeddings(VOCAB_SIZE)
    model = LTLMCausalLM(
        base,
        thoughts_per_layer=args.thoughts_per_layer,
        beta=args.kl_weight,
        isotropic_kl_weight=args.isotropic_kl_weight,
        elbo_reduction=args.elbo_reduction,
        planner_type=args.planner,
        diffusion_timesteps=args.diffusion_timesteps,
        diffusion_layers=args.diffusion_layers,
        ddim_steps=args.ddim_steps,
    )
    compile_on = bool(args.compile) and torch.cuda.is_available() and not args.force_cpu
    model = compile_ltlm(model, compile_on, args.compile_mode)

    adam_kw = dict(lr=args.learning_rate, eps=1e-6, weight_decay=0.01)
    if torch.cuda.is_available() and not args.force_cpu:
        adam_kw["fused"] = True
    try:
        optimizer = AdamW(model.parameters(), **adam_kw)
    except RuntimeError:
        adam_kw.pop("fused", None)
        optimizer = AdamW(model.parameters(), **adam_kw)
    model, optimizer, train_loader, val_loader = accelerator.prepare(
        model, optimizer, train_loader, val_loader
    )

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
        metrics = evaluate_paths(raw_model, posterior, val_loader, accelerator, max_batches=0)
        if accelerator.is_main_process:
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
                    # Must go through the DDP-wrapped `model`, not raw_model.elbo.
                    mu_q = mu_q.detach()
                    log_var_q = log_var_q.detach()
                    loss, stats, _ = slow_elbo(
                        model,
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
                            payload = {
                                "train/loss": float(stats["loss"]),
                                "train/mcmc_nll": float(stats["nll"]),
                                "train/planner_kl": float(stats["planner_kl"]),
                                "train/learning_rate": scheduler.get_last_lr()[0],
                            }
                            if "q_rms" in stats:
                                payload["train/q_rms"] = float(stats["q_rms"])
                            if raw_model.planner_type == "diffusion":
                                payload["train/diffusion_loss"] = float(stats["planner_kl"])
                            else:
                                payload["train/z_cosine_vs_planner"] = float(stats["z_cosine"])
                                payload["train/z_rmse_vs_planner"] = float(stats["z_rmse"])
                            wandb.log(payload, step=completed_steps)
                        if completed_steps % args.eval_steps == 0:
                            run_validation()
                        if completed_steps % args.save_steps == 0:
                            accelerator.wait_for_everyone()
                            if accelerator.is_main_process:
                                ckpt = args.output_dir / f"checkpoint-{completed_steps}"
                                ckpt.mkdir(parents=True, exist_ok=True)
                                to_save = compiled_root(accelerator.unwrap_model(model))
                                to_save.base_model.save_pretrained(ckpt)
                                extra_state = {
                                    k: v.detach().cpu()
                                    for k, v in to_save.state_dict().items()
                                    if not k.startswith("base_model.")
                                }
                                torch.save(
                                    {"ltlm": extra_state, "step": completed_steps, "args": vars(args)},
                                    ckpt / "ltlm_extra.pt",
                                )
                            accelerator.wait_for_everyone()
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
