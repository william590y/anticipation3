"""LTLM validation extras: AR score decode, slot accuracies, piano rolls.

Training itself is unchanged; this module is logging / eval only. Wandb keys
match the previous LTLM trainer (``val/<mode>/ar_*``, ``val/<mode>/piano_roll``)
with ``planner`` as the third mode (the current trainer's name for p_phi(z|P);
older diffusion jobs used ``diffusion``). Oracle aliases ``val/ar_*`` so the
train.py dashboards still populate.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable, Optional

import torch

from anticipation.ltlm_model import LTLMCausalLM, LTLMOutput, control_token_mask
from anticipation.packed_sequence import ALTERNATING_START, is_score_slot_start, iter_score_slot_positions
from anticipation.score_constraints import constrain_score_token_logits
from anticipation.vocab import REST

try:
    from transformers import AutoConfig, AutoModelForCausalLM
except ImportError:  # pragma: no cover
    AutoConfig = AutoModelForCausalLM = None

EVAL_MODES = ("oracle", "prefix", "planner")


def num_score_slots(sequence_length: int, score_start_idx: int = ALTERNATING_START) -> int:
    return len(list(iter_score_slot_positions(sequence_length, score_start_idx)))


def empty_slot_accums(heatmap_slots: int, device) -> list[torch.Tensor]:
    return [
        torch.zeros(heatmap_slots, dtype=torch.float64, device=device)
        for _ in range(6)
    ]


def accumulate_score_slot_stats(generated, reference, heatmap_slots, accums) -> None:
    """Count exact-match errors on real body score slots (index % 3 roles)."""
    pitch_err, onset_err, dur_err, onset_abs, dur_abs, slot_total = accums
    length = min(generated.shape[1], reference.shape[1])
    for slot_idx, pos in enumerate(iter_score_slot_positions(length, ALTERNATING_START)):
        if slot_idx >= heatmap_slots or pos + 2 >= length:
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


def finalize_slot_accums(accums, accelerator=None) -> dict:
    stacked = torch.stack(accums)
    if accelerator is not None:
        stacked = accelerator.reduce(stacked, reduction="sum")
    pitch_err, onset_err, dur_err, onset_abs, dur_abs, slot_total = stacked.detach().cpu()
    total = float(slot_total.sum())

    def acc(err):
        return 1.0 - (float(err.sum()) / total) if total > 0 else 0.0

    return {
        "pitch_accuracy": acc(pitch_err),
        "onset_accuracy": acc(onset_err),
        "duration_accuracy": acc(dur_err),
        "total_notes": int(total),
    }


def remap_checkpoint_state(safetensors_state: dict, extra_ltlm: dict) -> dict:
    """Map ``save_pretrained`` + ``ltlm_extra.pt`` onto an ``LTLMCausalLM`` state_dict.

    The trainer writes wrapped GPT-2 weights (``transformer.h.N.block.*`` plus
    thought cross-attn) to ``model.safetensors``, and everything that is not
    ``base_model.*`` — planner + a duplicate ``blocks.*`` alias — to
    ``ltlm_extra.pt``. After ``LTLMCausalLM`` wraps a vanilla GPT-2, those
    safetensor keys are ``base_model.transformer.h.*``. Planner keys load as-is.
    ``blocks.*`` aliases the same ModuleList as ``transformer.h`` and is skipped
    so we do not double-assign.
    """
    state = {f"base_model.{key}": value for key, value in safetensors_state.items()}
    for key, value in extra_ltlm.items():
        if key.startswith("planner."):
            state[key] = value
    return state


def load_ltlm_checkpoint(
    ckpt_dir,
    device,
    attn_implementation: str = "sdpa",
):
    """Rebuild ``LTLMCausalLM`` from a ``train_ltlm.py`` checkpoint directory."""
    if AutoConfig is None or AutoModelForCausalLM is None:
        raise ImportError("transformers is required to load an LTLM checkpoint")
    ckpt_dir = Path(ckpt_dir)
    extra = torch.load(ckpt_dir / "ltlm_extra.pt", map_location="cpu", weights_only=False)
    args = extra.get("args") or {}
    config = AutoConfig.from_pretrained(str(ckpt_dir))
    try:
        base = AutoModelForCausalLM.from_config(
            config, attn_implementation=attn_implementation, trust_remote_code=True
        )
    except TypeError:
        base = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    model = LTLMCausalLM(
        base,
        thoughts_per_layer=int(args.get("thoughts_per_layer", 4)),
        beta=float(args.get("kl_weight", 5.0)),
        isotropic_kl_weight=float(args.get("isotropic_kl_weight", 0.0)),
        elbo_reduction=str(args.get("elbo_reduction", "mean")),
        planner_type=str(args.get("planner", "gaussian")),
        diffusion_timesteps=int(args.get("diffusion_timesteps", 1000)),
        diffusion_layers=int(args.get("diffusion_layers", 3)),
        ddim_steps=int(args.get("ddim_steps", 50)),
    )
    try:
        from safetensors.torch import load_file
    except ImportError as exc:  # pragma: no cover
        raise ImportError("safetensors is required to load LTLM checkpoints") from exc
    wrapped = load_file(str(ckpt_dir / "model.safetensors"))
    state = remap_checkpoint_state(wrapped, extra["ltlm"])
    missing, unexpected = model.load_state_dict(state, strict=False)
    planner_missing = [key for key in missing if key.startswith("planner.")]
    if planner_missing:
        raise RuntimeError(f"planner weights missing from {ckpt_dir}: {planner_missing[:8]}")
    cross_loaded = any("cross_attn" in key for key in state)
    if not cross_loaded:
        raise RuntimeError(f"thought cross-attn missing from {ckpt_dir}")
    if unexpected:
        print(f"load_ltlm_checkpoint unexpected keys (first 8): {list(unexpected)[:8]}")
    if hasattr(model.base_model, "config"):
        model.base_model.config.use_cache = True
    if str(device).startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model.to(device=device, dtype=torch.bfloat16)
    else:
        model.to(device)
    model.eval()
    return model, extra


def checkpoint_step(ckpt_dir, extra=None) -> int:
    ckpt_dir = Path(ckpt_dir)
    if extra and extra.get("step") is not None:
        return int(extra["step"])
    name = ckpt_dir.name
    if name.startswith("checkpoint-"):
        return int(name.split("-", 1)[1])
    if name == "final":
        return -1
    raise ValueError(f"cannot parse step from {ckpt_dir}")


def is_complete_checkpoint(ckpt_dir) -> bool:
    ckpt_dir = Path(ckpt_dir)
    return (ckpt_dir / "model.safetensors").is_file() and (ckpt_dir / "ltlm_extra.pt").is_file()


def list_complete_checkpoints(output_dir) -> list[Path]:
    output_dir = Path(output_dir)
    found = []
    for path in sorted(output_dir.glob("checkpoint-*")):
        if path.is_dir() and is_complete_checkpoint(path):
            found.append(path)
    final = output_dir / "final"
    if final.is_dir() and is_complete_checkpoint(final):
        found.append(final)
    return found


def fixed_piano_roll_indices(dataset, count) -> list[int]:
    """First ``count`` validation windows, in file order — same every eval."""
    if count is None or count <= 0 or len(dataset) == 0:
        return []
    return list(range(min(int(count), len(dataset))))


@torch.inference_mode()
def ltlm_autoregressive_generate_score(
    model,
    tokens_batch,
    z,
    device,
    constrain_score_tokens=True,
    ground_truth_score_tokens_to_feed=0,
    score_start_idx=ALTERNATING_START,
):
    """Greedy score decode with teacher-forced controls, conditioned on ``z``.

    Same slot schedule as ``inference.batched_autoregressive_generate_score``:
    body score triplets are emitted left to right, each followed by the
    ground-truth control triplet. ``z`` is held fixed (the plan, not a token).
    """
    if ground_truth_score_tokens_to_feed < 0:
        raise ValueError("ground_truth_score_tokens_to_feed must be non-negative")

    tokens_batch = tokens_batch.to(device)
    z = z.to(device)
    length = tokens_batch.shape[1]
    context_chunks = [tokens_batch[:, :score_start_idx]]
    fed_score_tokens = 0

    model.set_thoughts(z)
    try:
        def _forward(token_chunk, past=None):
            out = model(token_chunk, past_key_values=past, use_cache=True)
            if isinstance(out, LTLMOutput):
                return out.logits, out.past_key_values
            return out.logits, getattr(out, "past_key_values", None)

        logits, past = _forward(context_chunks[0], past=None)
        next_logits = logits[:, -1, :]

        def append_and_feed(token_col):
            nonlocal past, next_logits
            context_chunks.append(token_col.unsqueeze(1))
            logits, past = _forward(token_col.unsqueeze(1), past=past)
            next_logits = logits[:, -1, :]

        pos = score_start_idx
        while pos + 5 < length:
            if is_score_slot_start(pos, score_start_idx):
                for slot in range(3):
                    if fed_score_tokens < ground_truth_score_tokens_to_feed:
                        token_col = tokens_batch[:, pos + slot]
                        fed_score_tokens += 1
                    else:
                        step_logits = next_logits
                        if constrain_score_tokens:
                            step_logits = constrain_score_token_logits(step_logits, slot)
                        token_col = step_logits.argmax(dim=-1)
                    append_and_feed(token_col)
                pos += 3
                if pos + 2 < length:
                    for k in range(3):
                        append_and_feed(tokens_batch[:, pos + k])
                    pos += 3
            else:
                append_and_feed(tokens_batch[:, pos])
                pos += 1
        return torch.cat(context_chunks, dim=1)
    finally:
        model.set_thoughts(None)


def infer_mode_z(model, posterior, input_ids, labels, attention_mask=None) -> dict:
    """Oracle / prefix AdamVI z plus planner-sampled z (no score tokens in p_phi)."""
    planner_z = model.sample_planner_z(input_ids)
    prefix_labels = labels.clone().masked_fill(~control_token_mask(input_ids), -100)
    oracle, prefix = posterior.infer_paired(
        input_ids, labels, prefix_labels, attention_mask=attention_mask
    )
    return {
        "oracle": oracle["z"],
        "prefix": prefix["z"],
        "planner": planner_z,
    }


@torch.no_grad()
def evaluate_ar_modes(
    model,
    posterior,
    dataloader,
    accelerator,
    modes: Iterable[str] = EVAL_MODES,
    heatmap_slots: Optional[int] = None,
):
    """AR pitch/onset/duration accuracy per plan mode. Every rank must enter."""
    model.eval()
    device = accelerator.device
    modes = tuple(modes)
    accums = None
    iterator = dataloader
    try:
        from tqdm import tqdm

        iterator = tqdm(
            dataloader,
            desc="LTLM AR eval",
            file=sys.stdout,
            mininterval=10,
            ascii=True,
            leave=True,
            disable=not getattr(accelerator, "is_local_main_process", True),
        )
    except ImportError:
        pass
    for batch in iterator:
        input_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        if heatmap_slots is None:
            heatmap_slots = num_score_slots(input_ids.shape[1])
        if accums is None:
            accums = {mode: empty_slot_accums(heatmap_slots, device) for mode in modes}
        if input_ids.shape[0] == 0 or input_ids.shape[1] <= ALTERNATING_START:
            continue
        zs = infer_mode_z(model, posterior, input_ids, labels, attention_mask)
        for mode in modes:
            generated = ltlm_autoregressive_generate_score(
                model,
                input_ids,
                zs[mode],
                device,
                constrain_score_tokens=True,
                ground_truth_score_tokens_to_feed=0,
            )
            accumulate_score_slot_stats(generated, input_ids, heatmap_slots, accums[mode])
            del generated
    if accums is None:
        if heatmap_slots is None:
            heatmap_slots = num_score_slots(1020)
        accums = {mode: empty_slot_accums(heatmap_slots, device) for mode in modes}
    return {mode: finalize_slot_accums(accums[mode], accelerator) for mode in modes}


def packed_notes_for_rolls(input_ids):
    """Performance / GT score / validity tensors for ``piano_roll_figure``.

    Delegates to ``plan_vq.split_packed_batch`` (packed_sequence slot layout)
    so the figure matches previous LTLM / VQ wandb images.
    """
    from plan_vq import split_packed_batch

    return split_packed_batch(input_ids)


def render_ltlm_piano_rolls(
    model,
    posterior,
    windows_batch,
    indices,
    step,
    device,
    modes: Iterable[str] = EVAL_MODES,
):
    """One (N windows x 3 columns) figure per plan mode. Rank 0 only."""
    from plan_vq_viz import piano_roll_figure

    modes = tuple(modes)
    input_ids = windows_batch["input_ids"].to(device)
    labels = windows_batch["labels"].to(device)
    attention_mask = windows_batch["attention_mask"].to(device)
    zs = infer_mode_z(model, posterior, input_ids, labels, attention_mask)
    perf, score, valid = packed_notes_for_rolls(input_ids)
    figures = {}
    for mode in modes:
        generated = ltlm_autoregressive_generate_score(
            model,
            input_ids,
            zs[mode],
            device,
            constrain_score_tokens=True,
            ground_truth_score_tokens_to_feed=0,
        )
        _, predicted, _ = packed_notes_for_rolls(generated)
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
        del generated
    return figures


def wandb_ar_payload(
    ar_by_mode: dict,
    figures: Optional[dict] = None,
    wandb_module=None,
    step: Optional[int] = None,
) -> dict:
    """Build the wandb dict previous LTLM runs logged (accuracies in percent)."""
    payload = {}
    for mode, block in ar_by_mode.items():
        payload[f"val/{mode}/ar_pitch_accuracy"] = block["pitch_accuracy"] * 100
        payload[f"val/{mode}/ar_onset_accuracy"] = block["onset_accuracy"] * 100
        payload[f"val/{mode}/ar_duration_accuracy"] = block["duration_accuracy"] * 100
    oracle = ar_by_mode.get("oracle")
    if oracle is not None:
        payload["val/ar_pitch_accuracy"] = oracle["pitch_accuracy"] * 100
        payload["val/ar_onset_accuracy"] = oracle["onset_accuracy"] * 100
        payload["val/ar_duration_accuracy"] = oracle["duration_accuracy"] * 100
    if figures:
        caption_step = f" checkpoint-{step}" if step is not None else ""
        for mode, figure in figures.items():
            image = figure
            if wandb_module is not None:
                image = wandb_module.Image(
                    figure,
                    caption=f"{mode} generated vs GT score vs performance{caption_step}",
                )
            payload[f"val/{mode}/piano_roll"] = image
    return payload
