#!/usr/bin/env python
"""Precompute one of four self-contained PPO rollout shards for the visualizer."""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "visualizer"))

from anticipation.config import CONTEXT_SIZE  # noqa: E402
from evaluate_muster import load_model  # noqa: E402
from atomic_json import atomic_dump_json  # noqa: E402
from compute_f1 import score_notes  # noqa: E402
from compute_sequence_ppl import (  # noqa: E402
    apply_gen_metrics,
    build_packed_tokens,
    control_notes_for_variant,
    notes_from_pred,
    score_packed_sequence,
)
from ppo_pipeline_common import (  # noqa: E402
    EXPECTED_SHARDS,
    WINDOWS_PER_SHARD,
    SEED_COUNTS,
    dataset_identity,
    load_data_js,
    ordered_example_ids,
    validate_rollout_block,
)
from precompute_visualizer import compute_rollout_set, tokens_from_controls  # noqa: E402
from select_ppo_best import validate_selected_manifest  # noqa: E402


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_note(note):
    return (
        isinstance(note, dict)
        and note.get("t") is not None
        and note.get("d") is not None
        and note.get("p") is not None
    )


def aligned_gt_notes(ex, variant, n_slots):
    """Ground truth in the filtered or raw slot coordinate system."""
    gt = ex.get("gt_score") or []
    if variant.startswith("raw"):
        aligned = []
        raw = ex.get("raw_notes") or []
        for slot in range(n_slots):
            j = raw[slot].get("j") if slot < len(raw) and isinstance(raw[slot], dict) else None
            if (
                isinstance(j, int)
                and not isinstance(j, bool)
                and 0 <= j < len(gt)
                and _is_note(gt[j])
            ):
                aligned.append(gt[j])
            else:
                aligned.append(None)
        return aligned
    return [gt[slot] if slot < len(gt) and _is_note(gt[slot]) else None for slot in range(n_slots)]


@torch.inference_mode()
def attach_ui_metrics(model, device, ex, block):
    """Attach note F1 and generated-vs-GT sequence PPL while the model is resident."""
    for variant, rollout in block.items():
        if not isinstance(rollout, dict):
            continue
        pred = rollout.get("pred_score") or []
        n_slots = len(pred)
        gt_aligned = aligned_gt_notes(ex, variant, n_slots)
        pred_notes = [note for note in pred if _is_note(note)]
        gt_notes = [note for note in gt_aligned if _is_note(note)]
        rollout["f1"] = score_notes(pred_notes, gt_notes)

        controls = control_notes_for_variant(ex, variant)
        gen_tokens, gen_positions, _ = build_packed_tokens(
            controls, notes_from_pred(pred)
        )
        gt_tokens, gt_positions, _ = build_packed_tokens(controls, gt_aligned)
        generated = score_packed_sequence(
            model,
            device,
            gen_tokens,
            gen_positions,
            n_slots,
            with_entropy=True,
        )
        ground_truth = score_packed_sequence(
            model,
            device,
            gt_tokens,
            gt_positions,
            n_slots,
            with_entropy=False,
        )
        pair = apply_gen_metrics(
            rollout,
            generated,
            ground_truth,
            write_entropy=True,
        )
        if pair is None:
            raise ValueError(f"could not compute sequence perplexity for {variant}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="visualizer/data.js")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=EXPECTED_SHARDS)
    parser.add_argument("--device", default=None)
    parser.add_argument("--topk-onset", type=int, default=5)
    parser.add_argument("--topk-dur", type=int, default=4)
    parser.add_argument("--topk-pitch", type=int, default=8)
    parser.add_argument("--max-candidates", type=int, default=40)
    args = parser.parse_args()
    if args.num_shards != EXPECTED_SHARDS:
        raise SystemExit(f"PPO visualization requires exactly {EXPECTED_SHARDS} shards")
    if not 0 <= args.shard_index < args.num_shards:
        raise SystemExit("invalid shard index")

    try:
        manifest = validate_selected_manifest(args.manifest, verify_hashes=False)
        payload, _ = load_data_js(args.data)
        full_order = ordered_example_ids(payload)
        source_identity = dataset_identity(payload)
    except ValueError as exc:
        raise SystemExit(f"refusing PPO precompute: {exc}") from exc
    order = full_order[args.shard_index :: args.num_shards]
    if len(order) != WINDOWS_PER_SHARD:
        raise SystemExit(
            f"expected {WINDOWS_PER_SHARD} windows in shard, got {len(order)}"
        )

    checkpoint = manifest["checkpoint"]
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model, loaded_device = load_model(checkpoint, config_source=None)
    if args.device is None and isinstance(loaded_device, (str, torch.device)):
        device = torch.device(loaded_device)
    model.to(device)
    model.eval()
    rollout_args = SimpleNamespace(
        topk_onset=args.topk_onset,
        topk_dur=args.topk_dur,
        topk_pitch=args.topk_pitch,
        max_candidates=args.max_candidates,
        slot_progress=False,
    )

    print(
        f"PPO shard {args.shard_index}/{args.num_shards}: {order}\n"
        f"checkpoint={checkpoint}\ncheckpoint_id={manifest['checkpoint_id']}",
        flush=True,
    )
    examples = payload["examples"]
    output_examples = {}
    started = time.perf_counter()
    for eid in tqdm(order, desc=f"PPO shard {args.shard_index}"):
        ex = examples[eid]
        tokens = tokens_from_controls(ex.get("perf_notes") or [], CONTEXT_SIZE - 4)
        window_started = time.perf_counter()
        block, _ = compute_rollout_set(
            model,
            device,
            tokens,
            ex.get("raw_notes"),
            ex.get("gt_score") or [],
            rollout_args,
            seed_counts=SEED_COUNTS,
        )
        attach_ui_metrics(model, device, ex, block)
        try:
            validate_rollout_block(block, ex, context=eid)
        except ValueError as exc:
            raise SystemExit(f"invalid generated PPO shard data: {exc}") from exc
        output_examples[eid] = {"rollouts_ppo": block}
        tqdm.write(
            f"  {eid}: {sum(value is not None for value in block.values())} variants "
            f"({time.perf_counter() - window_started:.1f}s)"
        )

    manifest_path = Path(args.manifest).resolve()
    shard = {
        "schema": "anticipation3.ppo-rollout-shard.v1",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "seed_counts": list(SEED_COUNTS),
        "train_job_id": manifest["train_job_id"],
        "best_step": manifest["best_step"],
        "val_reward": manifest["val_reward"],
        "checkpoint": manifest["checkpoint"],
        "checkpoint_id": manifest["checkpoint_id"],
        "manifest": str(manifest_path),
        "manifest_sha256": file_sha256(manifest_path),
        "source_data": str(Path(args.data).resolve()),
        "source_data_identity": source_identity,
        "example_order": order,
        "examples": output_examples,
    }
    atomic_dump_json(args.output, shard)
    print(
        f"Atomically wrote {Path(args.output).resolve()} in "
        f"{time.perf_counter() - started:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    main()

