#!/usr/bin/env python
"""Optimal Completion Distillation (Sabour et al. 2019, arXiv:1810.01398) on
the packed transcription task.

OCD trains exclusively on the model's OWN sampled rollouts: for every prefix
of a sampled sequence it computes the set of next tokens that lead to the best
achievable final task score, and distills a uniform distribution over that set
(the tau -> 0 soft-optimal policy) into the model at every position. No MLE
term, no exposure bias.

For this task the optimal-completion "DP" is exact and cheap. The metric is
the reward criterion (`f1_reward`, table column 3: pitch exact, onset within
+-1 bin, duration ignored) and every body slot counts toward n_pred whatever
it emits, so from any prefix the maximal achievable F1 is reached by emitting
YET-UNMATCHED ground-truth notes in the remaining slots:

  * onset position  -> onsets of all yet-unmatched GT notes;
  * duration/pitch  -> durations/pitches of the yet-unmatched GT notes within
    +-1 bin of the onset the model actually sampled (empty -> the slot can no
    longer match anything; those positions are masked out of the loss);
  * after the slot's three tokens are sampled, the emitted note is fed to the
    same greedy one-to-one matcher the reward uses, updating the unmatched set.

Differences from the collapsed `train_onpolicy_distill` arm (dense GT-CE on
own rollouts): that arm's target at slot k was THE slot-k GT token regardless
of what the prefix already achieved -- a Hamming-optimal completion that is
wrong for F1 (it re-targets already-matched notes and ignores the prefix).
OCD's target set is prefix-consistent and multi-modal.

Data: the unfiltered paper-split windows (`*_paper_unfiltered.txt`), the same
conditioning the results table scores. Validation: greedy unfiltered F1
(reusing `train_ppo_f1.validate_f1`) with a rolling `best-val-f1` checkpoint.

2-GPU launch:
  accelerate launch --num_processes 2 train_ocd.py --output_dir run_ocd_<stamp>
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F

import posttrain_common
from anticipation.vocab import DUR_OFFSET, NOTE_OFFSET, TIME_OFFSET
from f1_reward import ONSET_TOLERANCE
from onpolicy_rollout import (_role_constraint_mask, body_score_slot_starts,
                              rollout_score_slots, score_token_positions,
                              score_token_roles)
from train_ppo_f1 import gt_notes_from_batch, validate_f1


def parse_args():
    parser = posttrain_common.build_parser(
        "Optimal Completion Distillation on sampled rollouts (F1 metric)."
    )
    parser.set_defaults(
        data_file=Path("./data/train_paper_unfiltered.txt"),
        val_file=Path("./data/val_paper_unfiltered.txt"),
        learning_rate=5e-7,
        max_steps=6000,
        eval_steps=50,
        save_steps=250,
    )
    parser.add_argument("--save_best_val", action="store_true", default=True)
    parser.add_argument(
        "--tolerance", type=int, default=ONSET_TOLERANCE,
        help="Onset tolerance (bins) of the metric being distilled.",
    )
    parser.add_argument(
        "--ce_cap", type=float, default=4.0,
        help="Per-position CE clamp (nats); positions above it give zero "
             "gradient. Bounds the far-off-policy tail that collapsed the "
             "GT-CE distill arm (its post-mortem: 'CE is tail-dominated').",
    )
    return parser.parse_args()


def ocd_targets(rolled_rows, gt_per_window, slot_starts, tolerance, max_k):
    """Per-position optimal-token target sets for a batch of rollouts.

    Returns (targets (B, 3*S, max_k) long with -1 pad, achieved_f1 list).
    Position order matches `score_token_positions` (slot-major, roles 0/1/2).
    """
    batch = len(rolled_rows)
    n_pos = 3 * len(slot_starts)
    targets = torch.full((batch, n_pos, max_k), -1, dtype=torch.long)
    achieved = []
    for b in range(batch):
        # Unmatched pool with the reward's matching rule (pitch-exact bucket,
        # nearest onset within tolerance, one-to-one). Entries are SHARED
        # between the pitch-keyed pool and the chronological list.
        entries = [[onset, dur, pitch, False]
                   for onset, dur, pitch in gt_per_window[b]]
        chrono = sorted(entries, key=lambda e: (e[0], e[2]))
        pool = {}
        for e in entries:
            pool.setdefault(e[2], []).append(e)
        n_gt = len(entries)
        tp = 0
        ptr = 0
        row = rolled_rows[b]
        for s, start in enumerate(slot_starts):
            base = 3 * s
            # Onset position: the canonical optimal completion, with a
            # pointer that ADVANCES EVERY SLOT (skipping matched notes).
            # Targeting the bare earliest-unmatched note repeats one onset
            # across every non-matching slot -- a constant-output function
            # the model learns in ~20 steps while F1 collapses (observed).
            # The advancing pointer keeps targets distinct and time-aligned
            # with the slot structure; it never re-targets matched notes,
            # and when unmatched notes outnumber remaining slots it drops
            # the chronologically passed ones.
            while ptr < n_gt and chrono[ptr][3]:
                ptr += 1
            if ptr < n_gt:
                targets[b, base, 0] = TIME_OFFSET + chrono[ptr][0]
                ptr += 1
            o_samp = int(row[start]) - TIME_OFFSET
            # Duration/pitch positions: unmatched notes reachable from the
            # sampled onset.
            cand = [(e[1], p) for p, notes in pool.items() for e in notes
                    if not e[3] and abs(e[0] - o_samp) <= tolerance]
            durs = sorted({DUR_OFFSET + d for d, _ in cand})
            pits = sorted({NOTE_OFFSET + p for _, p in cand})
            for k, t in enumerate(durs[:max_k]):
                targets[b, base + 1, k] = t
            for k, t in enumerate(pits[:max_k]):
                targets[b, base + 2, k] = t
            # Commit the sampled note to the matcher.
            p_samp = int(row[start + 2]) - NOTE_OFFSET
            bucket = pool.get(p_samp)
            if bucket:
                best, best_d = None, None
                for e in bucket:
                    if e[3]:
                        continue
                    d = abs(e[0] - o_samp)
                    if d <= tolerance and (best_d is None or d < best_d):
                        best, best_d = e, d
                if best is not None:
                    best[3] = True
                    tp += 1
        achieved.append(2.0 * tp / max(len(slot_starts) + n_gt, 1))
    return targets, achieved


def main():
    args = parse_args()
    trainer = posttrain_common.PostTrainer(args, method="OCD")
    accelerator = trainer.accelerator

    seq_len = trainer.train_dataset[0]["input_ids"].shape[0]
    slot_starts = body_score_slot_starts(seq_len)
    positions = score_token_positions(seq_len, device=accelerator.device)
    roles = score_token_roles(positions)

    trainer.log(
        f"OCD: {args.prompts_per_step} windows/rank x "
        f"{accelerator.num_processes} ranks per step, rollout T="
        f"{args.rollout_temperature}, tolerance +-{args.tolerance} bin(s), "
        f"loss = CE(uniform over optimal completions) at every sampled "
        f"position, lr {args.learning_rate:g}, anchor {args.original_weight_l2:g}."
    )

    best_val_f1 = -float("inf")

    def validate_and_save_best(*, label=None):
        nonlocal best_val_f1
        results = validate_f1(trainer, args, label=label)
        mean_f1 = results["onset_pitch_tol1"]
        trainer.last_validated_step = int(trainer.completed_steps)
        if trainer.use_wandb and accelerator.is_main_process:
            import wandb
            wandb.log({"val/REWARD": mean_f1,
                       "val/f1_unfiltered": mean_f1,
                       "val/f1_onset_pitch": results["onset_pitch"],
                       "val/f1_onset_pitch_dur": results["onset_pitch_dur"]},
                      step=int(trainer.completed_steps))
        if accelerator.is_main_process:
            path = args.output_dir / "val_f1.csv"
            new = not path.exists()
            with path.open("a", encoding="utf-8") as handle:
                if new:
                    handle.write("step,REWARD,f1_onset_pitch,f1_onset_pitch_dur\n")
                handle.write(f"{int(trainer.completed_steps)},{mean_f1},"
                             f"{results['onset_pitch']},"
                             f"{results['onset_pitch_dur']}\n")
        if mean_f1 > best_val_f1:
            best_val_f1 = mean_f1
            trainer.save_checkpoint(name="best-val-f1")
            if accelerator.is_main_process:
                path = args.output_dir / "best_val_f1.json"
                tmp = path.with_suffix(path.suffix + ".tmp")
                tmp.write_text(json.dumps(
                    {"step": int(trainer.completed_steps), "val_f1": mean_f1,
                     "checkpoint": str(args.output_dir / "best-val-f1")},
                    indent=2) + "\n", encoding="utf-8")
                tmp.replace(path)
                trainer.log(f"New best unfiltered val F1 {100 * mean_f1:.3f}% "
                            f"at step {trainer.completed_steps}")
        return mean_f1

    validate_and_save_best(label="init (before any update)")

    mask_by_role = _role_constraint_mask(
        trainer.base_model.config.vocab_size, accelerator.device)

    training_failed = False
    try:
        while trainer.completed_steps < args.max_steps:
            for batch in trainer.train_dataloader:
                if trainer.completed_steps >= args.max_steps:
                    break
                t0 = time.perf_counter()
                input_ids = batch["input_ids"].to(accelerator.device)
                labels = batch["labels"].to(accelerator.device)

                # ---- on-policy rollout (no dropout, no grad) ---------------
                trainer.base_model.eval()
                with torch.no_grad():
                    rollout = rollout_score_slots(
                        trainer.base_model, input_ids, targets=labels,
                        temperature=args.rollout_temperature, constrain=True,
                        collect_logprobs=False, collect_gt_ce=False,
                        autocast_ctx=trainer.autocast,
                    )
                rolled = rollout["rolled"]
                valid_tok = rollout["valid"].to(accelerator.device)  # (B, 3S)

                # ---- optimal-completion target sets (CPU) ------------------
                gt_per_window = gt_notes_from_batch(labels.tolist(), slot_starts)
                max_k = max(1, max((len(g) for g in gt_per_window), default=1))
                targets, achieved = ocd_targets(
                    rolled.tolist(), gt_per_window, slot_starts,
                    args.tolerance, max_k)
                targets = targets.to(accelerator.device)      # (B, 3S, K)
                tgt_mask = targets >= 0
                pos_has_target = tgt_mask.any(dim=-1) & valid_tok.bool()
                n_pos_total = accelerator.reduce(
                    pos_has_target.sum().float(), reduction="sum")

                # ---- distillation update (chunked teacher-forced pass) -----
                total_loss = 0.0
                n_chunks = 0
                n_capped = torch.zeros((), device=accelerator.device)
                for lo in range(0, rolled.shape[0], args.micro_batch):
                    hi = min(lo + args.micro_batch, rolled.shape[0])
                    with trainer.autocast():
                        logits = trainer.model(
                            input_ids=rolled[lo:hi], use_cache=False).logits
                    sel = logits[:, positions - 1, :].float()
                    sel = sel.masked_fill(mask_by_role[roles].unsqueeze(0),
                                          float("-inf"))
                    logp = F.log_softmax(sel, dim=-1)         # (b, 3S, V)
                    tg = targets[lo:hi]
                    tm = tgt_mask[lo:hi]
                    pm = pos_has_target[lo:hi]
                    gathered = logp.gather(-1, tg.clamp(min=0))
                    # torch.where, NOT multiply-by-mask: pad entries gather
                    # -inf (masked vocab rows), and -inf * 0 = NaN poisons
                    # the reported value (gradients stay finite -- autograd
                    # sends zero grad through the masked edge -- so the NaN
                    # guard never fires while the loss prints nan).
                    gathered = torch.where(tm, gathered,
                                           torch.zeros_like(gathered))
                    per_pos = -gathered.sum(-1) \
                        / tm.float().sum(-1).clamp(min=1)
                    with torch.no_grad():
                        n_capped_local = ((per_pos > args.ce_cap)
                                          & pm).sum().float()
                    per_pos = per_pos.clamp(max=args.ce_cap)
                    chunk_loss = (per_pos * pm.float()).sum() \
                        / n_pos_total.clamp(min=1) * accelerator.num_processes
                    anchor = trainer.anchor_penalty()
                    if anchor is not None:
                        chunk_loss = chunk_loss + anchor * (hi - lo) \
                            / rolled.shape[0]
                    accelerator.backward(chunk_loss)
                    total_loss += float(chunk_loss.detach())
                    n_capped += n_capped_local
                    n_chunks += 1
                trainer.optimizer_step()
                trainer.completed_steps += 1
                frac_capped = float(
                    accelerator.reduce(n_capped, reduction="sum")
                    / n_pos_total.clamp(min=1))

                mean_f1 = sum(achieved) / max(len(achieved), 1)
                if accelerator.is_main_process and trainer.use_wandb:
                    import wandb
                    wandb.log({"train/ocd_loss": total_loss,
                               "train/frac_capped": frac_capped,
                               "train/rollout_f1": mean_f1,
                               "train/lr": trainer.optimizer.param_groups[0]["lr"],
                               "train/step_seconds":
                                   time.perf_counter() - t0},
                              step=int(trainer.completed_steps))
                if trainer.completed_steps % 10 == 0 \
                        or trainer.completed_steps <= 5:
                    trainer.log(
                        f"step {trainer.completed_steps} | ocd_loss "
                        f"{total_loss:.4f} | capped {100 * frac_capped:.1f}% | "
                        f"rollout F1 {100 * mean_f1:.2f}% | "
                        f"{time.perf_counter() - t0:.1f}s")
                if trainer.due_for_validation():
                    validate_and_save_best()
                if trainer.due_for_checkpoint():
                    trainer.save_checkpoint()
    except Exception:
        training_failed = True
        raise
    finally:
        if not training_failed:
            validate_and_save_best(label="final")
            trainer.save_checkpoint(name="final")
        trainer.finish()


if __name__ == "__main__":
    main()
