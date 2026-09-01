#!/usr/bin/env python
"""Backfill LTLM AR accuracies + piano rolls onto an existing wandb run.

The live Gaussian planner job (81973) trains from
``/home/wjl86/anticipation3-planner`` and will not pick up new Python. This
script loads that clone's checkpoints, runs the same AR / piano-roll eval the
updated ``train_ltlm.py`` now logs, and commits them onto the live run with
``checkpoint_step`` as the chart x-axis.

Do **not** ``wandb.log(..., step=checkpoint_step)`` on a live trainer: wandb
drops any ``step`` behind the run's monotonic ``_step``, which is why the
first sidecar uploaded nothing the user could see.

Does not finish the wandb run while the training job is still heartbeating.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import subprocess
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from anticipation.ltlm_eval import (
    EVAL_MODES,
    checkpoint_step,
    evaluate_ar_modes,
    fixed_piano_roll_indices,
    list_complete_checkpoints,
    load_ltlm_checkpoint,
    render_ltlm_piano_rolls,
    wandb_ar_payload,
)
from anticipation.ltlm_posterior import PosteriorOptimizer
from train import TokenizedDataset
from train_ltlm import collate_fn, compile_ltlm, enable_fast_kernels

try:
    import wandb
except ImportError:
    wandb = None


MARKER_DIRNAME = "wandb_ar_backfill"


class _SingleAccelerator:
    def __init__(self, device):
        self.device = device
        self.is_main_process = True
        self.is_local_main_process = True
        self.num_processes = 1

    def reduce(self, tensor, reduction="sum"):
        return tensor

    def wait_for_everyone(self):
        return


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("/home/wjl86/anticipation3-planner/run_ltlm_planner"),
    )
    parser.add_argument("--val_file", type=Path, default=Path("data/val_paper.txt"))
    parser.add_argument("--wandb_project", type=str, default="anticipation-asap")
    parser.add_argument("--wandb_entity", type=str, default="wjl86-cornell-university")
    parser.add_argument("--wandb_run_id", type=str, default="ri7dgjs2")
    parser.add_argument(
        "--wandb_dir",
        type=Path,
        default=Path("/home/wjl86/anticipation3/logs/wandb_ltlm_planner_backfill"),
        help="Local wandb directory. Must NOT be the training output_dir "
        "(that hijacks latest-run and can fight the live trainer).",
    )
    parser.add_argument("--eval_max_samples", type=int, default=100)
    parser.add_argument("--piano_roll_examples", type=int, default=3)
    parser.add_argument("--val_batch_size", type=int, default=4)
    parser.add_argument(
        "--compile",
        action="store_true",
        default=False,
        help="Compile GPT-2. Off by default: AR + KV-cache recompiles every step.",
    )
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--attn_implementation", type=str, default="sdpa")
    parser.add_argument(
        "--steps",
        type=str,
        default="",
        help="Comma-separated checkpoint steps to run. Empty = every complete checkpoint.",
    )
    parser.add_argument("--force", action="store_true", help="Re-log even if a done marker exists.")
    parser.add_argument(
        "--upload_only",
        action="store_true",
        help="CPU-only: log existing piano-roll PNGs and marker metrics. No model load.",
    )
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--watch_job_id", type=str, default="")
    parser.add_argument("--poll_seconds", type=int, default=120)
    parser.add_argument("--settle_seconds", type=int, default=20)
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--no_wandb",
        action="store_true",
        help="Compute only (write PNGs/markers). Upload with --upload_only "
        "on the training node via srun --overlap.",
    )
    return parser.parse_args()


def marker_path(output_dir: Path, step: int) -> Path:
    return output_dir / MARKER_DIRNAME / f"step-{step}.done"


def slurm_job_running(job_id: str) -> bool:
    if not job_id:
        return False
    proc = subprocess.run(
        ["squeue", "-j", str(job_id), "-h", "-o", "%T"],
        capture_output=True,
        text=True,
        check=False,
    )
    states = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    live = {"PENDING", "RUNNING", "COMPLETING", "CONFIGURING", "REQUEUED"}
    return bool(states & live)


def wait_until_stable(ckpt: Path, settle_seconds: int) -> None:
    extra = ckpt / "ltlm_extra.pt"
    last = None
    while True:
        if not extra.is_file():
            time.sleep(min(settle_seconds, 5))
            continue
        stat = extra.stat()
        sig = (stat.st_mtime_ns, stat.st_size)
        if last == sig:
            return
        last = sig
        time.sleep(settle_seconds)


def requested_steps(raw: str) -> set[int] | None:
    if not raw.strip():
        return None
    return {int(part.strip()) for part in raw.split(",") if part.strip()}


AR_METRIC_KEYS = [
    "val/ar_pitch_accuracy",
    "val/ar_onset_accuracy",
    "val/ar_duration_accuracy",
]
for _mode in EVAL_MODES:
    AR_METRIC_KEYS.extend(
        [
            f"val/{_mode}/ar_pitch_accuracy",
            f"val/{_mode}/ar_onset_accuracy",
            f"val/{_mode}/ar_duration_accuracy",
            f"val/{_mode}/piano_roll",
        ]
    )


def configure_checkpoint_axis(run) -> None:
    """Bind AR / piano-roll keys to checkpoint_step.

    wandb silently *drops* ``log(..., step=N)`` when N is behind the live
    trainer's monotonic ``_step`` (handler.py: "User provided step is less
    than current step. Dropping entry"). By the time a sidecar finishes AR
    eval, job 81973 has already moved past that checkpoint, so historical
    ``step=2500`` never appears. ``define_metric(..., step_metric=)`` is the
    supported way to put points at 2500/5000 on the charts.
    """
    run.define_metric("checkpoint_step")
    for key in AR_METRIC_KEYS:
        run.define_metric(key, step_metric="checkpoint_step")


def discover_trainer_wandb_service() -> str | None:
    """Build WANDB_SERVICE for the live trainer's wandb-core on this node."""
    try:
        import os as _os

        cores = []
        for pid in _os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                cmd = _os.readlink(f"/proc/{pid}/exe")
            except OSError:
                continue
            if "wandb-core" not in cmd and not cmd.endswith("wandb-core"):
                # cmdline fallback
                try:
                    with open(f"/proc/{pid}/cmdline", "rb") as fh:
                        raw = fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
                except OSError:
                    continue
                if "wandb/bin/wandb-core" not in raw:
                    continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as fh:
                    raw = fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                continue
            if "wandb-core" not in raw:
                continue
            parent = None
            parts = raw.split()
            if "--pid" in parts:
                try:
                    parent = int(parts[parts.index("--pid") + 1])
                except (ValueError, IndexError):
                    parent = None
            cores.append((int(pid), parent))
        for core_pid, parent in cores:
            if parent is None:
                continue
            try:
                with open(f"/proc/{parent}/cmdline", "rb") as fh:
                    parent_cmd = fh.read().replace(b"\0", b" ").decode("utf-8", "replace")
            except OSError:
                continue
            if "train_ltlm.py" not in parent_cmd:
                continue
            pattern = f"/tmp/wandb-{parent}-{core_pid}-*/socket"
            import glob as _glob

            matches = _glob.glob(pattern)
            if matches:
                path = matches[0]
                token = f"3-{parent}-unix-{path}"
                print(f"discovered WANDB_SERVICE={token}", flush=True)
                return token
    except Exception as exc:
        print(f"wandb service discovery failed: {exc}", flush=True)
    return None


def attach_wandb(args, keep_alive: bool):
    if args.dry_run or getattr(args, "no_wandb", False) or wandb is None:
        return None
    token = os.environ.get("WANDB_SERVICE") or discover_trainer_wandb_service()
    run = None
    attached = False
    if token:
        os.environ["WANDB_SERVICE"] = token
        try:
            run = wandb.attach(attach_id=args.wandb_run_id)
            attached = True
            print(f"attached to live wandb run {args.wandb_run_id}", flush=True)
        except Exception as exc:
            print(f"wandb.attach failed ({exc}); will not open a second filestream", flush=True)
            run = None
    if run is None:
        raise SystemExit(
            "Refusing to wandb.init() a second writer on the live trainer. "
            "Run --upload_only via `srun --jobid=81973 --overlap` so we can "
            "attach to the trainer's wandb-core unix socket."
        )
    args._wandb_attached = attached
    configure_checkpoint_axis(run)
    hook = getattr(run, "_atexit_cleanup", None)
    if callable(hook):
        try:
            atexit.unregister(hook)
        except Exception:
            pass
    return run


def flush_wandb(run) -> None:
    if run is None:
        return
    try:
        backend = getattr(run, "_backend", None)
        interface = getattr(backend, "interface", None) if backend is not None else None
        if interface is not None and hasattr(interface, "publish_flush"):
            interface.publish_flush()
    except Exception as exc:
        print(f"wandb flush warning: {exc}", flush=True)


def piano_roll_pngs(output_dir: Path, step: int) -> dict[str, Path]:
    roll_dir = output_dir / "piano_rolls"
    found = {}
    for mode in EVAL_MODES:
        path = roll_dir / f"{mode}_step{step:06d}.png"
        if path.is_file():
            found[mode] = path
    return found


def payload_from_pngs(pngs: dict[str, Path], step: int) -> dict:
    if wandb is None:
        return {}
    payload = {}
    for mode, path in pngs.items():
        payload[f"val/{mode}/piano_roll"] = wandb.Image(
            str(path),
            caption=f"{mode} generated vs GT score vs performance checkpoint-{step}",
        )
    return payload


def upload_saved_artifacts(args, wandb_run) -> list[int]:
    """Log marker metrics + existing PNGs. Used by --upload_only and as a retry."""
    uploaded = []
    marker_dir = args.output_dir / MARKER_DIRNAME
    steps = set()
    if marker_dir.is_dir():
        for path in marker_dir.glob("step-*.done"):
            try:
                steps.add(int(path.stem.split("-", 1)[1]))
            except ValueError:
                continue
    for png in (args.output_dir / "piano_rolls").glob("*_step*.png"):
        try:
            steps.add(int(png.stem.rsplit("step", 1)[1]))
        except ValueError:
            continue
    wanted = requested_steps(args.steps)
    for step in sorted(steps):
        if wanted is not None and step not in wanted:
            continue
        payload = {}
        marker = marker_path(args.output_dir, step)
        if marker.is_file():
            numeric = json.loads(marker.read_text())
            payload.update(
                {key: value for key, value in numeric.items() if key.startswith("val/")}
            )
        pngs = piano_roll_pngs(args.output_dir, step)
        payload.update(payload_from_pngs(pngs, step))
        if not payload:
            continue
        uploaded_mark = args.output_dir / MARKER_DIRNAME / f"step-{step}.uploaded"
        if uploaded_mark.is_file() and not args.force:
            marker_mtime = marker.stat().st_mtime if marker.is_file() else 0
            if marker_mtime <= uploaded_mark.stat().st_mtime:
                print(f"upload_only skip step={step} (already uploaded)", flush=True)
                continue
        print(f"upload_only step={step} keys={sorted(payload)}", flush=True)
        if wandb_run is not None:
            log_checkpoint_payload(wandb_run, payload, step)
            uploaded_mark.parent.mkdir(parents=True, exist_ok=True)
            uploaded_mark.write_text("ok\n")
        uploaded.append(step)
    return uploaded


def log_checkpoint_payload(run, payload: dict, step: int) -> None:
    """Commit onto the live run without using a decreasing wandb ``step=``.

    Always ``commit=True``. Passing ``step=`` defaults to commit=False *and*
    is dropped when ``step < live _step``.
    """
    body = {"checkpoint_step": int(step), **payload}
    print(
        f"wandb.log checkpoint_step={step} keys={sorted(body)}",
        flush=True,
    )
    wandb.log(body, commit=True)
    flush_wandb(run)


def detach_wandb(run, finish: bool) -> None:
    if run is None or wandb is None:
        return
    flush_wandb(run)
    # Never finish an attached trainer run.
    if finish:
        print("skipping wandb.finish() on attached live trainer run", flush=True)
    hook = getattr(run, "_atexit_cleanup", None)
    if callable(hook):
        try:
            atexit.unregister(hook)
        except Exception:
            pass


def build_val(args, extra_args: dict):
    dataset = TokenizedDataset(
        args.val_file,
        onset_jitter_std=0.0,
        dur_jitter_range=0.0,
        mask_prob=0.0,
        transpose_range_semitones=0,
        tempo_scale_range=0.0,
        loss_mask_performance_tokens=bool(extra_args.get("loss_mask_performance_tokens", False)),
        is_training=False,
    )
    n_val = min(args.eval_max_samples, len(dataset))
    subset = Subset(dataset, range(n_val))
    loader = DataLoader(
        subset,
        batch_size=args.val_batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    roll_idx = fixed_piano_roll_indices(dataset, args.piano_roll_examples)
    return dataset, loader, roll_idx


def backfill_one(ckpt: Path, args, dataset, loader, roll_idx, accelerator, wandb_run) -> dict:
    wait_until_stable(ckpt, args.settle_seconds)
    print(f"Loading {ckpt}", flush=True)
    model, extra = load_ltlm_checkpoint(
        ckpt, accelerator.device, attn_implementation=args.attn_implementation
    )
    if args.compile:
        model = compile_ltlm(model, True, "default")
    step = checkpoint_step(ckpt, extra)
    mcmc = int((extra.get("args") or {}).get("mcmc_steps", 16))
    fast_lr = float((extra.get("args") or {}).get("fast_lr", 0.3))
    final_fast_lr = float((extra.get("args") or {}).get("final_fast_lr", 0.34))
    posterior = PosteriorOptimizer(model, num_steps=mcmc, lr=fast_lr, final_lr=final_fast_lr)

    # Piano rolls first so the Media panel populates before the 100-window AR pass.
    figures = {}
    existing_pngs = piano_roll_pngs(args.output_dir, step)
    if roll_idx and len(existing_pngs) == len(EVAL_MODES) and not args.force:
        print(f"Reusing piano-roll PNGs for step={step}", flush=True)
        if wandb_run is not None:
            log_checkpoint_payload(wandb_run, payload_from_pngs(existing_pngs, step), step)
    elif roll_idx:
        print(f"Piano rolls step={step}  windows={roll_idx}", flush=True)
        windows = {
            key: torch.stack([dataset[i][key] for i in roll_idx])
            for key in ("input_ids", "attention_mask", "labels")
        }
        figures = render_ltlm_piano_rolls(
            model, posterior, windows, roll_idx, step, accelerator.device
        )
        roll_dir = args.output_dir / "piano_rolls"
        roll_dir.mkdir(parents=True, exist_ok=True)
        import matplotlib.pyplot as plt

        for mode, figure in figures.items():
            out = roll_dir / f"{mode}_step{step:06d}.png"
            figure.savefig(out, dpi=110)
            print(f"wrote {out}", flush=True)
        if wandb_run is not None:
            image_payload = wandb_ar_payload(
                {},
                figures,
                wandb_module=wandb,
                step=step,
            )
            log_checkpoint_payload(wandb_run, image_payload, step)

    print(f"AR eval step={step}  samples={args.eval_max_samples}", flush=True)
    ar_by_mode = evaluate_ar_modes(model, posterior, loader, accelerator, modes=EVAL_MODES)
    payload = wandb_ar_payload(
        ar_by_mode, figures, wandb_module=wandb if wandb_run else None, step=step
    )
    if not figures:
        payload.update(payload_from_pngs(piano_roll_pngs(args.output_dir, step), step))
    numeric = {
        key: value
        for key, value in payload.items()
        if not key.endswith("/piano_roll")
    }
    print(json.dumps({"checkpoint_step": step, **numeric}, indent=2), flush=True)
    if wandb_run is not None:
        log_checkpoint_payload(wandb_run, payload, step)
    if figures:
        import matplotlib.pyplot as plt

        plt.close("all")
    marker = marker_path(args.output_dir, step)
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"step": step, "checkpoint": str(ckpt), **numeric}, indent=2) + "\n")
    del model, posterior
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return numeric


def pending_checkpoints(args) -> list[Path]:
    wanted = requested_steps(args.steps)
    pending = []
    for ckpt in list_complete_checkpoints(args.output_dir):
        extra = None
        try:
            step = checkpoint_step(ckpt, extra)
        except ValueError:
            continue
        if wanted is not None and step not in wanted:
            continue
        if not args.force and marker_path(args.output_dir, step).is_file():
            continue
        pending.append(ckpt)
    return pending


def main():
    args = parse_args()
    if args.upload_only:
        keep_alive = slurm_job_running(args.watch_job_id) or args.watch
        wandb_run = None if args.dry_run else attach_wandb(args, keep_alive=keep_alive)
        try:
            uploaded = upload_saved_artifacts(args, wandb_run)
            print(f"upload_only done: {uploaded}", flush=True)
            if args.watch:
                while True:
                    if args.watch_job_id and not slurm_job_running(args.watch_job_id):
                        upload_saved_artifacts(args, wandb_run)
                        break
                    time.sleep(args.poll_seconds)
                    upload_saved_artifacts(args, wandb_run)
        finally:
            still_training = slurm_job_running(args.watch_job_id)
            detach_wandb(wandb_run, finish=not still_training)
        return

    if not args.val_file.is_file():
        raise SystemExit(f"val file missing: {args.val_file}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda" and "A6000" in torch.cuda.get_device_name(0):
        os.environ.setdefault("NCCL_P2P_DISABLE", "1")
    enable_fast_kernels()
    print(
        f"Device: {device}  output_dir={args.output_dir}  wandb_dir={args.wandb_dir}",
        flush=True,
    )

    first = list_complete_checkpoints(args.output_dir)
    if not first and not args.watch:
        raise SystemExit(f"no complete checkpoints under {args.output_dir}")
    extra_args = {}
    if first:
        payload = torch.load(first[0] / "ltlm_extra.pt", map_location="cpu", weights_only=False)
        extra_args = payload.get("args") or {}
    dataset, loader, roll_idx = build_val(args, extra_args)
    accelerator = _SingleAccelerator(device)

    keep_alive = slurm_job_running(args.watch_job_id) or args.watch
    wandb_run = None if args.dry_run else attach_wandb(args, keep_alive=keep_alive)

    processed = []
    try:
        while True:
            todo = pending_checkpoints(args)
            if todo:
                for ckpt in todo:
                    stats = backfill_one(
                        ckpt, args, dataset, loader, roll_idx, accelerator, wandb_run
                    )
                    processed.append((str(ckpt), stats))
            if not args.watch:
                break
            if args.watch_job_id and not slurm_job_running(args.watch_job_id):
                leftover = pending_checkpoints(args)
                for ckpt in leftover:
                    stats = backfill_one(
                        ckpt, args, dataset, loader, roll_idx, accelerator, wandb_run
                    )
                    processed.append((str(ckpt), stats))
                print(f"watch job {args.watch_job_id} is gone; exiting", flush=True)
                break
            print(f"watch: sleeping {args.poll_seconds}s", flush=True)
            time.sleep(args.poll_seconds)
    finally:
        still_training = slurm_job_running(args.watch_job_id)
        detach_wandb(wandb_run, finish=not still_training)
    print(f"done: {len(processed)} checkpoint(s)", flush=True)


if __name__ == "__main__":
    main()
