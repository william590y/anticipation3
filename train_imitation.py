import argparse
import gc
import math
import os
from pathlib import Path
import random
import traceback

import numpy as np
import torch
from accelerate import Accelerator
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForCausalLM

import train as base_train
from anticipation.packed_sequence import ALTERNATING_START, iter_score_slot_positions
from anticipation.score_constraints import constrain_score_token_logits


def collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch])
    attention_mask = torch.stack([item["attention_mask"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    score_token_mask = torch.stack([item["score_token_mask"] for item in batch])
    score_mask = torch.stack([item["score_mask"] for item in batch])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "score_token_mask": score_token_mask,
        "score_mask": score_mask,
    }


def _sample_from_logits(logits, temperature):
    if temperature is None or temperature <= 0:
        return int(logits.argmax().item())

    scaled_logits = logits.float() / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    return int(torch.multinomial(probs, num_samples=1).item())


def _linear_schedule(step, start, end, decay_steps):
    if decay_steps <= 0:
        return end

    progress = min(1.0, max(0.0, float(step) / float(decay_steps)))
    return start + (end - start) * progress


def _should_apply_imitation_step(completed_steps, rollin_interval):
    if rollin_interval <= 0:
        return False
    return completed_steps % rollin_interval == 0


def _save_metrics_npz(
    destination,
    train_losses,
    train_loss_steps,
    val_losses,
    val_accuracies,
    val_autoregressive_accuracies,
    validation_steps,
    imitation_teacher_probs,
    imitation_policy_rates,
):
    np.savez(
        destination,
        train_losses=np.array(train_losses),
        train_loss_steps=np.array(train_loss_steps),
        val_losses=np.array(val_losses),
        val_accuracies=np.array(val_accuracies),
        val_autoregressive_accuracies=np.array(val_autoregressive_accuracies),
        validation_steps=np.array(validation_steps),
        imitation_teacher_probs=np.array(imitation_teacher_probs),
        imitation_policy_rates=np.array(imitation_policy_rates),
    )


def build_imitation_batch(
    policy_model,
    batch,
    teacher_prob,
    max_rollin_score_slots,
    sequence_fraction,
    constrain_score_tokens,
    rollin_temperature,
    keep_score_mask,
):
    input_ids = batch["input_ids"]
    attention_mask = batch["attention_mask"]
    labels = batch["labels"]
    score_mask = batch.get("score_mask")

    imitation_input_ids = input_ids.clone()
    if score_mask is None or not keep_score_mask:
        imitation_score_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    else:
        imitation_score_mask = score_mask.clone()

    batch_size = input_ids.size(0)
    if batch_size <= 0:
        return {
            "input_ids": imitation_input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "score_token_mask": batch["score_token_mask"],
            "score_mask": imitation_score_mask,
        }, {
            "teacher_prob": teacher_prob,
            "policy_triplets": 0,
            "teacher_triplets": 0,
            "rolled_sequences": 0,
            "selected_sequences": 0,
        }

    if sequence_fraction >= 1.0:
        selected_indices = list(range(batch_size))
    elif sequence_fraction <= 0.0:
        selected_indices = []
    else:
        count = max(1, int(round(batch_size * sequence_fraction)))
        count = min(batch_size, count)
        selected_indices = random.sample(range(batch_size), count)

    selected_index_set = set(selected_indices)
    teacher_triplets = 0
    policy_triplets = 0
    rolled_sequences = 0

    was_training = policy_model.training
    policy_model.eval()

    try:
        with torch.inference_mode():
            for batch_index in range(batch_size):
                if batch_index not in selected_index_set:
                    continue

                expert_seq = input_ids[batch_index]
                seq_len = int(attention_mask[batch_index].sum().item())
                if seq_len <= ALTERNATING_START:
                    continue

                prefix = expert_seq[:ALTERNATING_START].tolist()
                prefix_tensor = torch.tensor([prefix], device=input_ids.device, dtype=torch.long)
                primed = policy_model(prefix_tensor, use_cache=True)
                past = primed.past_key_values
                next_logits = primed.logits[0, -1, :]

                def feed_token(current_past, token):
                    output = policy_model(
                        torch.tensor([[token]], device=input_ids.device, dtype=torch.long),
                        past_key_values=current_past,
                        use_cache=True,
                    )
                    return output.past_key_values, output.logits[0, -1, :]

                rolled_any_slot = False
                slot_count = 0

                for pos in iter_score_slot_positions(seq_len, ALTERNATING_START):
                    if pos + 5 >= seq_len:
                        break
                    if max_rollin_score_slots > 0 and slot_count >= max_rollin_score_slots:
                        break

                    trial_past = past
                    trial_next_logits = next_logits
                    predicted_triplet = []
                    for slot in range(3):
                        logits = trial_next_logits
                        if constrain_score_tokens:
                            logits = constrain_score_token_logits(logits, slot)
                        token = _sample_from_logits(logits, rollin_temperature)
                        predicted_triplet.append(token)
                        trial_past, trial_next_logits = feed_token(trial_past, token)

                    if random.random() < teacher_prob:
                        chosen_triplet = expert_seq[pos:pos + 3].tolist()
                        teacher_triplets += 1
                    else:
                        chosen_triplet = predicted_triplet
                        policy_triplets += 1

                    imitation_input_ids[batch_index, pos:pos + 3] = torch.tensor(
                        chosen_triplet,
                        device=input_ids.device,
                        dtype=imitation_input_ids.dtype,
                    )

                    for token in chosen_triplet:
                        past, next_logits = feed_token(past, token)

                    control_triplet = expert_seq[pos + 3:pos + 6].tolist()
                    for token in control_triplet:
                        past, next_logits = feed_token(past, token)

                    rolled_any_slot = True
                    slot_count += 1

                if rolled_any_slot:
                    rolled_sequences += 1
    finally:
        policy_model.train(was_training)

    imitation_batch = {
        "input_ids": imitation_input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "score_token_mask": batch["score_token_mask"],
        "score_mask": imitation_score_mask,
    }
    imitation_stats = {
        "teacher_prob": teacher_prob,
        "policy_triplets": policy_triplets,
        "teacher_triplets": teacher_triplets,
        "rolled_sequences": rolled_sequences,
        "selected_sequences": len(selected_indices),
    }
    return imitation_batch, imitation_stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_file", type=Path, default=Path("./data/train_normalized.txt"))
    parser.add_argument("--val_file", type=Path, default=Path("./data/test_normalized.txt"))
    parser.add_argument("--model_name", type=str, default="stanford-crfm/music-medium-800k")
    parser.add_argument("--output_dir", type=Path, default=Path("./imitation_learning"))
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--val_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_steps", type=int, default=40000)
    parser.add_argument("--save_steps", type=int, default=2500)
    parser.add_argument("--eval_steps", type=int, default=500)
    parser.add_argument(
        "--eval_max_samples",
        type=int,
        default=500,
        help="Random validation sequences for teacher-forced eval. <= 0 uses the full validation set.",
    )
    parser.add_argument(
        "--eval_autoregressive_samples",
        type=int,
        default=100,
        help="Random validation sequences for autoregressive eval. <= 0 disables autoregressive eval.",
    )
    parser.add_argument(
        "--eval_num_workers",
        type=int,
        default=0,
        help="Dataloader workers used for sampled validation subsets.",
    )
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--force_cpu", action="store_true", help="Force CPU usage even if GPU is available")
    parser.add_argument("--reduce_memory", action="store_true", help="Use memory-saving techniques")
    parser.add_argument(
        "--onset_jitter_std",
        type=float,
        default=0.05,
        help="Std of N(1, std^2) multiplier applied to each inter-onset interval of control tokens (training only)",
    )
    parser.add_argument(
        "--dur_jitter_range",
        type=float,
        default=0.05,
        help="Half-range of U(1-r, 1+r) duration rescaling per control note (training only)",
    )
    parser.add_argument(
        "--mask_prob",
        type=float,
        default=0.5,
        help="Fraction of score triplets whose token embeddings are zeroed in the input context (training only)",
    )
    parser.add_argument(
        "--loss_mask_performance_tokens",
        action="store_true",
        help="Exclude performance/control triplets from the loss by setting their labels to -100",
    )
    parser.add_argument(
        "--transpose_range_semitones",
        type=int,
        default=12,
        help="Max transposition shift in semitones, uniform in [-range, +range] (training only)",
    )
    parser.add_argument(
        "--tempo_scale_range",
        type=float,
        default=0.2,
        help="Tempo scale half-range sampled uniformly from [1-range, 1+range] and applied only to performance/control timing (training only)",
    )
    parser.add_argument(
        "--original_weight_l2",
        type=float,
        default=1e5,
        help="Coefficient for L2 anchoring to the model weights immediately after load/resize. Set to 0 to disable.",
    )
    parser.add_argument(
        "--il_rollin_interval",
        type=int,
        default=1,
        help="Apply an on-policy imitation roll-in every N optimizer steps. Set <= 0 to disable.",
    )
    parser.add_argument(
        "--il_rollin_score_slots",
        type=int,
        default=48,
        help="Maximum score slots per selected sequence to roll in with the current policy. <= 0 means full sequence.",
    )
    parser.add_argument(
        "--il_sequence_fraction",
        type=float,
        default=1.0,
        help="Fraction of sequences in each batch that receive on-policy roll-ins.",
    )
    parser.add_argument(
        "--il_teacher_prob_start",
        type=float,
        default=1.0,
        help="Probability of rolling in the expert action at the start of training.",
    )
    parser.add_argument(
        "--il_teacher_prob_end",
        type=float,
        default=0.0,
        help="Probability of rolling in the expert action after the decay schedule completes.",
    )
    parser.add_argument(
        "--il_teacher_prob_decay_steps",
        type=int,
        default=0,
        help="Optimizer steps over which to decay the expert roll-in probability. 0 uses max_steps.",
    )
    parser.add_argument(
        "--il_rollin_temperature",
        type=float,
        default=0.0,
        help="Sampling temperature for policy roll-ins. <= 0 uses greedy argmax roll-ins.",
    )
    parser.add_argument(
        "--il_keep_score_mask",
        action="store_true",
        help="Preserve score masking during imitation batches. By default imitation batches disable score masking.",
    )
    parser.add_argument(
        "--il_no_constrain_score_tokens",
        action="store_true",
        help="Do not constrain score-slot decoding ranges during policy roll-ins.",
    )
    args = parser.parse_args()

    if args.original_weight_l2 < 0:
        raise ValueError("--original_weight_l2 must be non-negative.")
    if args.eval_num_workers < 0:
        raise ValueError("--eval_num_workers must be non-negative.")
    if args.il_rollin_interval < 0:
        raise ValueError("--il_rollin_interval must be non-negative.")
    if not 0.0 <= args.il_sequence_fraction <= 1.0:
        raise ValueError("--il_sequence_fraction must be in [0, 1].")
    if not 0.0 <= args.il_teacher_prob_start <= 1.0:
        raise ValueError("--il_teacher_prob_start must be in [0, 1].")
    if not 0.0 <= args.il_teacher_prob_end <= 1.0:
        raise ValueError("--il_teacher_prob_end must be in [0, 1].")

    base_train.validate_selected_cuda_device_or_raise(force_cpu=args.force_cpu)
    base_train.report_runtime_device(force_cpu=args.force_cpu)
    print(f"Per-rank effective batch size: {args.batch_size * args.gradient_accumulation_steps}")

    try:
        mixed_precision = "bf16" if torch.cuda.is_available() and not args.force_cpu else "no"
        print(f"Mixed precision mode: {mixed_precision}")

        accelerator = Accelerator(
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            cpu=args.force_cpu,
            mixed_precision=mixed_precision,
        )
        print(
            "Distributed setup: "
            f"type={accelerator.distributed_type}, "
            f"world_size={accelerator.num_processes}, "
            f"rank={accelerator.process_index}, "
            f"local_rank={accelerator.local_process_index}, "
            f"device={accelerator.device}"
        )
        if accelerator.is_main_process:
            print(
                "Global effective batch size: "
                f"{args.batch_size * args.gradient_accumulation_steps * accelerator.num_processes}"
            )
            print(
                "Imitation learning config: "
                f"rollin_interval={args.il_rollin_interval}, "
                f"rollin_score_slots={args.il_rollin_score_slots}, "
                f"sequence_fraction={args.il_sequence_fraction:.2f}, "
                f"teacher_prob_start={args.il_teacher_prob_start:.2f}, "
                f"teacher_prob_end={args.il_teacher_prob_end:.2f}"
            )

        if accelerator.is_main_process:
            os.makedirs(args.output_dir, exist_ok=True)
        accelerator.wait_for_everyone()

        print("Initial GPU memory stats:")
        base_train.print_gpu_memory_stats()

        print(f"Loading training dataset from {args.data_file}...")
        train_dataset = base_train.TokenizedDataset(
            args.data_file,
            onset_jitter_std=args.onset_jitter_std,
            dur_jitter_range=args.dur_jitter_range,
            mask_prob=args.mask_prob,
            transpose_range_semitones=args.transpose_range_semitones,
            tempo_scale_range=args.tempo_scale_range,
            loss_mask_performance_tokens=args.loss_mask_performance_tokens,
            is_training=True,
        )
        if len(train_dataset) == 0:
            raise ValueError(
                "Training dataset is empty. Check the tokenized training file and rerun tokenization if needed."
            )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            pin_memory=torch.cuda.is_available() and not args.force_cpu,
            num_workers=0,
        )

        print(f"Loading validation dataset from {args.val_file}...")
        val_dataset = base_train.TokenizedDataset(
            args.val_file,
            loss_mask_performance_tokens=args.loss_mask_performance_tokens,
            is_training=False,
        )
        if len(val_dataset) == 0:
            raise ValueError(
                f"Validation dataset is empty: {args.val_file}. Check the tokenized validation file."
            )

        val_loader_kwargs = {
            "batch_size": args.val_batch_size,
            "collate_fn": collate_fn,
            "pin_memory": torch.cuda.is_available() and not args.force_cpu,
            "num_workers": args.eval_num_workers,
        }

        print(f"Loading model {args.model_name}...")
        model_kwargs = {
            "trust_remote_code": True,
            "use_cache": False,
        }
        if args.reduce_memory and torch.cuda.is_available():
            print("Using memory reduction techniques...")
            model_kwargs.update(
                {
                    "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                    "low_cpu_mem_usage": True,
                }
            )

        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                **model_kwargs,
            )
        except Exception as exc:
            print(f"Error loading model with advanced options: {exc}")
            print("Trying with basic options...")
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                trust_remote_code=True,
                use_cache=False,
            )

        from anticipation.vocab import VOCAB_SIZE

        current_vocab_size = model.config.vocab_size
        if current_vocab_size != VOCAB_SIZE:
            print(f"Resizing model embeddings from {current_vocab_size} to {VOCAB_SIZE}")
            model.resize_token_embeddings(VOCAB_SIZE)
            print("Model embeddings resized successfully")
        else:
            print(f"Model vocabulary size matches tokenization ({VOCAB_SIZE})")

        print("GPU memory after loading model:")
        base_train.print_gpu_memory_stats()

        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
            eps=1e-6,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        )

        model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
        print(f"After accelerator preparation, model device: {next(model.parameters()).device}")

        policy_model = accelerator.unwrap_model(model)
        original_weight_references = {}
        if args.original_weight_l2 > 0:
            (
                original_weight_references,
                anchored_tensor_count,
                anchored_parameter_count,
                anchored_bytes,
            ) = base_train._capture_reference_parameters(policy_model)
            if accelerator.is_main_process:
                anchored_megabytes = anchored_bytes / (1024 ** 2)
                print(
                    "Original-weight L2 regularization enabled: "
                    f"lambda={args.original_weight_l2}, "
                    f"{anchored_tensor_count} tensors, "
                    f"{anchored_parameter_count:,} parameters, "
                    f"~{anchored_megabytes:.1f} MiB snapshot."
                )
        elif accelerator.is_main_process:
            print("Original-weight L2 regularization disabled.")

        initial_lr = args.learning_rate
        final_lr = 3e-6
        decay_steps = args.il_teacher_prob_decay_steps if args.il_teacher_prob_decay_steps > 0 else args.max_steps

        def lr_lambda(current_step):
            progress = float(current_step) / float(max(1, args.max_steps))
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return (final_lr / initial_lr) + (1.0 - final_lr / initial_lr) * cosine_decay

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

        print("GPU memory before training:")
        base_train.print_gpu_memory_stats()

        torch.autograd.set_detect_anomaly(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        if torch.cuda.is_available() and accelerator.device.type == "cuda":
            print(f"Clearing CUDA cache before training on {accelerator.device}")
            torch.cuda.empty_cache()
            device_index = accelerator.device.index
            if device_index is not None:
                torch.cuda.set_device(device_index)

        print("Starting imitation-learning training...")
        model.train()
        completed_steps = 0
        train_losses = []
        train_loss_steps = []
        val_losses = []
        val_accuracies = []
        val_autoregressive_accuracies = []
        validation_steps = []
        imitation_teacher_probs = []
        imitation_policy_rates = []

        progress_bar = tqdm(total=args.max_steps, desc="Training", disable=not accelerator.is_main_process)
        training_failed = False

        def run_validation(validation_label):
            validation_step = int(completed_steps)
            accelerator.wait_for_everyone()

            if accelerator.is_main_process:
                print(f"\nRunning validation at {validation_label}...")

            val_loss, val_acc, val_auto_acc = base_train.evaluate_model(
                model,
                accelerator,
                val_dataset,
                **val_loader_kwargs,
                max_samples=args.eval_max_samples,
                autoregressive_samples=args.eval_autoregressive_samples,
            )

            if accelerator.is_main_process:
                validation_steps.append(validation_step)
                val_losses.append(val_loss)
                val_accuracies.append(val_acc * 100)
                val_autoregressive_accuracies.append(val_auto_acc * 100)
                print(
                    f"Validation Loss: {val_loss:.4f}, "
                    f"Teacher-Forced Accuracy: {val_acc * 100:.2f}%, "
                    f"Autoregressive Accuracy: {val_auto_acc * 100:.2f}%"
                )

            accelerator.wait_for_everyone()
            model.train()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                gc.collect()

        try:
            while completed_steps < args.max_steps:
                for batch in train_dataloader:
                    try:
                        with accelerator.accumulate(model):
                            teacher_prob = _linear_schedule(
                                completed_steps,
                                args.il_teacher_prob_start,
                                args.il_teacher_prob_end,
                                decay_steps,
                            )
                            use_imitation = _should_apply_imitation_step(
                                completed_steps,
                                args.il_rollin_interval,
                            )
                            imitation_stats = {
                                "teacher_prob": teacher_prob,
                                "policy_triplets": 0,
                                "teacher_triplets": 0,
                                "rolled_sequences": 0,
                                "selected_sequences": 0,
                            }

                            if use_imitation:
                                effective_rollin_slots = args.il_rollin_score_slots
                                if effective_rollin_slots <= 0:
                                    effective_rollin_slots = 10**9
                                training_batch, imitation_stats = build_imitation_batch(
                                    policy_model=policy_model,
                                    batch=batch,
                                    teacher_prob=teacher_prob,
                                    max_rollin_score_slots=effective_rollin_slots,
                                    sequence_fraction=args.il_sequence_fraction,
                                    constrain_score_tokens=not args.il_no_constrain_score_tokens,
                                    rollin_temperature=args.il_rollin_temperature,
                                    keep_score_mask=args.il_keep_score_mask,
                                )
                            else:
                                training_batch = batch

                            outputs = base_train.forward_batch(model, training_batch)
                            loss = outputs.loss
                            l2_penalty = None
                            if original_weight_references:
                                l2_penalty = base_train._compute_original_weight_l2_penalty(
                                    policy_model,
                                    original_weight_references,
                                )
                                loss = loss + args.original_weight_l2 * l2_penalty

                            local_invalid_loss = bool(torch.isnan(loss).any() or torch.isinf(loss).any())
                            invalid_loss_processes = base_train._count_flagged_processes(accelerator, local_invalid_loss)
                            if invalid_loss_processes > 0:
                                if accelerator.is_main_process:
                                    print(
                                        f"WARNING: NaN or Inf loss detected on {invalid_loss_processes}/"
                                        f"{accelerator.num_processes} rank(s); skipping this synchronized step."
                                    )
                                optimizer.zero_grad()
                                continue

                            accelerator.backward(loss)

                            if accelerator.sync_gradients:
                                invalid_grad_name = base_train._find_invalid_gradient_parameter(model)
                                invalid_grad_processes = base_train._count_flagged_processes(
                                    accelerator,
                                    invalid_grad_name is not None,
                                )
                                if invalid_grad_processes > 0:
                                    if accelerator.is_main_process:
                                        detail = f" Example parameter: {invalid_grad_name}." if invalid_grad_name else ""
                                        print(
                                            f"WARNING: NaN or Inf gradients detected on {invalid_grad_processes}/"
                                            f"{accelerator.num_processes} rank(s); skipping optimizer step.{detail}"
                                        )
                                    optimizer.zero_grad()
                                    continue

                                accelerator.clip_grad_norm_(model.parameters(), max_norm=2.0)
                                optimizer.step()
                                scheduler.step()
                                optimizer.zero_grad()

                                reduced_loss = accelerator.reduce(
                                    loss.detach().to(device=accelerator.device, dtype=torch.float64),
                                    reduction="mean",
                                ).item()
                                reduced_l2_penalty = None
                                reduced_anchor_term = None
                                if l2_penalty is not None:
                                    reduced_l2_penalty = accelerator.reduce(
                                        l2_penalty.detach().to(device=accelerator.device, dtype=torch.float64),
                                        reduction="mean",
                                    ).item()
                                    reduced_anchor_term = args.original_weight_l2 * reduced_l2_penalty

                                policy_triplets = int(
                                    accelerator.reduce(
                                        torch.tensor(
                                            imitation_stats["policy_triplets"],
                                            device=accelerator.device,
                                            dtype=torch.float64,
                                        ),
                                        reduction="sum",
                                    ).item()
                                )
                                teacher_triplets = int(
                                    accelerator.reduce(
                                        torch.tensor(
                                            imitation_stats["teacher_triplets"],
                                            device=accelerator.device,
                                            dtype=torch.float64,
                                        ),
                                        reduction="sum",
                                    ).item()
                                )
                                total_rollin_triplets = policy_triplets + teacher_triplets
                                policy_rate = (
                                    policy_triplets / total_rollin_triplets if total_rollin_triplets > 0 else 0.0
                                )

                                completed_steps += 1
                                progress_bar.update(1)

                                if completed_steps % 10 == 0 and accelerator.is_main_process:
                                    train_losses.append(reduced_loss)
                                    train_loss_steps.append(completed_steps)
                                    imitation_teacher_probs.append(teacher_prob)
                                    imitation_policy_rates.append(policy_rate)

                                    l2_detail = ""
                                    if reduced_l2_penalty is not None:
                                        l2_detail = (
                                            f", AnchorL2: {reduced_l2_penalty:.6e}, "
                                            f"AnchorTerm: {reduced_anchor_term:.6e}"
                                        )

                                    imitation_detail = ""
                                    if use_imitation:
                                        imitation_detail = (
                                            f", IL teacher_prob: {teacher_prob:.3f}, "
                                            f"IL policy_triplet_rate: {policy_rate:.3f}, "
                                            f"IL policy_triplets: {policy_triplets}"
                                        )

                                    print(
                                        f"Step: {completed_steps}/{args.max_steps}, Loss: {reduced_loss:.4f}, "
                                        f"LR: {scheduler.get_last_lr()[0]:.8e}{l2_detail}{imitation_detail}"
                                    )

                                    if base_train.check_model_for_nans(model):
                                        print("NaN parameters detected in model! Training may be unstable.")

                                    if completed_steps % 100 == 0:
                                        base_train.print_gpu_memory_stats()

                                is_checkpoint_step = completed_steps % args.save_steps == 0
                                if completed_steps % args.eval_steps == 0 and not is_checkpoint_step:
                                    run_validation(f"step {completed_steps}")

                                if is_checkpoint_step:
                                    run_validation(f"checkpoint step {completed_steps}")

                                    checkpoint_dir = args.output_dir / f"checkpoint-{completed_steps}"
                                    if accelerator.is_main_process:
                                        os.makedirs(checkpoint_dir, exist_ok=True)
                                    accelerator.wait_for_everyone()

                                    unwrapped_model = accelerator.unwrap_model(model)
                                    unwrapped_model.save_pretrained(
                                        checkpoint_dir,
                                        is_main_process=accelerator.is_main_process,
                                        save_function=accelerator.save,
                                    )
                                    accelerator.wait_for_everyone()

                                    if accelerator.is_main_process:
                                        print(f"Saved checkpoint to {checkpoint_dir}")
                                        _save_metrics_npz(
                                            checkpoint_dir / "losses.npz",
                                            train_losses,
                                            train_loss_steps,
                                            val_losses,
                                            val_accuracies,
                                            val_autoregressive_accuracies,
                                            validation_steps,
                                            imitation_teacher_probs,
                                            imitation_policy_rates,
                                        )
                                        base_train.plot_losses(
                                            train_losses,
                                            train_loss_steps,
                                            val_losses,
                                            val_accuracies,
                                            val_autoregressive_accuracies,
                                            validation_steps,
                                            checkpoint_dir,
                                        )
                                    accelerator.wait_for_everyone()

                                    if torch.cuda.is_available():
                                        torch.cuda.empty_cache()
                                        gc.collect()

                            if completed_steps >= args.max_steps:
                                break

                    except RuntimeError as exc:
                        if "CUDA out of memory" in str(exc):
                            print(f"CUDA OOM error! Current batch size: {args.batch_size}")
                            print("Current memory usage:")
                            base_train.print_gpu_memory_stats()
                            print("Consider reducing batch size, il_sequence_fraction, or il_rollin_score_slots.")
                            print(f"Error details: {str(exc)}")
                            raise
                        if "nan" in str(exc).lower() or "inf" in str(exc).lower():
                            print(f"NaN/Inf error: {str(exc)}")
                            if accelerator.num_processes > 1:
                                print("Distributed run detected; aborting instead of skipping locally to avoid rank desync.")
                                raise
                            print("Trying to recover by skipping this batch...")
                            optimizer.zero_grad()
                            continue

                        print(f"Runtime error: {str(exc)}")
                        print(traceback.format_exc())
                        raise

        except Exception as exc:
            training_failed = True
            print(f"Error during training: {exc}")
            print(traceback.format_exc())
            raise
        finally:
            progress_bar.close()

            if training_failed:
                if accelerator.is_main_process:
                    print("Skipping final validation/save because training exited with an error.")
            else:
                try:
                    run_validation(f"final step {completed_steps}")

                    final_dir = args.output_dir / "final"
                    if accelerator.is_main_process:
                        os.makedirs(final_dir, exist_ok=True)
                    accelerator.wait_for_everyone()

                    unwrapped_model = accelerator.unwrap_model(model)
                    unwrapped_model.save_pretrained(
                        final_dir,
                        is_main_process=accelerator.is_main_process,
                        save_function=accelerator.save,
                    )
                    accelerator.wait_for_everyone()

                    if accelerator.is_main_process:
                        print(f"Saved final model to {final_dir}")
                        _save_metrics_npz(
                            final_dir / "losses.npz",
                            train_losses,
                            train_loss_steps,
                            val_losses,
                            val_accuracies,
                            val_autoregressive_accuracies,
                            validation_steps,
                            imitation_teacher_probs,
                            imitation_policy_rates,
                        )
                        base_train.plot_losses(
                            train_losses,
                            train_loss_steps,
                            val_losses,
                            val_accuracies,
                            val_autoregressive_accuracies,
                            validation_steps,
                            final_dir,
                        )
                    accelerator.wait_for_everyone()

                except Exception as save_error:
                    print(f"Error saving final model or generating plot: {save_error}")

    except Exception as setup_error:
        print(f"Error in setup: {setup_error}")
        print(traceback.format_exc())


if __name__ == "__main__":
    main()
