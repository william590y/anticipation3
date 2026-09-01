#!/usr/bin/env python
"""Precompute aligned-native-GT-seeded rollouts for Paper 1 and Paper 2.

The released models are autoregressive encoder/decoders and both public
``generate`` methods accept an output-token prefix.  Each visual GT note is
certified back through the aligned-stream cache to a unique score-MIDI note and
then to the paper tokenizer's unique original MusicXML-native target row.  The
intervention copies that row's native offset/downbeat/duration/pitch and non-pad
mask exactly.  This avoids trying to invert the visualizer's beat-normalized grid
through one whole-piece meter scalar.  Ancillary notation streams remain freshly
model-generated.  Only if the sampled row was pad-masked (and its ancillary
values were consequently zeroed) are those ancillaries filled from the same
uniquely aligned official MusicXML row.

Inference preserves each paper's official full-piece chunk phase and overlap:
Paper 1 uses chunk=256/overlap=64; Paper 2 uses 512/64.  Chunks are replayed from
the beginning of the piece.  Within the branch chunk, generation stops at each
seed, the complete row is replaced, and generation resumes, so even unmatched
notes between seed 1 and seed 2 are causally regenerated.

Four Slurm array tasks each process six of the 24 windows.  Shards are validated
and merged only by ``publish_paper_seed_rollouts.py``.
"""
from __future__ import annotations

import argparse
import contextlib
import gc
import glob
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import torch


REPO = Path(__file__).resolve().parent.parent
VIS = REPO / "visualizer"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(VIS))

from alignment import align_tokens2, load_annotation_file  # noqa: E402
from anticipation.asap_aligned_stream import (  # noqa: E402
    normalize_score_triplet_to_fixed_beat,
)
from anticipation.convert import midi_to_events  # noqa: E402
from anticipation.vocab import (  # noqa: E402
    ADUR_OFFSET,
    ANOTE_OFFSET,
    ATIME_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    TIME_OFFSET,
)
from atomic_json import atomic_dump_json  # noqa: E402
from compute_sequence_ppl import load_payload  # noqa: E402
from precompute_visualizer import locate_window  # noqa: E402
from ppo_pipeline_common import ordered_example_ids  # noqa: E402
from run_paper_models import (  # noqa: E402
    _add_repo_to_path,
    decode_score_streams,
    quarters_per_annotated_beat,
    working_dir,
)


SEED_COUNTS = (1, 2, 3, 4, 5)
SEEDED_CONDITIONING = (
    "autoregressive aligned GT note in paper-native timing/quantization"
)
MODEL_SPECS = {
    "paper1": {
        "repo": "external/paper1_joint_apt_epr",
        "checkpoint": "external/weights/joint_apt_epr.ckpt",
        "identity": "Zeng+ joint-APT-EPR (ICLR 2026 released weights)",
        "chunk": 256,
        "overlap": 64,
    },
    "paper2": {
        "repo": "external/paper2_midi2score",
        "checkpoint": "external/weights/MIDI2ScoreTF.ckpt",
        "identity": "Beyer & Dai MIDI2ScoreTransformer (ISMIR 2024 released weights)",
        "chunk": 512,
        "overlap": 64,
    },
}


def variant_name(count: int) -> str:
    return "filtered_seeded" if count == 1 else f"filtered_seed{count}"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def file_identity(path: Path, *, include_sha256: bool = True) -> dict:
    path = path.resolve()
    stat = path.stat()
    out = {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if include_sha256:
        out["sha256"] = sha256_file(path)
    return out


def source_data_identity(payload: dict) -> str:
    """Hash only paper rollout inputs, ignoring independent publications."""
    order = ordered_example_ids(payload)
    examples = payload["examples"]
    identity = {
        "example_order": order,
        "examples": {
            eid: {
                "piece": examples[eid].get("piece"),
                "perf_notes": examples[eid].get("perf_notes"),
                "raw_notes": examples[eid].get("raw_notes"),
                "gt_score": examples[eid].get("gt_score"),
            }
            for eid in order
        },
    }
    encoded = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return "sha256:" + hashlib.sha256(encoded).hexdigest()


def load_cache_pieces(cache_dir: Path) -> list[dict]:
    pieces = []
    for cache_path in sorted(glob.glob(str(cache_dir / "*.json"))):
        try:
            payload = json.loads(Path(cache_path).read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        fingerprint = payload.get("fingerprint", {})
        raw, matched, scores = [], [], []
        for item in payload.get("items", []):
            control = item["control"]
            raw.append(
                (
                    int(control[0] - ATIME_OFFSET),
                    int(control[1] - ADUR_OFFSET),
                    int(control[2] - ANOTE_OFFSET),
                )
            )
            score = item.get("score")
            matched.append(score is not None)
            scores.append(
                None
                if score is None
                else {
                    "t": int(score[0] - TIME_OFFSET),
                    "d": int(score[1] - DUR_OFFSET),
                    "p": int(score[2] - NOTE_OFFSET),
                }
            )
        filtered_to_raw = [i for i, value in enumerate(matched) if value]
        pieces.append(
            {
                "cache_path": str(Path(cache_path).resolve()),
                "piece_id": fingerprint.get("perf_midi", {}).get("path"),
                "score_midi": fingerprint.get("score_midi", {}).get("path"),
                "perf_beats": fingerprint.get("perf_beats", {}).get("path"),
                "score_beats": fingerprint.get("score_beats", {}).get("path"),
                "raw": raw,
                "matched": matched,
                "scores": scores,
                "filtered_to_raw": filtered_to_raw,
                "pitch_bytes": bytes(raw[i][2] % 256 for i in filtered_to_raw),
            }
        )
    return pieces


def load_external_model(kind: str, checkpoint: Path, device: torch.device):
    repo = REPO / MODEL_SPECS[kind]["repo"]
    _add_repo_to_path(repo, "midi2scoretransformer" if kind == "paper2" else None)
    external_roots = tuple(
        str((REPO / "external" / name).resolve())
        for name in ("paper1_joint_apt_epr", "paper2_midi2score")
    )
    for name, cached in list(sys.modules.items()):
        origin = getattr(cached, "__file__", None) or ""
        if origin.startswith(external_roots):
            del sys.modules[name]

    import config as config_mod
    import tokenizer as tokenizer_mod
    import utils as utils_mod

    if hasattr(config_mod, "MyModelConfig"):
        torch.serialization.add_safe_globals([config_mod.MyModelConfig])
    with working_dir(repo):
        if kind == "paper2":
            from models.roformer import Roformer

            model = Roformer.load_from_checkpoint(str(checkpoint), map_location=device)
        else:
            from config import MyModelConfig
            from train import JointModel

            common = dict(
                num_hidden_layers=6,
                hidden_size=512,
                intermediate_size=3072,
                num_attention_heads=8,
            )
            model = JointModel.load_from_checkpoint(
                str(checkpoint),
                enc_config=MyModelConfig(**common),
                style_enc_config=MyModelConfig(**common, is_style_encoder=True),
                dec_config=MyModelConfig(
                    **common,
                    is_decoder=True,
                    is_autoregressive=True,
                    add_cross_attention=True,
                ),
                map_location=device,
            )
    return model.to(device).eval(), tokenizer_mod, utils_mod


def prepare_full_input(kind: str, tokenized: dict[str, torch.Tensor], device):
    if kind == "paper1":
        return {
            key: ((value + 1) if key != "pad" else value).unsqueeze(0).to(device)
            for key, value in tokenized.items()
        }
    return {key: value.unsqueeze(0).to(device) for key, value in tokenized.items()}


def prepare_full_gt(kind: str, tokenized: dict[str, torch.Tensor], device):
    if kind == "paper1":
        return {
            key: ((value + 1) if key != "pad" else value).unsqueeze(0).to(device)
            for key, value in tokenized.items()
        }
    return {key: value.unsqueeze(0).to(device) for key, value in tokenized.items()}


def prefix_for_generate(kind: str, output: dict[str, torch.Tensor], length: int | None = None):
    prefix = {
        key: (value if length is None else value[:, :length]).clone()
        for key, value in output.items()
    }
    if kind == "paper2" and prefix["pad"].ndim == 3:
        prefix["pad"] = prefix["pad"][..., 0].long()
    return prefix


def pad_value_at(kind: str, output: dict[str, torch.Tensor], position: int) -> torch.Tensor:
    """Return one scalar validity value from a BxT or BxTx1 pad stream."""
    row = output["pad"][0, position]
    if row.numel() != 1:
        raise ValueError(f"{kind} pad row is not scalar: shape={tuple(row.shape)}")
    return row.reshape(-1)[0]


def set_pad_value(
    kind: str, output: dict[str, torch.Tensor], position: int, value: int
) -> None:
    """Set one scalar validity value while preserving the paper's pad layout."""
    row = output["pad"][0, position]
    if row.numel() != 1:
        raise ValueError(f"{kind} pad row is not scalar: shape={tuple(row.shape)}")
    row.reshape(-1)[0] = value


def clone_output(output: dict[str, torch.Tensor] | None):
    return None if output is None else {key: value.clone() for key, value in output.items()}


def concat_output(left, right):
    if left is None:
        return clone_output(right)
    return {key: torch.cat((left[key], right[key]), dim=1) for key in left}


def slice_output(output, start: int, end: int | None = None):
    return {key: value[:, start:end].clone() for key, value in output.items()}


def force_triplet_row(
    kind: str,
    generated: dict[str, torch.Tensor],
    local_position: int,
    gt_tokens: dict[str, torch.Tensor],
    xml_row: int,
) -> dict:
    # Paper 1's tokenizer exposes an XML velocity stream that its APT decoder
    # intentionally does not generate.  Every generated stream must have a GT
    # counterpart for fallback, but tokenizer-only extras are harmless.
    missing = sorted(set(generated) - set(gt_tokens))
    if missing:
        raise ValueError(f"decoder streams missing from GT tokens: {missing}")
    triplet_streams = {"offset", "downbeat", "duration", "pitch"}
    ancillary = sorted(set(generated) - triplet_streams - {"pad"})
    sampled_valid = bool(float(pad_value_at(kind, generated, local_position)) > 0.5)
    before = {key: generated[key][0, local_position].clone() for key in ancillary}
    for key in triplet_streams:
        generated[key][0, local_position] = gt_tokens[key][0, xml_row]
        if not torch.equal(
            generated[key][0, local_position], gt_tokens[key][0, xml_row]
        ):
            raise ValueError(f"native GT token assignment failed for stream {key}")
    set_pad_value(kind, generated, local_position, 1)
    if not pad_is_valid(kind, generated, local_position):
        raise ValueError("native GT token assignment failed for pad stream")

    used_xml_fallback = not sampled_valid
    if used_xml_fallback:
        for key in ancillary:
            generated[key][0, local_position] = gt_tokens[key][0, xml_row]
            if not torch.equal(generated[key][0, local_position], gt_tokens[key][0, xml_row]):
                raise ValueError(f"pad fallback failed for ancillary stream {key}")
    else:
        for key in ancillary:
            if not torch.equal(generated[key][0, local_position], before[key]):
                raise ValueError(f"non-pad ancillary stream {key} changed during intervention")
    return {
        "sampled_row_was_nonpad": sampled_valid,
        "xml_ancillary_fallback": used_xml_fallback,
        "ancillary_streams": ancillary,
        "ancillaries_preserved": True if sampled_valid else None,
        "fallback_matches_xml": True if used_xml_fallback else None,
        "native_core_token_assignment_exact": True,
        "native_pad_assignment_exact": True,
    }


def generation_context(device: torch.device, chunk_start: int):
    # This matches both papers' official infer(): the initial chunk runs at the
    # model dtype, subsequent overlap-carried chunks run under device autocast.
    if chunk_start == 0:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type)


@torch.inference_mode()
def generate_chunk(
    kind: str,
    model,
    x_chunk: dict[str, torch.Tensor],
    carry,
    chunk_start: int,
    chunk_size: int,
    force_rows: list[tuple[int, int]],
    gt_tokens,
    device,
):
    """Official chunk generation, stopping/forcing sequentially at target rows."""
    current = clone_output(carry)
    certifications = []
    with generation_context(device, chunk_start):
        for local_position, xml_row in force_rows:
            prefix = None if current is None else prefix_for_generate(kind, current)
            current = model.generate(
                x=x_chunk,
                y=prefix,
                top_k=1,
                max_length=local_position + 1,
                kv_cache=True,
            )
            certification = force_triplet_row(
                kind, current, local_position, gt_tokens, xml_row
            )
            certification["local_position"] = local_position
            certification["musicxml_token_row"] = xml_row
            certifications.append(certification)
        prefix = None if current is None else prefix_for_generate(kind, current)
        current = model.generate(
            x=x_chunk,
            y=prefix,
            top_k=1,
            max_length=chunk_size,
            kv_cache=True,
        )
    return current, certifications


def chunk_starts(n_notes: int, chunk: int, overlap: int) -> list[int]:
    return list(range(0, max(n_notes - overlap, 1), chunk - overlap))


def owner_chunk_index(position: int, starts: list[int], chunk: int, overlap: int) -> int:
    for index, start in enumerate(starts):
        owned_lo = start if index == 0 else start + overlap
        if owned_lo <= position < start + chunk:
            return index
    raise ValueError(f"performance position {position} is not owned by an official chunk")


@torch.inference_mode()
def replay_canonical_prefix(kind, model, full_input, first_chunk_index: int, device):
    spec = MODEL_SPECS[kind]
    starts = chunk_starts(full_input["pitch"].shape[1], spec["chunk"], spec["overlap"])
    carry = None
    for index in range(first_chunk_index):
        start = starts[index]
        x_chunk = {key: value[:, start : start + spec["chunk"]] for key, value in full_input.items()}
        output, _certifications = generate_chunk(
            kind,
            model,
            x_chunk,
            carry,
            start,
            spec["chunk"],
            [],
            None,
            device,
        )
        carry = slice_output(output, -spec["overlap"], None)
    return carry


def reference_to_batched(kind: str, reference: dict[str, torch.Tensor]):
    # Paper 1 is called with an explicitly batched shifted input and retains B;
    # Paper 2 is called with its native unbatched one-hot input and removes B.
    pitch_ndim = reference["pitch"].ndim
    if kind == "paper1":
        if pitch_ndim != 2 or reference["pitch"].shape[0] != 1:
            raise ValueError(f"unexpected Paper 1 reference shape {reference['pitch'].shape}")
        return reference
    if pitch_ndim != 2:
        raise ValueError(f"unexpected Paper 2 reference shape {reference['pitch'].shape}")
    return {key: value.unsqueeze(0) for key, value in reference.items()}


def merged_midi_note_ticks(path: str) -> tuple[int, list[dict]]:
    """Note-on order used by project MIDI conversion, retaining exact MIDI ticks."""
    import mido

    midi = mido.MidiFile(path)
    absolute = 0
    notes = []
    open_notes = {}
    for message in mido.merge_tracks(midi.tracks):
        absolute += message.time
        if message.type == "note_on" and message.velocity > 0:
            index = len(notes)
            notes.append({"onset_tick": absolute, "duration_tick": None, "pitch": message.note})
            open_notes.setdefault((message.channel, message.note), []).append(index)
        elif message.type in ("note_on", "note_off"):
            queue = open_notes.get((message.channel, message.note)) or []
            if queue:
                index = queue.pop(0)
                notes[index]["duration_tick"] = absolute - notes[index]["onset_tick"]
    if any(note["duration_tick"] is None for note in notes):
        raise ValueError(f"unterminated score-MIDI notes in {path}")
    return midi.ticks_per_beat, notes


def map_raw_input_positions(
    kind: str, tokenizer_mod, piece: dict, target_perf: list[int]
) -> tuple[list[int], list[dict]]:
    """Map project/cache order to paper input order, allowing chord permutations.

    Paper 1's Partitura loader realizes pedal in ``sound_off`` and therefore
    changes MIDI note durations; onset+pitch is its stable identity.  Paper 2's
    PrettyMIDI path preserves duration, so require all three fields there.
    """
    parsed = tokenizer_mod.MultistreamTokenizer.parse_midi(piece["piece_id"])
    project_events = midi_to_events(piece["piece_id"], quantize=False)
    project = [
        {
            "onset_s": float(project_events[3 * index] - TIME_OFFSET) / 100.0,
            "duration_s": float(project_events[3 * index + 1] - DUR_OFFSET) / 100.0,
            "pitch": int(project_events[3 * index + 2] - NOTE_OFFSET),
        }
        for index in range(len(project_events) // 3)
    ]
    if len(parsed["pitch"]) != len(project) or len(project) != len(piece["raw"]):
        raise ValueError(
            "paper tokenizer/project/cache performance lengths differ: "
            f"{len(parsed['pitch'])}/{len(project)}/{len(piece['raw'])}"
        )
    mapped = []
    certifications = []
    for perf_index in target_perf:
        row = project[perf_index]
        cache_t, cache_d, cache_p = piece["raw"][perf_index]
        if (
            abs(row["onset_s"] * 100.0 - cache_t) > 1.1
            or abs(row["duration_s"] * 100.0 - cache_d) > 1.1
            or row["pitch"] != cache_p
        ):
            raise ValueError(
                f"project/cache raw input mismatch at {perf_index}: project={row}, "
                f"cache={(cache_t,cache_d,cache_p)}"
            )
        candidates = []
        for index in range(len(parsed["pitch"])):
            stable = (
                int(parsed["pitch"][index]) == row["pitch"]
                and abs(float(parsed["onset"][index]) - row["onset_s"]) <= 0.011
            )
            if kind == "paper2":
                stable = stable and (
                    abs(float(parsed["duration"][index]) - row["duration_s"]) <= 0.011
                )
            if stable:
                candidates.append(index)
        if len(candidates) != 1:
            raise ValueError(
                f"raw input {perf_index} has {len(candidates)} matches in paper tokenizer order"
            )
        mapped.append(candidates[0])
        certifications.append(
            {
                "paper_input_order_uniquely_mapped": True,
                "matching_fields": (
                    ["onset_seconds", "pitch"]
                    if kind == "paper1"
                    else ["onset_seconds", "duration_seconds", "pitch"]
                ),
                "paper1_pedal_realized_duration_excluded": kind == "paper1",
            }
        )
    if len(set(mapped)) != len(mapped):
        raise ValueError("multiple cache performance notes mapped to one paper tokenizer row")
    return mapped, certifications


def prepare_gt_rows(
    kind,
    tokenizer_mod,
    piece: dict,
    target_project_perf: list[int],
    ex: dict,
    device,
):
    """Map cache GT -> exact aligned score row -> paper order -> MusicXML row."""
    raw_events = midi_to_events(piece["score_midi"], quantize=False)
    project_raw_score = [
        {
            "onset_s": float(raw_events[3 * index] - TIME_OFFSET) / 100.0,
            "duration_s": float(raw_events[3 * index + 1] - DUR_OFFSET) / 100.0,
            "pitch": int(raw_events[3 * index + 2] - NOTE_OFFSET),
        }
        for index in range(len(raw_events) // 3)
    ]
    # Recompute the same deterministic association used to build the cache.  The
    # returned score triplet has its onset rounded to a 10 ms token *before*
    # beat-normalization; the cache builder normalizes that rounded triplet.  Do
    # not compare it to a separately normalized fractional MIDI onset (those can
    # legitimately differ by one visualizer bin).  Instead certify the exact
    # cache-builder path, then independently certify the returned raw score index
    # against the original MIDI row within only that known rounding tolerance.
    aligned = align_tokens2(
        piece["piece_id"],
        piece["score_midi"],
        piece["perf_beats"],
        piece["score_beats"],
        skip_Nones=False,
    )
    if len(aligned) != len(piece["raw"]):
        raise ValueError("recomputed alignment/cache performance lengths differ")
    score_beat_times = [
        annotation[0] for annotation in load_annotation_file(piece["score_beats"])
    ]
    aligned_score_rows = {}
    for _control, perf_index, score_triplet, score_index in aligned:
        if perf_index in aligned_score_rows:
            raise ValueError(f"duplicate aligned performance index {perf_index}")
        aligned_score_rows[perf_index] = (score_triplet, score_index)
        cached = piece["scores"][perf_index]
        if (score_index is None) != (cached is None):
            raise ValueError(f"recomputed alignment/cache match state differs at {perf_index}")
        if score_index is None:
            continue
        rebuilt = normalize_score_triplet_to_fixed_beat(
            score_triplet, score_beat_times
        )
        rebuilt = {
            "t": int(rebuilt[0] - TIME_OFFSET),
            "d": int(rebuilt[1] - DUR_OFFSET),
            "p": int(rebuilt[2] - NOTE_OFFSET),
        }
        if rebuilt != cached:
            raise ValueError(
                f"recomputed cache-builder alignment differs at {perf_index}: "
                f"rebuilt={rebuilt}, cached={cached}"
            )
        if not 0 <= score_index < len(project_raw_score):
            raise ValueError(f"aligned score index is out of range at {perf_index}")
        indexed = project_raw_score[score_index]
        aligned_raw = {
            "onset_s": float(score_triplet[0] - TIME_OFFSET) / 100.0,
            "duration_s": float(score_triplet[1] - DUR_OFFSET) / 100.0,
            "pitch": int(score_triplet[2] - NOTE_OFFSET),
        }
        if (
            abs(indexed["onset_s"] - aligned_raw["onset_s"]) > 0.0051
            or abs(indexed["duration_s"] - aligned_raw["duration_s"]) > 1e-9
            or indexed["pitch"] != aligned_raw["pitch"]
        ):
            raise ValueError(
                f"aligned raw score index differs beyond token rounding at {perf_index}: "
                f"indexed={indexed}, aligned={aligned_raw}"
            )

    score_parsed = tokenizer_mod.MultistreamTokenizer.parse_midi(piece["score_midi"])
    xml_path = Path(piece["score_midi"]).parent / "xml_score.musicxml"
    xml_list, _xml_keepalive = tokenizer_mod.MultistreamTokenizer.mxl_to_list(str(xml_path))
    # ``music21`` Notes are stateful and retain an active-site hierarchy.  The
    # tokenizer parses the file a second time inside ``parse_mxl``; keep mapping
    # identity in plain immutable values *before* that second parse so no later
    # hierarchy/cache activity can change the onset, duration, pitch, or measure
    # context being certified.
    # Capture every flattened/global note value before *any* context lookup:
    # ``getContextByClass`` changes a note's active site and therefore changes
    # what ``note.offset`` means from absolute-score to measure-relative time.
    xml_core_identity = [
        (
            float(note.offset),
            float(note.duration.quarterLength),
            int(note.pitch.midi),
        )
        for note in xml_list
    ]
    xml_identity = []
    for index, (note, (onset_q, duration_q, pitch)) in enumerate(
        zip(xml_list, xml_core_identity)
    ):
        measure = note.getContextByClass("Measure")
        if measure is None:
            raise ValueError(f"MusicXML row {index} has no measure context")
        xml_identity.append(
            {
                "onset_q": onset_q,
                "duration_q": duration_q,
                "pitch": pitch,
                "measure_offset_q": float(measure.offset),
                "measure_length_q": float(measure.barDuration.quarterLength),
            }
        )
    xml_streams = tokenizer_mod.MultistreamTokenizer.parse_mxl(str(xml_path))
    if len(xml_identity) != len(xml_streams["pitch"]):
        raise ValueError("MusicXML list/token lengths differ")
    for index, identity in enumerate(xml_identity):
        parsed_offset = float(xml_streams["offset"][index])
        expected_offset = identity["onset_q"] - identity["measure_offset_q"]
        parsed_duration = float(xml_streams["duration"][index])
        parsed_pitch = int(round(float(xml_streams["pitch"][index])))
        if (
            abs(parsed_offset - expected_offset) > 1e-5
            or abs(parsed_duration - identity["duration_q"]) > 1e-5
            or parsed_pitch != identity["pitch"]
        ):
            raise ValueError(
                f"MusicXML row {index} differs between identity and token parses: "
                f"identity={identity}, parsed_offset={parsed_offset}, "
                f"parsed_duration={parsed_duration}, parsed_pitch={parsed_pitch}"
            )
        if "abs_offset" in xml_streams and (
            abs(float(xml_streams["abs_offset"][index]) - identity["onset_q"]) > 1e-5
        ):
            raise ValueError(
                f"MusicXML row {index} absolute onset differs between parses"
            )
    ticks_per_quarter, tick_rows = merged_midi_note_ticks(piece["score_midi"])
    if len(tick_rows) != len(project_raw_score):
        raise ValueError("merged-tick/project score-MIDI note counts differ")

    xml_rows = []
    score_indices = []
    project_score_indices = []
    mapping_certifications = []
    anchor_cache_t = None
    for slot, perf_index in enumerate(target_project_perf[: max(SEED_COUNTS)]):
        cache_score = piece["scores"][perf_index]
        score_row = aligned_score_rows.get(perf_index)
        project_index = None if score_row is None else score_row[1]
        if cache_score is None or project_index is None:
            raise ValueError(f"seed slot {slot}: aligned-stream GT is missing")
        project_row = project_raw_score[project_index]
        paper_candidates = [
            index
            for index in range(len(score_parsed["pitch"]))
            if int(score_parsed["pitch"][index]) == project_row["pitch"]
            and abs(float(score_parsed["onset"][index]) - project_row["onset_s"]) <= 0.011
            and abs(float(score_parsed["duration"][index]) - project_row["duration_s"])
            <= 0.011
        ]
        if len(paper_candidates) != 1:
            raise ValueError(
                f"seed slot {slot}: project score row {project_index} has "
                f"{len(paper_candidates)} matches in the paper tokenizer's score order"
            )
        score_index = paper_candidates[0]
        gt = ex["gt_score"][slot]
        if anchor_cache_t is None:
            anchor_cache_t = int(cache_score["t"])
        expected_visual_gt = {
            "t": int(cache_score["t"]) - anchor_cache_t,
            "d": int(cache_score["d"]),
            "p": int(cache_score["p"]),
        }
        if (
            not isinstance(gt, dict)
            or int(score_parsed["pitch"][score_index]) != cache_score["p"]
            or any(int(gt[key]) != expected_visual_gt[key] for key in ("t", "d", "p"))
        ):
            raise ValueError(
                f"seed slot {slot}: score-MIDI/cache/window-GT alignment failed: "
                f"expected={expected_visual_gt}, visual_gt={gt}"
            )

        tick_row = tick_rows[project_index]
        if tick_row["pitch"] != cache_score["p"]:
            raise ValueError(f"seed slot {slot}: merged-tick/project pitch ordering differs")
        candidates = [
            index
            for index, row in enumerate(xml_identity)
            if row["pitch"] == cache_score["p"]
            and abs(row["onset_q"] * ticks_per_quarter - tick_row["onset_tick"]) <= 0.5
            # ASAP's MIDI export can shorten an XML duration by exactly one tick.
            and abs(
                row["duration_q"] * ticks_per_quarter - tick_row["duration_tick"]
            )
            <= 1.1
        ]
        if len(candidates) != 1:
            nearest = sorted(
                (
                    {
                        "row": index,
                        "onset_tick": row["onset_q"] * ticks_per_quarter,
                        "duration_tick": row["duration_q"] * ticks_per_quarter,
                        "onset_error": abs(
                            row["onset_q"] * ticks_per_quarter - tick_row["onset_tick"]
                        ),
                        "duration_error": abs(
                            row["duration_q"] * ticks_per_quarter
                            - tick_row["duration_tick"]
                        ),
                    }
                    for index, row in enumerate(xml_identity)
                    if row["pitch"] == cache_score["p"]
                ),
                key=lambda row: (row["onset_error"], row["duration_error"]),
            )[:4]
            raise ValueError(
                f"seed slot {slot}: absolute onset/pitch/duration XML mapping is not unique; "
                f"project score row {project_index}, paper score row {score_index}, "
                f"MIDI ticks={tick_row}, candidates={candidates[:8]}, "
                f"nearest same-pitch XML rows={nearest}"
            )
        xml_row = candidates[0]
        xml_row_identity = xml_identity[xml_row]
        xml_absolute_q = xml_row_identity["onset_q"]
        original_offset_q = float(xml_streams["offset"][xml_row])
        original_duration_q = float(xml_streams["duration"][xml_row])
        measure_length_q = xml_row_identity["measure_length_q"]
        if (
            original_offset_q < -1e-6
            or original_offset_q > measure_length_q + 1e-6
            or original_offset_q > 6.0
            or original_duration_q < 0.0
            or original_duration_q > 4.0
            or not 0 <= xml_row_identity["pitch"] <= 127
        ):
            raise ValueError(
                f"seed slot {slot}: aligned original MusicXML GT is outside the "
                f"paper tokenizer range: offset={original_offset_q}, "
                f"measure={measure_length_q}, duration={original_duration_q}, "
                f"pitch={xml_row_identity['pitch']}"
            )
        onset_error = abs(
            xml_absolute_q * ticks_per_quarter - tick_row["onset_tick"]
        )
        duration_error = abs(
            xml_row_identity["duration_q"] * ticks_per_quarter
            - tick_row["duration_tick"]
        )
        xml_rows.append(xml_row)
        score_indices.append(score_index)
        project_score_indices.append(project_index)
        mapping_certifications.append(
            {
                "cache_builder_triplet_exact": True,
                "cache_window_gt_triplet_exact": True,
                "raw_score_index_within_10ms_token_rounding": True,
                "absolute_tick_onset_pitch_duration_unique": True,
                "project_score_order_uniquely_mapped": True,
                "paper_score_order_uniquely_mapped": True,
                "musicxml_second_parse_row_identity_exact": True,
                "native_xml_core_unmodified": True,
                "score_onset_tick": tick_row["onset_tick"],
                "score_duration_tick": tick_row["duration_tick"],
                "ticks_per_quarter": ticks_per_quarter,
                "xml_onset_tick": xml_absolute_q * ticks_per_quarter,
                "xml_duration_tick": (
                    xml_row_identity["duration_q"] * ticks_per_quarter
                ),
                "onset_tick_abs_error": onset_error,
                "duration_tick_abs_error": duration_error,
                "onset_tick_tolerance": 0.5,
                "duration_tick_tolerance": 1.1,
                "xml_original_offset_quarters": original_offset_q,
                "xml_original_duration_quarters": original_duration_q,
                "mapped_measure_length_quarters": measure_length_q,
            }
        )

    # Each visual seed slot must own a distinct row at every stage.  Otherwise
    # a duplicated native target would make the visual prefix ambiguous.
    for label, indices in (
        ("project score", project_score_indices),
        ("paper score", score_indices),
        ("MusicXML", xml_rows),
    ):
        if len(set(indices)) != len(indices):
            raise ValueError(f"multiple visual seed slots mapped to one {label} row")

    xml_tokenized = tokenizer_mod.MultistreamTokenizer.bucket_mxl(xml_streams)
    gt_tokens = prepare_full_gt(kind, xml_tokenized, device)
    native_decoded = decode_contiguous(kind, gt_tokens)

    # Validate the original MusicXML target after the paper's own native
    # bucketing.  The displayed beat-normalized triplet is certified separately
    # through the cache mapping above; it is intentionally not inverted through
    # a whole-piece quarters-per-beat approximation.
    half_native_bin_q = 1.0 / 48.0 + 1e-5
    for slot, xml_row in enumerate(xml_rows):
        note = native_decoded[xml_row]
        identity = xml_identity[xml_row]
        onset_q_error = abs(note["onset_q"] - identity["onset_q"])
        duration_q_error = abs(note["dur_q"] - identity["duration_q"])
        mapping_certifications[slot].update(
            {
                "native_xml_bucket_roundtrip": True,
                "native_xml_onset_quarter_error": onset_q_error,
                "native_xml_duration_quarter_error": duration_q_error,
                "native_xml_half_bin_quarter_tolerance": half_native_bin_q,
            }
        )
        if (
            not pad_is_valid(kind, gt_tokens, xml_row)
            or onset_q_error > half_native_bin_q
            or duration_q_error > half_native_bin_q
            or int(note["pitch"]) != identity["pitch"]
        ):
            raise ValueError(
                f"seed slot {slot}: original XML row does not survive native bucketing: "
                f"decoded={note}, identity={identity}"
            )
    return (
        gt_tokens,
        xml_rows,
        score_indices,
        project_score_indices,
        mapping_certifications,
    )


def pad_is_valid(kind: str, output: dict[str, torch.Tensor], position: int) -> bool:
    return bool(float(pad_value_at(kind, output, position)) > 0.5)


def decode_contiguous(kind: str, output: dict[str, torch.Tensor]):
    n_notes = output["pitch"].shape[1]
    if kind == "paper2":
        cpu = {key: value[0].detach().cpu() for key, value in output.items()}
    else:
        cpu = {key: value.detach().cpu() for key, value in output.items()}
    return decode_score_streams(cpu, n_notes, kind)


@torch.inference_mode()
def branch_variant(
    kind,
    model,
    full_input,
    canonical_carry,
    target_model,
    xml_rows,
    gt_tokens,
    count,
    qpb,
    ex,
    device,
):
    spec = MODEL_SPECS[kind]
    starts = chunk_starts(full_input["pitch"].shape[1], spec["chunk"], spec["overlap"])
    owner_indices = [
        owner_chunk_index(position, starts, spec["chunk"], spec["overlap"])
        for position in target_model
    ]
    first_chunk, last_chunk = min(owner_indices), max(owner_indices)
    carry = clone_output(canonical_carry)
    captured = None
    capture_lo, capture_hi = min(target_model), max(target_model) + 1

    forced = dict(zip(target_model[:count], xml_rows[:count]))
    if len(forced) != count:
        raise ValueError("multiple visual seed notes mapped to one paper input row")
    intervention_certifications = []
    for chunk_index in range(first_chunk, last_chunk + 1):
        start = starts[chunk_index]
        x_chunk = {key: value[:, start : start + spec["chunk"]] for key, value in full_input.items()}
        owned_lo = start if chunk_index == 0 else start + spec["overlap"]
        owned_hi = start + spec["chunk"]
        force_rows = sorted(
            (position - start, xml_row)
            for position, xml_row in forced.items()
            if owned_lo <= position < owned_hi
        )
        output, chunk_certifications = generate_chunk(
            kind,
            model,
            x_chunk,
            carry,
            start,
            spec["chunk"],
            force_rows,
            gt_tokens,
            device,
        )
        intervention_certifications.extend(chunk_certifications)
        take_lo = max(capture_lo, owned_lo)
        take_hi = min(capture_hi, owned_hi)
        if take_lo < take_hi:
            captured = concat_output(captured, slice_output(output, take_lo - start, take_hi - start))
        carry = slice_output(output, -spec["overlap"], None)

    expected_length = capture_hi - capture_lo
    if captured is None or captured["pitch"].shape[1] != expected_length:
        raise ValueError("official chunk replay did not capture the complete visualizer window")
    decoded = decode_contiguous(kind, captured)
    selected = [position - capture_lo for position in target_model]
    if not pad_is_valid(kind, captured, selected[0]):
        raise ValueError("first forced output row is padded")
    anchor_q = decoded[selected[0]]["onset_q"]
    scale = 50.0 / qpb
    pred = []
    certified = []
    padded_rows = 0
    for slot, position in enumerate(selected):
        valid = pad_is_valid(kind, captured, position)
        note = decoded[position]
        triplet = {
            "t": int(round((note["onset_q"] - anchor_q) * scale)),
            "d": max(1, int(round(note["dur_q"] * scale))),
            "p": int(note["pitch"]),
        }
        if slot < count:
            gt = ex["gt_score"][slot]
            xml_row = xml_rows[slot]
            mismatched_streams = [
                key
                for key in ("offset", "downbeat", "duration", "pitch")
                if not torch.equal(
                    captured[key][0, position], gt_tokens[key][0, xml_row]
                )
            ]
            if float(pad_value_at(kind, captured, position)) != float(
                pad_value_at(kind, gt_tokens, xml_row)
            ):
                mismatched_streams.append("pad")
            if not valid or mismatched_streams:
                raise ValueError(
                    f"forced slot {slot} differs from its original native GT token row: "
                    f"streams={mismatched_streams}, valid={valid}"
                )
            # Conditioning is certified in original paper-native tokens above.
            # Drawing uses the exactly cache-certified, beat-normalized visual GT
            # so native and common display grids are never conflated.
            triplet = dict(gt)
            triplet["seeded"] = True
            certified.append(slot)
        if valid and 0 <= triplet["p"] <= 127:
            pred.append(triplet)
        elif not valid:
            padded_rows += 1
        else:
            raise ValueError(f"{kind}: valid seeded branch row has invalid pitch {triplet['p']}")

    gt_span = max((note["t"] for note in ex["gt_score"] if note), default=0) or 1
    pred_span = max((note["t"] for note in pred), default=0) or 1
    return {
        "source": MODEL_SPECS[kind]["identity"],
        "input_protocol": "official full-piece chunk replay on original unfiltered performance MIDI",
        "conditioning": SEEDED_CONDITIONING,
        "seed_count": count,
        "pred_score": pred,
        "matched_notes": len(pred),
        "padded_notes": padded_rows,
        "span_ratio": round(pred_span / gt_span, 4),
        "span_ok": bool(1 / 1.35 <= pred_span / gt_span <= 1.35),
        "seed_certification": {
            "complete_notation_rows": False,
            "common_triplet_only": True,
            "explicit_pad_mask_applied": True,
            "native_forced_token_rows_exact": True,
            "display_seed_triplets_exact_gt": True,
            "cache_to_visual_gt_exact": True,
            "native_target_representation": (
                "original uniquely aligned MusicXML token row"
            ),
            "certified_slots": certified,
            "forced_streams": ["downbeat", "duration", "offset", "pad", "pitch"],
            "intervention_rows": intervention_certifications,
        },
    }


def replay_unforced_window(kind, model, full_input, canonical_carry, target_model, device):
    """Replay the branched chunk path with zero forces for parity certification."""
    spec = MODEL_SPECS[kind]
    starts = chunk_starts(full_input["pitch"].shape[1], spec["chunk"], spec["overlap"])
    owner_indices = [
        owner_chunk_index(position, starts, spec["chunk"], spec["overlap"])
        for position in target_model
    ]
    first_chunk, last_chunk = min(owner_indices), max(owner_indices)
    capture_lo, capture_hi = min(target_model), max(target_model) + 1
    carry = clone_output(canonical_carry)
    captured = None
    for chunk_index in range(first_chunk, last_chunk + 1):
        start = starts[chunk_index]
        x_chunk = {key: value[:, start : start + spec["chunk"]] for key, value in full_input.items()}
        output, certs = generate_chunk(
            kind, model, x_chunk, carry, start, spec["chunk"], [], None, device
        )
        if certs:
            raise ValueError("zero-force replay unexpectedly produced intervention records")
        owned_lo = start if chunk_index == 0 else start + spec["overlap"]
        owned_hi = start + spec["chunk"]
        take_lo = max(capture_lo, owned_lo)
        take_hi = min(capture_hi, owned_hi)
        if take_lo < take_hi:
            captured = concat_output(captured, slice_output(output, take_lo - start, take_hi - start))
        carry = slice_output(output, -spec["overlap"], None)
    return captured


def validate_legacy_baseline(kind: str, legacy: dict, corrected_pred: list[dict]) -> dict:
    """Certify corrected explicit-pad output against the already-published baseline."""
    if not isinstance(legacy, dict) or not isinstance(legacy.get("pred_score"), list):
        raise ValueError(f"{kind}: legacy filtered baseline is missing")
    legacy_pred = legacy["pred_score"]
    # The old Paper 2 decoder argmaxed an all-zero padded one-hot row, creating
    # exactly pitch=0,duration=1.  Paper 1's index shift made those pitch=-1 and
    # the old path already dropped them.  This known artifact is the only row we
    # remove before demanding parity.
    cleaned = [
        note
        for note in legacy_pred
        if not (kind == "paper2" and note.get("p") == 0 and note.get("d") == 1)
    ]
    removed = len(legacy_pred) - len(cleaned)
    if len(cleaned) != len(corrected_pred):
        raise ValueError(
            f"{kind}: corrected/legacy-clean baseline lengths differ: "
            f"{len(corrected_pred)} vs {len(cleaned)} (removed {removed})"
        )
    max_t_error = 0
    max_d_error = 0
    for index, (old, new) in enumerate(zip(cleaned, corrected_pred)):
        if old.get("p") != new.get("p"):
            raise ValueError(f"{kind}: corrected baseline pitch differs at row {index}")
        t_error = abs(int(old["t"]) - int(new["t"]))
        d_error = abs(int(old["d"]) - int(new["d"]))
        max_t_error = max(max_t_error, t_error)
        max_d_error = max(max_d_error, d_error)
        if t_error > 1 or d_error > 1:
            raise ValueError(
                f"{kind}: corrected baseline differs beyond one bin at row {index}: "
                f"legacy={old}, corrected={new}"
            )
    return {
        "legacy_rows": len(legacy_pred),
        "legacy_known_pad_artifacts_removed": removed,
        "legacy_clean_rows": len(cleaned),
        "pitch_order_exact": True,
        "max_onset_bin_error": max_t_error,
        "max_duration_bin_error": max_d_error,
        "within_one_quantization_bin": True,
    }


def corrected_baseline(
    kind: str,
    reference: dict[str, torch.Tensor],
    target_model: list[int],
    qpb: float,
    ex: dict,
) -> dict:
    """Decode canonical official inference using its explicit pad stream."""
    capture_lo, capture_hi = min(target_model), max(target_model) + 1
    captured = slice_output(reference, capture_lo, capture_hi)
    decoded = decode_contiguous(kind, captured)
    selected = [position - capture_lo for position in target_model]
    valid_rows = [position for position in selected if pad_is_valid(kind, captured, position)]
    if not valid_rows:
        raise ValueError(f"{kind}: canonical baseline pads every visualizer row")
    base_q = min(decoded[position]["onset_q"] for position in valid_rows)
    scale = 50.0 / qpb
    pred = []
    quarters = []
    for position in selected:
        if not pad_is_valid(kind, captured, position):
            continue
        note = decoded[position]
        triplet = {
            "t": int(round((note["onset_q"] - base_q) * scale)),
            "d": max(1, int(round(note["dur_q"] * scale))),
            "p": int(note["pitch"]),
        }
        if not 0 <= triplet["p"] <= 127:
            raise ValueError(f"{kind}: valid output row decoded invalid pitch {triplet['p']}")
        pred.append(triplet)
        quarters.append(
            {
                "on": round(note["onset_q"] - base_q, 6),
                "dur": round(note["dur_q"], 6),
                "p": triplet["p"],
            }
        )
    legacy = (ex.get(f"rollouts_{kind}") or {}).get("filtered")
    comparison = validate_legacy_baseline(kind, legacy, pred)
    padded = len(selected) - len(valid_rows)
    if kind == "paper2" and comparison["legacy_known_pad_artifacts_removed"] != padded:
        raise ValueError(
            "Paper 2 explicit pad count does not equal legacy fake-row count: "
            f"{padded} vs {comparison['legacy_known_pad_artifacts_removed']}"
        )
    gt_span = max((note["t"] for note in ex["gt_score"] if note), default=0) or 1
    pred_span = max((note["t"] for note in pred), default=0) or 1
    return {
        "source": MODEL_SPECS[kind]["identity"],
        "input_protocol": "official full-piece chunk replay on original unfiltered performance MIDI",
        "conditioning": "unseeded canonical official infer",
        "seed_count": 0,
        "quarters_per_beat": qpb,
        "pred_score": pred,
        "pred_quarters": quarters,
        "matched_notes": len(pred),
        "padded_notes": padded,
        "span_ratio": round(pred_span / gt_span, 4),
        "span_ok": bool(1 / 1.35 <= pred_span / gt_span <= 1.35),
        "baseline_certification": {
            "canonical_official_infer": True,
            "explicit_pad_mask_applied": True,
            "selected_input_rows": len(selected),
            "valid_rows": len(valid_rows),
            "padded_rows": padded,
            "legacy_comparison": comparison,
        },
    }


def compute_window(kind, model, tokenizer_mod, utils_mod, piece, start_filtered, ex, qpb, device):
    n_gt = len(ex.get("gt_score") or [])
    if n_gt < 5:
        raise ValueError("visualizer window has fewer than five GT notes")
    target_project_perf = piece["filtered_to_raw"][start_filtered : start_filtered + n_gt]
    if len(target_project_perf) != n_gt:
        raise ValueError("aligned stream ends before the visualizer score window")
    target_model, input_mapping_certifications = map_raw_input_positions(
        kind, tokenizer_mod, piece, target_project_perf
    )

    midi_tokens = tokenizer_mod.MultistreamTokenizer.tokenize_midi(piece["piece_id"])
    full_input = prepare_full_input(kind, midi_tokens, device)
    (
        gt_tokens,
        xml_rows,
        score_indices,
        project_score_indices,
        mapping_certifications,
    ) = prepare_gt_rows(
        kind, tokenizer_mod, piece, target_project_perf, ex, device
    )
    spec = MODEL_SPECS[kind]
    starts = chunk_starts(full_input["pitch"].shape[1], spec["chunk"], spec["overlap"])
    owner_indices = [
        owner_chunk_index(position, starts, spec["chunk"], spec["overlap"])
        for position in target_model
    ]
    first_chunk = min(owner_indices)
    # Use the paper's own infer() as the canonical unseeded reference and exact
    # carried overlap.  A separate zero-force replay below must be token-identical
    # before any intervention is accepted.
    # Paper 1 must receive the +1 shifted, batched index streams; Paper 2 must
    # receive its native unbatched one-hot streams.  These exactly mirror the
    # existing official-baseline path in run_paper_models.transcribe_piece.
    reference_input = (
        full_input
        if kind == "paper1"
        else {key: value[0] for key, value in full_input.items()}
    )
    reference = utils_mod.infer(
        reference_input,
        model,
        overlap=spec["overlap"],
        chunk=spec["chunk"],
        verbose=False,
        kv_cache=True,
        **({"device": device.type} if kind == "paper1" else {}),
    )
    reference = reference_to_batched(kind, reference)
    if reference["pitch"].shape[1] <= max(target_model):
        raise ValueError("official reference ends before the visualizer window")
    branch_start = starts[first_chunk]
    canonical_carry = (
        None
        if first_chunk == 0
        else {
            key: value[:, branch_start : branch_start + spec["overlap"]].to(device)
            for key, value in reference.items()
        }
    )
    zero_force = replay_unforced_window(
        kind, model, full_input, canonical_carry, target_model, device
    )
    capture_lo, capture_hi = min(target_model), max(target_model) + 1
    reference_window = {
        key: value[:, capture_lo:capture_hi].to(device)
        for key, value in reference.items()
    }
    parity = all(torch.equal(zero_force[key], reference_window[key]) for key in zero_force)
    if not parity:
        differing = [key for key in zero_force if not torch.equal(zero_force[key], reference_window[key])]
        raise ValueError(f"zero-force replay differs from official infer for streams {differing}")

    variants = {"filtered": corrected_baseline(kind, reference, target_model, qpb, ex)}
    for count in SEED_COUNTS:
        print(f"      seed {count}: official replay + GT triplet intervention", flush=True)
        variants[variant_name(count)] = branch_variant(
            kind,
            model,
            full_input,
            canonical_carry,
            target_model,
            xml_rows,
            gt_tokens,
            count,
            qpb,
            ex,
            device,
        )
        variants[variant_name(count)]["seed_certification"].update(
            {
                "project_performance_indices": target_project_perf[:count],
                "paper_input_positions": target_model[:count],
                "input_mapping_certifications": input_mapping_certifications[:count],
                "score_midi_indices": score_indices[:count],
                "project_score_midi_indices": project_score_indices[:count],
                "musicxml_token_rows": xml_rows[:count],
                "mapping_certifications": mapping_certifications[:count],
                "zero_force_matches_official_infer": True,
            }
        )
    del full_input, gt_tokens, canonical_carry, midi_tokens, reference, zero_force, reference_window
    return variants


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="visualizer/data.js")
    parser.add_argument("--output", required=True)
    parser.add_argument("--cache-dir", default="data/asap_aligned_stream_cache")
    parser.add_argument("--paper1-ckpt", default=MODEL_SPECS["paper1"]["checkpoint"])
    parser.add_argument("--paper2-ckpt", default=MODEL_SPECS["paper2"]["checkpoint"])
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--device", default=None)
    parser.add_argument("--only", choices=("paper1", "paper2"), default=None)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Diagnostic only: process the first N windows assigned to this shard.",
    )
    args = parser.parse_args()
    if args.num_shards != 4 or not 0 <= args.shard_index < 4:
        raise SystemExit("paper seed precompute requires four shards and index 0..3")

    data_path = Path(args.data).resolve()
    payload, _prefix = load_payload(data_path)
    examples = payload.get("examples") or {}
    full_order = ordered_example_ids(payload)
    order = full_order[args.shard_index :: 4]
    if len(order) != 6:
        raise SystemExit(f"expected six windows per shard, got {len(order)}")
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        order = order[: args.limit]
    diagnostic_only = args.only is not None or args.limit is not None
    pieces = load_cache_pieces((REPO / args.cache_dir).resolve())
    located = {}
    for eid in order:
        ex = examples[eid]
        controls = [(note["t"], note["d"], note["p"]) for note in ex["perf_notes"]]
        piece, start = locate_window(pieces, controls)
        if piece is None:
            raise SystemExit(f"{eid}: cannot locate window in aligned-stream cache")
        expected = str(ex.get("piece") or "")
        if expected and not str(piece["piece_id"]).endswith(expected):
            raise SystemExit(f"{eid}: located {piece['piece_id']}, expected {expected}")
        qpb = quarters_per_annotated_beat(expected)
        if qpb is None or not math.isfinite(qpb) or qpb <= 0:
            raise SystemExit(f"{eid}: invalid quarters-per-annotated-beat")
        located[eid] = piece, start, qpb

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type != "cuda":
        raise SystemExit("paper seed precompute is GPU-only")
    kinds = [args.only] if args.only else ["paper1", "paper2"]
    checkpoints = {
        kind: (REPO / (args.paper1_ckpt if kind == "paper1" else args.paper2_ckpt)).resolve()
        for kind in kinds
    }
    identities = {kind: file_identity(path) for kind, path in checkpoints.items()}
    data_identity = source_data_identity(payload)

    shard = {
        "format": 3,
        "shard_index": args.shard_index,
        "num_shards": 4,
        "diagnostic_only": diagnostic_only,
        "seed_counts": list(SEED_COUNTS),
        "input_protocol": "official full-piece chunk replay on original unfiltered performance MIDI",
        "conditioning": SEEDED_CONDITIONING,
        "source_data_identity": data_identity,
        "example_order": order,
        "models": {
            kind: {
                "identity": MODEL_SPECS[kind]["identity"],
                "checkpoint": identities[kind],
                "chunk": MODEL_SPECS[kind]["chunk"],
                "overlap": MODEL_SPECS[kind]["overlap"],
            }
            for kind in kinds
        },
        "examples": {eid: {} for eid in order},
    }

    torch.manual_seed(3407)
    for kind in kinds:
        started = time.perf_counter()
        print(f"\n=== {kind}: {checkpoints[kind]} ===", flush=True)
        model, tokenizer_mod, utils_mod = load_external_model(kind, checkpoints[kind], device)
        for eid in order:
            piece, start, qpb = located[eid]
            print(f"  {eid}: qpb={qpb:g}", flush=True)
            variants = compute_window(
                kind, model, tokenizer_mod, utils_mod, piece, start, examples[eid], qpb, device
            )
            shard["examples"][eid][f"rollouts_{kind}"] = variants
        print(f"{kind} finished in {time.perf_counter() - started:.1f}s", flush=True)
        del model, tokenizer_mod, utils_mod
        gc.collect()
        torch.cuda.empty_cache()

    atomic_dump_json(Path(args.output), shard)
    print(f"Wrote {args.output}", flush=True)


if __name__ == "__main__":
    main()
