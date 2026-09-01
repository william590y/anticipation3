"""Collate the JSON files under bench/results/ into the report tables.

Kept as a script rather than hand-copied numbers so the tables in any write-up
can be regenerated from the raw measurements after a re-run.

    python bench/collect_results.py
"""

from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"


def load(name):
    path = RESULTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text())


def throughput_table(name, title):
    data = load(name)
    if not data:
        print(f"\n[{title}] missing: {name}")
        return
    gpu = data["info"].get("gpu", "?")
    dtype = data["info"].get("dtype", "?")
    print(f"\n=== {title} ({gpu}, {dtype}, attn={data['info'].get('attn_implementation')}) ===")
    baselines = {
        row["batch"]: row["best_s"] for row in data["results"] if row["variant"] == "baseline"
    }
    print(f"{'variant':<34}{'batch':>6}{'s/rollout':>11}{'win/s':>9}{'ms/step':>9}{'peak GiB':>10}{'vs base':>9}")
    for row in data["results"]:
        base = baselines.get(row["batch"])
        speedup = f"{base / row['best_s']:.2f}x" if base else "-"
        print(
            f"{row['variant']:<34}{row['batch']:>6}{row['best_s']:>11.3f}"
            f"{row['windows_per_s']:>9.2f}{row['ms_per_step']:>9.2f}"
            f"{row['peak_mem_gib']:>10.2f}{speedup:>9}"
        )


def gate_table(name, title):
    data = load(name)
    if not data:
        print(f"\n[{title}] missing: {name}")
        return
    print(f"\n=== {title} ({data['windows']} windows, candidate dtype {data['info']['dtype']}) ===")
    for spec, entry in data["variants"].items():
        verdict = entry.get("bit_identical")
        mark = "BIT-IDENTICAL" if verdict else ("DIFFERS" if verdict is False else "parity-only")
        agreement = entry.get("token_agreement")
        line = f"{spec:<34}{mark:<16}"
        if agreement is not None:
            line += f"tokens {agreement:.6f}  windows {entry.get('identical_windows')}"
        print(line)
        if "gt_ce_max_abs_delta" in entry:
            print(f"{'':<34}gt_ce max |delta| = {entry['gt_ce_max_abs_delta']:.3e} "
                  f"(bit-identical: {entry['gt_ce_bit_identical']})")
        ref, cand = entry["parity"]["reference"], entry["parity"]["candidate"]
        print(
            f"{'':<34}ref  onset {ref['onset_acc']:.4f} dur {ref['duration_acc']:.4f} "
            f"pitch {ref['pitch_acc']:.4f} F1 {ref['f1_onset_pitch_tol1']:.4f}"
        )
        print(
            f"{'':<34}cand onset {cand['onset_acc']:.4f} dur {cand['duration_acc']:.4f} "
            f"pitch {cand['pitch_acc']:.4f} F1 {cand['f1_onset_pitch_tol1']:.4f}"
        )


def probe_table(name, title):
    data = load(name)
    if not data:
        print(f"\n[{title}] missing: {name}")
        return
    print(f"\n=== {title} ({data['info'].get('gpu')}) ===")
    print(f"{'batch':>6}{'cache':>7}{'ms/step':>10}{'KV GiB':>9}{'implied GiB/s':>15}")
    for row in data["probe"]:
        if row.get("oom"):
            print(f"{row['batch']:>6}{row['cache_len']:>7}{'OOM':>10}")
            continue
        print(
            f"{row['batch']:>6}{row['cache_len']:>7}{row['ms']:>10.2f}"
            f"{row['kv_gib']:>9.2f}{row['implied_gib_s']:>15.1f}"
        )


def tensorrt_summary(name):
    data = load(name)
    if not data:
        print(f"\n[TensorRT] missing: {name}")
        return
    print("\n=== TensorRT ===")
    print(json.dumps(data["versions"], indent=1))
    for stage in ("prefill", "decode_step"):
        for entry in data.get(stage, []):
            if "error" in entry:
                print(f"{stage} batch {entry['batch']}: FAILED -- {entry['error'].splitlines()[1]}")
            else:
                print(
                    f"{stage} batch {entry['batch']}: eager {entry['eager_ms']:.1f} ms -> "
                    f"TRT {entry['trt_ms']:.1f} ms ({entry['speedup']:.2f}x), "
                    f"build {entry['build_s']:.1f} s, "
                    f"max|delta| {entry['max_abs_delta']:.4g}, "
                    f"bit-identical {entry['bit_identical']}"
                )


def main():
    probe_table("probe_3090.json", "Forward-cost probe (DynamicCache, fp32)")
    throughput_table("baseline_ada.json", "BASELINE fp32")
    throughput_table("ablation_3090_v2.json", "Ablation fp32, RTX 3090")
    throughput_table("ablation_ada_v2.json", "Ablation fp32, RTX 6000 Ada")
    throughput_table("ablation_buckets_fp32_3090.json", "Bucketed KV, fp32, RTX 3090")
    throughput_table("ablation_buckets_fp32_ada.json", "Bucketed KV, fp32, RTX 6000 Ada")
    throughput_table("ablation_bf16_ada.json", "Ablation bf16, RTX 6000 Ada")
    throughput_table("ablation_fp16_ada.json", "Ablation fp16, RTX 6000 Ada")
    throughput_table("ablation_buckets_fp16_ada.json", "Bucketed KV, fp16, RTX 6000 Ada")
    gate_table("gate_greedy_208.json", "GATE greedy fp32")
    gate_table("gate_gtce_96_v2.json", "GATE greedy fp32 + gt_ce")
    gate_table("gate_buckets_64.json", "GATE bucketed KV cache")
    gate_table("gate_bf16_208.json", "GATE bfloat16")
    gate_table("gate_bf16_208_v2.json", "GATE bfloat16 (repeat)")
    gate_table("gate_fp16_208.json", "GATE float16")
    gate_table("gate_eager_32.json", "GATE eager attention")
    tensorrt_summary("tensorrt2.json")


if __name__ == "__main__":
    main()
