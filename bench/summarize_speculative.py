"""Render `bench/results_speculative.json` as the report tables.

Kept separate from the benchmark so the (long) GPU job never has to be re-run to
change how a number is presented.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nbest.speculative import crossover_cost_ratio, predicted_speedup


def _fmt(value, spec=".3f"):
    if value is None:
        return "-"
    try:
        if value != value:  # NaN
            return "-"
        return format(value, spec)
    except (TypeError, ValueError):
        return str(value)


def micro_table(results):
    micro = results.get("micro")
    if not micro:
        return
    print("\n## Per-forward latency (ms), 1-token decode step on a ~600-token cache")
    batches = sorted({int(b) for rows in micro.values() for b in rows})
    header = f"{'model':24s} {'layers':>6s} {'width':>6s}" + "".join(
        f"{'b=' + str(b):>10s}" for b in batches
    )
    print(header)
    print("-" * len(header))
    for name, rows in micro.items():
        first = next(iter(rows.values()))
        line = f"{name:24s} {first['n_layer']:6d} {first['n_embd']:6d}"
        for b in batches:
            row = rows.get(str(b), rows.get(b))
            line += f"{_fmt(row['ms_1tok'], '10.3f') if row else '-':>10s}"
        print(line)

    ratios = results.get("cost_ratio_vs_target", {})
    print("\n## Cost ratio c_level / c_target (same batch), from the same measurement")
    header = f"{'level':24s}" + "".join(f"{'b=' + str(b):>10s}" for b in batches)
    print(header)
    names = sorted({n for r in ratios.values() for n in r})
    for name in names:
        line = f"{name:24s}"
        for b in batches:
            row = ratios.get(str(b), ratios.get(b, {}))
            line += f"{_fmt(row.get(name), '10.3f'):>10s}"
        print(line)


def accept_table(results):
    rows = results.get("accept")
    if not rows:
        return
    print("\n## Acceptance and per-forward efficiency (batch 1 = per-sequence, no shared-cache penalty)")
    header = (
        f"{'config':14s}{'T':>5s}{'gamma':>6s}{'onset':>8s}{'dur':>8s}{'pitch':>8s}"
        f"{'all':>8s}{'tok/Tfwd':>10s}{'Tfwd/win':>10s}{'neuralfwd/win':>14s}{'lookups/win':>12s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        print(
            f"{row['config']:14s}{_fmt(row['temperature'], '5.1f'):>5s}"
            f"{row['slots_per_block']:6d}"
            f"{_fmt(row.get('acceptance_target_onset')):>8s}"
            f"{_fmt(row.get('acceptance_target_duration')):>8s}"
            f"{_fmt(row.get('acceptance_target_pitch')):>8s}"
            f"{_fmt(row.get('acceptance_target')):>8s}"
            f"{_fmt(row.get('tokens_per_target_forward')):>10s}"
            f"{_fmt(row.get('forwards_per_window_target'), '10.1f'):>10s}"
            f"{_fmt(row.get('forwards_per_window_draft') or row.get('forwards_per_window_d1'), '14.1f'):>14s}"
            f"{_fmt(row.get('forwards_per_window_ngram') or row.get('forwards_per_window_d2'), '12.1f'):>12s}"
        )


def speed_table(results):
    rows = results.get("speed")
    if not rows:
        return
    print("\n## End-to-end throughput (windows/sec) vs the unmodified rollout_score_slots")
    for temperature in sorted({r["temperature"] for r in rows}):
        print(f"\n  temperature = {temperature}")
        subset = [r for r in rows if r["temperature"] == temperature]
        batches = sorted({r["batch"] for r in subset})
        base = {
            r["batch"]: r["windows_per_sec"] for r in subset if r["config"] == "baseline"
        }
        keys = []
        for r in subset:
            key = (r["config"], r["slots_per_block"])
            if r["config"] != "baseline" and key not in keys:
                keys.append(key)
        header = f"{'config':16s}{'gamma':>6s}" + "".join(
            f"{'b=' + str(b):>18s}" for b in batches
        )
        print(header)
        print("-" * len(header))
        line = f"{'baseline':16s}{'-':>6s}"
        for b in batches:
            line += f"{_fmt(base.get(b), '11.3f') + ' (1.00x)':>18s}"
        print(line)
        for config, slots in keys:
            line = f"{config:16s}{slots:6d}"
            for b in batches:
                match = [
                    r for r in subset
                    if r["config"] == config and r["slots_per_block"] == slots and r["batch"] == b
                ]
                if not match or b not in base:
                    line += f"{'-':>18s}"
                    continue
                wps = match[0]["windows_per_sec"]
                line += f"{_fmt(wps, '11.3f') + f' ({wps / base[b]:.2f}x)':>18s}"
            print(line)


def exact_table(results):
    exact = results.get("exact")
    if not exact:
        return
    print("\n## Exactness")
    control = exact.get("control_baseline_batch_reshape", {})
    print(
        f"  control (baseline vs baseline, batch {control.get('batch_a')} vs "
        f"{control.get('batch_b')}): "
        f"{_fmt(control.get('identical_windows'), '.4f')} windows, "
        f"{_fmt(control.get('identical_tokens'), '.4f')} tokens identical"
    )
    noise = exact.get("numerical_noise_floor", {})
    print(
        f"  numerical noise floor: argmax flips between the incremental and the "
        f"single-forward path at "
        f"{_fmt(noise.get('argmax_flip_rate_incremental_vs_single_forward'), '.5f')} of "
        f"{noise.get('positions')} positions; median top1-top2 logit gap "
        f"{_fmt(noise.get('median_top2_gap'), '.3f')}"
    )
    print(f"\n  greedy (T=0) vs baseline:")
    print(f"    {'config':16s}{'windows identical':>20s}{'tokens identical':>20s}")
    for name, row in exact.get("greedy", {}).items():
        print(
            f"    {name:16s}{_fmt(row['identical_windows'], '20.4f'):>20s}"
            f"{_fmt(row['identical_tokens'], '20.4f'):>20s}"
        )


def dist_table(results):
    dist = results.get("distributional")
    if not dist:
        return
    print(
        f"\n## Distributional exactness at T=1 "
        f"({dist['windows']} windows x {dist['replicates']} independent rollouts per sampler)"
    )
    print(f"    {'config':16s}{'tests':>7s}{'min p':>9s}{'frac p<0.05':>13s}{'uniformity KS p':>17s}")
    for name, row in dist.get("pooled", {}).items():
        print(
            f"    {name:16s}{row['n_tests']:7d}{_fmt(row['min_p'], '9.4f'):>9s}"
            f"{_fmt(row['frac_below_0.05'], '13.3f'):>13s}"
            f"{_fmt(row['uniformity_ks_p'], '17.4f'):>17s}"
        )
    print("\n  per-window mean role accuracy, baseline vs speculative:")
    for entry in dist.get("per_window", []):
        for name, record in entry["configs"].items():
            parts = []
            for role in ("onset", "duration", "pitch"):
                item = record["accuracy_ks"][role]
                parts.append(
                    f"{role[:3]} {item['baseline_mean']:.4f}/{item['spec_mean']:.4f} "
                    f"(p={item['p']:.3f})"
                )
            print(f"    window {entry['window']} {name:16s} " + "  ".join(parts))


def cost_model(results):
    rows = results.get("accept")
    ratios = results.get("cost_ratio_vs_target", {})
    if not rows or not ratios:
        return
    print("\n## Cost model: speedup predicted from per-forward accounting alone")
    print("   (recompute with cost_ratio * s if the target step is later made s times cheaper)")
    header = (
        f"{'config':14s}{'T':>5s}{'gamma':>6s}{'tok/Tfwd':>10s}"
        f"{'pred b=1':>10s}{'pred b=16':>11s}{'breakeven c_draft/c_target':>28s}"
    )
    print(header)
    print("-" * len(header))
    for row in rows:
        levels = {
            k[len("forwards_per_target_forward_") :]: v
            for k, v in row.items()
            if k.startswith("forwards_per_target_forward_") and k.split("_")[-1] != "target"
        }
        levels.pop("target", None)
        tok = row.get("tokens_per_target_forward")
        preds = []
        for batch in (1, 16):
            table = ratios.get(str(batch), ratios.get(batch))
            if not table:
                preds.append(None)
                continue
            # Which measured per-forward cost belongs to this config's levels.
            config = row["config"]
            draft_name = config.split(":")[-1]
            mapping = {}
            for level in levels:
                if level in ("draft", "d1"):
                    key = "target" if config == "self" else f"draft:{draft_name}"
                elif level in ("ngram", "d2"):
                    key = "ngram_lookup"
                else:
                    key = None
                if key == "target":
                    mapping[level] = 1.0
                else:
                    mapping[level] = table.get(key, 0.0) if key else 0.0
            preds.append(predicted_speedup(tok, levels, mapping))
        total_draft = sum(v for k, v in levels.items() if k != "target")
        print(
            f"{row['config']:14s}{_fmt(row['temperature'], '5.1f'):>5s}"
            f"{row['slots_per_block']:6d}{_fmt(tok):>10s}"
            f"{_fmt(preds[0], '10.2f'):>10s}{_fmt(preds[1], '11.2f'):>11s}"
            f"{_fmt(crossover_cost_ratio(tok, total_draft), '28.3f'):>28s}"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default="bench/results_speculative.json")
    args = parser.parse_args()
    results = json.loads(Path(args.path).read_text())
    print(f"# Speculative decoding results")
    print(f"GPU: {results.get('gpu')}   torch: {results.get('torch')}")
    print(f"target: {results.get('target_checkpoint')}")
    micro_table(results)
    accept_table(results)
    speed_table(results)
    exact_table(results)
    dist_table(results)
    cost_model(results)


if __name__ == "__main__":
    main()
