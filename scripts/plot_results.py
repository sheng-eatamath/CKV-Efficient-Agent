"""
Plot benchmark results from profiler output.

Usage:
  python scripts/plot_results.py --results-dir results --output-dir results/figures
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    print("Warning: matplotlib not available. Skipping plots.")


def plot_retry_savings(summary: dict, raw: dict, output_dir: Path, run_id: str = ""):
    """Plot retry TTFT savings (Benchmark A)."""
    if not HAS_MPL:
        return

    savings = summary.get("derived_metrics", {}).get("retry_prefill_savings", [])
    if not savings:
        print("No retry savings data to plot.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart: initial vs retry TTFT
    sessions = [s["session_id"][:20] for s in savings]
    initial = [s["initial_ttft_avg"] * 1000 for s in savings]
    retry = [s["retry_ttft_avg"] * 1000 for s in savings]

    x = range(len(sessions))
    width = 0.35
    ax1.bar([i - width / 2 for i in x], initial, width, label="Initial TTFT", color="#2196F3")
    ax1.bar([i + width / 2 for i in x], retry, width, label="Retry TTFT", color="#4CAF50")
    ax1.set_xlabel("Session")
    ax1.set_ylabel("TTFT (ms)")
    ax1.set_title("Retry TTFT Savings")
    ax1.legend()
    ax1.tick_params(axis="x", rotation=45)

    # Savings ratio
    ratios = [s["savings_ratio"] for s in savings]
    ax2.bar(range(len(ratios)), ratios, color="#FF9800")
    ax2.set_xlabel("Session")
    ax2.set_ylabel("Savings Ratio")
    ax2.set_title("Retry Prefill Savings Ratio")
    ax2.axhline(y=0, color="gray", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_dir / f"{run_id}_retry_savings.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {run_id}_retry_savings.png")


def plot_kv_timeline(summary: dict, kv_log_path: Path, output_dir: Path, run_id: str = ""):
    """Plot KV cache usage over time."""
    if not HAS_MPL:
        return

    if not kv_log_path.exists():
        return

    timestamps = []
    gpu_usages = []
    with open(kv_log_path) as f:
        for line in f:
            if not line.strip():
                continue
            snap = json.loads(line)
            timestamps.append(snap["timestamp"])
            gpu_usages.append(snap["gpu_cache_usage_pct"] * 100)

    if not timestamps:
        return

    t0 = timestamps[0]
    rel_times = [t - t0 for t in timestamps]

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(rel_times, gpu_usages, color="#2196F3", linewidth=1.5)
    ax.fill_between(rel_times, gpu_usages, alpha=0.2, color="#2196F3")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("GPU KV Cache Usage (%)")
    ax.set_title("KV Cache GPU Occupancy Over Time")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_dir / f"{run_id}_kv_timeline.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {run_id}_kv_timeline.png")


def plot_branch_reuse(summary: dict, raw: dict, output_dir: Path, run_id: str = ""):
    """Plot branch TTFT showing prefix reuse."""
    if not HAS_MPL:
        return

    reuse = summary.get("derived_metrics", {}).get("branch_prefix_reuse", [])
    if not reuse:
        print("No branch reuse data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    branch_ids = [r["branch_id"] for r in reuse]
    first_ttfts = [r["first_branch_ttft"] * 1000 for r in reuse]
    branch_ttfts = [r["branch_ttft"] * 1000 for r in reuse]
    hits = [r["reuse_hit"] for r in reuse]

    colors = ["#4CAF50" if h else "#F44336" for h in hits]
    ax.bar(range(len(branch_ids)), branch_ttfts, color=colors, alpha=0.8)
    ax.axhline(y=first_ttfts[0] if first_ttfts else 0, color="blue",
               linestyle="--", label="First branch TTFT", alpha=0.7)
    ax.set_xlabel("Branch Index")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Branch TTFT — Prefix Cache Reuse")
    ax.legend()

    plt.tight_layout()
    fig.savefig(output_dir / f"{run_id}_branch_reuse.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {run_id}_branch_reuse.png")


def plot_stall_waste(summary: dict, output_dir: Path, run_id: str = ""):
    """Plot stall KV waste analysis."""
    if not HAS_MPL:
        return

    stalls = summary.get("derived_metrics", {}).get("stall_periods", [])
    if not stalls:
        print("No stall data to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    durations = [s["duration_sec"] for s in stalls]
    buckets = [s.get("bucket", "unknown") for s in stalls]
    colors_map = {"short": "#4CAF50", "medium": "#FF9800", "long": "#F44336", "unknown": "#9E9E9E"}
    colors = [colors_map.get(b, "#9E9E9E") for b in buckets]

    ax.bar(range(len(durations)), durations, color=colors, alpha=0.8)
    ax.set_xlabel("Stall Event Index")
    ax.set_ylabel("Stall Duration (sec)")
    ax.set_title("Tool Stall Durations (KV blocks idle on GPU)")

    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=c, label=l)
                       for l, c in colors_map.items() if l in set(buckets)]
    ax.legend(handles=legend_elements)

    plt.tight_layout()
    fig.savefig(output_dir / f"{run_id}_stall_waste.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {run_id}_stall_waste.png")


def generate_all_plots(results_dir: str, output_dir: str):
    results_path = Path(results_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Find summary and raw files
    summaries = sorted(results_path.glob("*_summary.json"))
    # Also check logs/
    logs_dir = results_path.parent / "logs"
    if logs_dir.exists():
        summaries.extend(sorted(logs_dir.glob("*_summary.json")))

    for summary_file in summaries:
        print(f"\nProcessing: {summary_file}")
        with open(summary_file) as f:
            summary = json.load(f)

        run_id = summary.get("run_id", summary_file.stem.replace("_summary", ""))

        # Find corresponding raw file
        raw = {}
        raw_file = summary_file.parent / f"{run_id}_raw.json"
        if raw_file.exists():
            with open(raw_file) as f:
                raw = json.load(f)

        # Find KV timeseries
        kv_log = summary_file.parent / f"{run_id}_kv_timeseries.jsonl"

        # Generate plots
        plot_retry_savings(summary, raw, out_path, run_id)
        plot_kv_timeline(summary, kv_log, out_path, run_id)
        plot_branch_reuse(summary, raw, out_path, run_id)
        plot_stall_waste(summary, out_path, run_id)


def main():
    parser = argparse.ArgumentParser(description="Plot CKV-Agent benchmark results")
    parser.add_argument("--results-dir", default="results",
                        help="Directory containing result JSON files")
    parser.add_argument("--output-dir", default="results/figures",
                        help="Output directory for plots")
    args = parser.parse_args()

    generate_all_plots(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
