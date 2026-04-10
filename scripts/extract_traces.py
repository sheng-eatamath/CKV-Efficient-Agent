"""
Extract τ-bench traces into ReplayPlan JSON files.

Usage:
  python scripts/extract_traces.py --tau-bench-dir /path/to/tau-bench --output-dir traces
  
If τ-bench is not available, generates synthetic traces for testing.
"""
from __future__ import annotations

import json
import random
import argparse
from pathlib import Path


# Realistic tool configurations drawn from τ-bench domains
RETAIL_TOOLS = [
    {"name": "get_order_details", "latency_range": (0.5, 2.0), "fail_rate": 0.05},
    {"name": "search_product", "latency_range": (0.8, 3.0), "fail_rate": 0.1},
    {"name": "update_order", "latency_range": (1.0, 4.0), "fail_rate": 0.2},
    {"name": "cancel_order", "latency_range": (0.5, 2.0), "fail_rate": 0.15},
    {"name": "get_user_info", "latency_range": (0.3, 1.0), "fail_rate": 0.02},
]

AIRLINE_TOOLS = [
    {"name": "search_flights", "latency_range": (1.0, 5.0), "fail_rate": 0.1},
    {"name": "get_booking", "latency_range": (0.5, 2.0), "fail_rate": 0.05},
    {"name": "update_booking", "latency_range": (1.0, 4.0), "fail_rate": 0.25},
    {"name": "cancel_booking", "latency_range": (0.5, 2.0), "fail_rate": 0.15},
    {"name": "get_seat_map", "latency_range": (0.8, 3.0), "fail_rate": 0.08},
]


def generate_synthetic_trace(trace_id: str, domain: str,
                             num_turns: int = None) -> dict:
    """Generate a single synthetic trace mimicking τ-bench patterns."""
    tools = RETAIL_TOOLS if domain == "retail" else AIRLINE_TOOLS
    if num_turns is None:
        num_turns = random.randint(6, 14)

    turns = []
    total_tokens = 0
    num_tool_calls = 0
    num_retries = 0

    # System prompt (1000-2000 tokens)
    sys_tokens = random.randint(1000, 2000)
    turns.append({
        "role": "system",
        "token_count": sys_tokens,
    })
    total_tokens += sys_tokens

    # First user message (100-400 tokens)
    user_tokens = random.randint(100, 400)
    turns.append({
        "role": "user",
        "token_count": user_tokens,
    })
    total_tokens += user_tokens

    for i in range(num_turns):
        # Assistant turn (sometimes with tool call)
        asst_tokens = random.randint(50, 300)
        use_tool = random.random() < 0.6
        tool = random.choice(tools) if use_tool else None

        turn = {
            "role": "assistant",
            "token_count": asst_tokens,
        }
        if tool:
            turn["tool_name"] = tool["name"]
            lat = random.uniform(*tool["latency_range"])
            turn["tool_latency_sec"] = round(lat, 2)
            num_tool_calls += 1
        turns.append(turn)
        total_tokens += asst_tokens

        # Tool response
        if tool:
            tool_tokens = random.randint(100, 800)
            success = random.random() > tool["fail_rate"]
            is_retry = not success and random.random() < 0.7
            turns.append({
                "role": "tool",
                "token_count": tool_tokens,
                "tool_name": tool["name"],
                "tool_latency_sec": 0,
                "tool_success": success,
                "is_retry": is_retry,
            })
            total_tokens += tool_tokens
            if is_retry:
                num_retries += 1
        else:
            # User follow-up
            fu_tokens = random.randint(50, 200)
            turns.append({
                "role": "user",
                "token_count": fu_tokens,
            })
            total_tokens += fu_tokens

    return {
        "trace_id": trace_id,
        "domain": domain,
        "turns": turns,
        "total_tokens": total_tokens,
        "num_tool_calls": num_tool_calls,
        "num_retries": num_retries,
    }


def generate_synthetic_traces(output_dir: str, num_per_domain: int = 30):
    """Generate a set of synthetic replay traces."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for domain in ["retail", "airline"]:
        for i in range(num_per_domain):
            trace_id = f"{domain}_{i + 1:03d}"
            trace = generate_synthetic_trace(trace_id, domain)
            path = out / f"{trace_id}.json"
            with open(path, "w") as f:
                json.dump(trace, f, indent=2)

    total = num_per_domain * 2
    print(f"Generated {total} synthetic traces in {output_dir}/")

    # Summary stats
    all_tokens = []
    all_tools = []
    all_retries = []
    for p in out.glob("*.json"):
        with open(p) as f:
            d = json.load(f)
        all_tokens.append(d["total_tokens"])
        all_tools.append(d["num_tool_calls"])
        all_retries.append(d["num_retries"])

    print(f"  Avg tokens/trace: {sum(all_tokens) / len(all_tokens):.0f}")
    print(f"  Avg tool calls:   {sum(all_tools) / len(all_tools):.1f}")
    print(f"  Avg retries:      {sum(all_retries) / len(all_retries):.1f}")


def main():
    parser = argparse.ArgumentParser(description="Extract or generate replay traces")
    parser.add_argument("--tau-bench-dir", default=None,
                        help="Path to cloned τ-bench repo (if available)")
    parser.add_argument("--output-dir", default="traces",
                        help="Output directory for ReplayPlan JSON files")
    parser.add_argument("--num-per-domain", type=int, default=30,
                        help="Number of traces per domain")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    if args.tau_bench_dir and Path(args.tau_bench_dir).exists():
        print(f"τ-bench found at {args.tau_bench_dir}")
        print("Note: τ-bench extraction not yet implemented. Generating synthetic traces.")

    generate_synthetic_traces(args.output_dir, args.num_per_domain)


if __name__ == "__main__":
    main()
