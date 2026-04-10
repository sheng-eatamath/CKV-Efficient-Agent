"""
Controlled A/B experiments: Does prefix caching speedup actually work?

Design:
  For each experiment, compare two conditions:
    COLD: each generate call uses a UNIQUE prefix -> no cache reuse possible
    WARM: generate calls REUSE the SAME prefix -> cache reuse expected

  If prefix caching works, WARM 2nd-call TTFT << COLD 2nd-call TTFT.

  8 experiments:
    E1: Retry cold-vs-warm, prefix=1024
    E2: Retry cold-vs-warm, prefix=2048
    E3: Retry cold-vs-warm, prefix=4096
    E4: Branch cold-vs-warm, branches=2
    E5: Branch cold-vs-warm, branches=4
    E6: Branch cold-vs-warm, branches=8
    E7: Concurrent cold-vs-warm, sessions=4
    E8: Concurrent cold-vs-warm, sessions=8

Usage:
  python scripts/run_speedup_experiments.py --host localhost --port 8000
  python scripts/run_speedup_experiments.py --host localhost --port 8000 --output-dir results/speedup
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src_workflow.vllm_backend import VLLMBackend
from src_workflow.workflows.base import build_filler_prefix


@dataclass
class CallResult:
    label: str
    call_index: int
    ttft_sec: float
    total_sec: float
    prompt_tokens: int
    completion_tokens: int


@dataclass
class ExperimentResult:
    experiment_id: str
    description: str
    params: dict
    calls: list
    warm_first_ttft: float = 0.0
    warm_second_ttft: float = 0.0
    cold_first_ttft: float = 0.0
    cold_second_ttft: float = 0.0
    speedup_ratio: float = 0.0
    cache_hit_detected: bool = False


def make_unique_prefix(base_tokens: int, salt: str) -> str:
    """Build a prefix that is unique (salt ensures no cache hit)."""
    base = build_filler_prefix(base_tokens)
    return f"[UID={salt}] " + base


async def measure_call(backend: VLLMBackend, messages: list[dict],
                       label: str, call_index: int,
                       max_tokens: int = 16) -> CallResult:
    result = await backend.generate(messages, max_tokens=max_tokens, temperature=0.0)
    return CallResult(
        label=label,
        call_index=call_index,
        ttft_sec=result.ttft_sec,
        total_sec=result.total_sec,
        prompt_tokens=result.prompt_tokens or 0,
        completion_tokens=result.completion_tokens or 0,
    )


async def run_retry_experiment(backend: VLLMBackend, prefix_tokens: int,
                               exp_id: str) -> ExperimentResult:
    gen_tokens = 16
    warm_prefix = build_filler_prefix(prefix_tokens)
    warm_msgs = [{"role": "user", "content": warm_prefix + "\nSummarize briefly."}]
    warm_1 = await measure_call(backend, warm_msgs, "warm", 0, gen_tokens)
    warm_2 = await measure_call(backend, warm_msgs, "warm", 1, gen_tokens)

    cold_prefix_1 = make_unique_prefix(prefix_tokens, uuid.uuid4().hex[:8])
    cold_msgs_1 = [{"role": "user", "content": cold_prefix_1 + "\nSummarize briefly."}]
    cold_1 = await measure_call(backend, cold_msgs_1, "cold", 0, gen_tokens)

    cold_prefix_2 = make_unique_prefix(prefix_tokens, uuid.uuid4().hex[:8])
    cold_msgs_2 = [{"role": "user", "content": cold_prefix_2 + "\nSummarize briefly."}]
    cold_2 = await measure_call(backend, cold_msgs_2, "cold", 1, gen_tokens)

    speedup = cold_2.ttft_sec / warm_2.ttft_sec if warm_2.ttft_sec > 0 else 0
    cache_hit = warm_2.ttft_sec < warm_1.ttft_sec * 0.8

    return ExperimentResult(
        experiment_id=exp_id,
        description=f"Retry cold-vs-warm, prefix={prefix_tokens}",
        params={"prefix_tokens": prefix_tokens, "gen_tokens": gen_tokens, "type": "retry"},
        calls=[asdict(c) for c in [warm_1, warm_2, cold_1, cold_2]],
        warm_first_ttft=warm_1.ttft_sec,
        warm_second_ttft=warm_2.ttft_sec,
        cold_first_ttft=cold_1.ttft_sec,
        cold_second_ttft=cold_2.ttft_sec,
        speedup_ratio=speedup,
        cache_hit_detected=cache_hit,
    )


async def run_branch_experiment(backend: VLLMBackend, branch_factor: int,
                                prefix_tokens: int,
                                exp_id: str) -> ExperimentResult:
    gen_tokens = 16
    suffix_tokens = 128

    shared_prefix = build_filler_prefix(prefix_tokens)
    warm_calls = []
    for k in range(branch_factor):
        suffix = f"\n[Branch {k}] Analyze from perspective {k}. " * (suffix_tokens // 10)
        msgs = [{"role": "user", "content": shared_prefix + suffix}]
        c = await measure_call(backend, msgs, "warm", k, gen_tokens)
        warm_calls.append(c)

    cold_calls = []
    for k in range(branch_factor):
        unique_prefix = make_unique_prefix(prefix_tokens, uuid.uuid4().hex[:8])
        suffix = f"\n[Branch {k}] Analyze from perspective {k}. " * (suffix_tokens // 10)
        msgs = [{"role": "user", "content": unique_prefix + suffix}]
        c = await measure_call(backend, msgs, "cold", k, gen_tokens)
        cold_calls.append(c)

    warm_later = [c.ttft_sec for c in warm_calls[1:]]
    cold_later = [c.ttft_sec for c in cold_calls[1:]]
    avg_warm_later = sum(warm_later) / len(warm_later) if warm_later else 0
    avg_cold_later = sum(cold_later) / len(cold_later) if cold_later else 0
    speedup = avg_cold_later / avg_warm_later if avg_warm_later > 0 else 0

    return ExperimentResult(
        experiment_id=exp_id,
        description=f"Branch cold-vs-warm, branches={branch_factor}, prefix={prefix_tokens}",
        params={"branch_factor": branch_factor, "prefix_tokens": prefix_tokens,
                "suffix_tokens": suffix_tokens, "gen_tokens": gen_tokens, "type": "branch"},
        calls=[asdict(c) for c in warm_calls + cold_calls],
        warm_first_ttft=warm_calls[0].ttft_sec,
        warm_second_ttft=avg_warm_later,
        cold_first_ttft=cold_calls[0].ttft_sec,
        cold_second_ttft=avg_cold_later,
        speedup_ratio=speedup,
        cache_hit_detected=avg_warm_later < warm_calls[0].ttft_sec * 0.8,
    )


async def run_concurrent_experiment(backend: VLLMBackend, num_sessions: int,
                                    prefix_tokens: int,
                                    exp_id: str) -> ExperimentResult:
    gen_tokens = 32
    shared_prefix = build_filler_prefix(prefix_tokens)

    async def warm_session(idx):
        msgs = [{"role": "user", "content": shared_prefix + f"\nSession {idx}: summarize."}]
        return await measure_call(backend, msgs, "warm", idx, gen_tokens)

    t0 = time.perf_counter()
    warm_calls = await asyncio.gather(*[warm_session(i) for i in range(num_sessions)])
    warm_wall = time.perf_counter() - t0

    async def cold_session(idx):
        unique_prefix = make_unique_prefix(prefix_tokens, uuid.uuid4().hex[:8])
        msgs = [{"role": "user", "content": unique_prefix + f"\nSession {idx}: summarize."}]
        return await measure_call(backend, msgs, "cold", idx, gen_tokens)

    t0 = time.perf_counter()
    cold_calls = await asyncio.gather(*[cold_session(i) for i in range(num_sessions)])
    cold_wall = time.perf_counter() - t0

    avg_warm_ttft = sum(c.ttft_sec for c in warm_calls) / len(warm_calls)
    avg_cold_ttft = sum(c.ttft_sec for c in cold_calls) / len(cold_calls)
    speedup = avg_cold_ttft / avg_warm_ttft if avg_warm_ttft > 0 else 0

    return ExperimentResult(
        experiment_id=exp_id,
        description=f"Concurrent cold-vs-warm, sessions={num_sessions}, prefix={prefix_tokens}",
        params={"num_sessions": num_sessions, "prefix_tokens": prefix_tokens,
                "gen_tokens": gen_tokens, "type": "concurrent",
                "warm_wall_sec": round(warm_wall, 4),
                "cold_wall_sec": round(cold_wall, 4),
                "wall_speedup": round(cold_wall / warm_wall, 4) if warm_wall > 0 else 0},
        calls=[asdict(c) for c in list(warm_calls) + list(cold_calls)],
        warm_first_ttft=warm_calls[0].ttft_sec,
        warm_second_ttft=avg_warm_ttft,
        cold_first_ttft=cold_calls[0].ttft_sec,
        cold_second_ttft=avg_cold_ttft,
        speedup_ratio=speedup,
        cache_hit_detected=avg_warm_ttft < avg_cold_ttft * 0.8,
    )


async def run_all_experiments(backend: VLLMBackend, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    results: list[ExperimentResult] = []

    experiments = [
        ("E1", lambda: run_retry_experiment(backend, 1024, "E1")),
        ("E2", lambda: run_retry_experiment(backend, 2048, "E2")),
        ("E3", lambda: run_retry_experiment(backend, 4096, "E3")),
        ("E4", lambda: run_branch_experiment(backend, 2, 2048, "E4")),
        ("E5", lambda: run_branch_experiment(backend, 4, 2048, "E5")),
        ("E6", lambda: run_branch_experiment(backend, 8, 2048, "E6")),
        ("E7", lambda: run_concurrent_experiment(backend, 4, 2048, "E7")),
        ("E8", lambda: run_concurrent_experiment(backend, 8, 2048, "E8")),
    ]

    for eid, fn in experiments:
        print(f"\n{'='*60}", flush=True)
        print(f"Running Experiment {eid}...", flush=True)
        print(f"{'='*60}", flush=True)
        try:
            result = await fn()
            results.append(result)
            print(f"  Warm 1st TTFT:  {result.warm_first_ttft*1000:8.2f} ms", flush=True)
            print(f"  Warm 2nd TTFT:  {result.warm_second_ttft*1000:8.2f} ms", flush=True)
            print(f"  Cold 1st TTFT:  {result.cold_first_ttft*1000:8.2f} ms", flush=True)
            print(f"  Cold 2nd TTFT:  {result.cold_second_ttft*1000:8.2f} ms", flush=True)
            print(f"  Speedup ratio:  {result.speedup_ratio:.3f}x", flush=True)
            print(f"  Cache hit:      {result.cache_hit_detected}", flush=True)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  FAILED: {e}", flush=True)

    print("\n" + "=" * 80, flush=True)
    print("SPEEDUP EXPERIMENT SUMMARY", flush=True)
    print("=" * 80, flush=True)
    print(f"{'Exp':>4s}  {'Description':<50s}  {'Warm 2nd':>9s}  {'Cold 2nd':>9s}  {'Speedup':>8s}  {'Cache?':>6s}", flush=True)
    print("-" * 80, flush=True)
    for r in results:
        print(f"{r.experiment_id:>4s}  {r.description:<50s}  "
              f"{r.warm_second_ttft*1000:8.2f}ms  {r.cold_second_ttft*1000:8.2f}ms  "
              f"{r.speedup_ratio:7.3f}x  {'YES' if r.cache_hit_detected else 'no':>6s}", flush=True)
    print("-" * 80, flush=True)

    avg_speedup = sum(r.speedup_ratio for r in results) / len(results) if results else 0
    cache_hits = sum(1 for r in results if r.cache_hit_detected)
    print(f"\nAvg speedup ratio: {avg_speedup:.3f}x", flush=True)
    print(f"Cache hits detected: {cache_hits}/{len(results)}", flush=True)
    if avg_speedup > 1.1:
        print("VERDICT: Prefix caching provides measurable speedup (>10% improvement)", flush=True)
    elif avg_speedup > 1.0:
        print("VERDICT: Prefix caching provides marginal speedup (<10%)", flush=True)
    else:
        print("VERDICT: No measurable speedup", flush=True)

    output = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "num_experiments": len(results),
        "avg_speedup_ratio": round(avg_speedup, 4),
        "cache_hits": cache_hits,
        "experiments": [asdict(r) for r in results],
    }
    out_path = output_dir / "speedup_experiments.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}", flush=True)

    return output


def main():
    parser = argparse.ArgumentParser(description="Prefix Caching Speedup Experiments")
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--output-dir", default="results/speedup")
    args = parser.parse_args()

    print(f"Starting experiments: host={args.host}, port={args.port}, model={args.model}", flush=True)
    backend = VLLMBackend(host=args.host, port=args.port, model=args.model)
    try:
        asyncio.run(run_all_experiments(backend, Path(args.output_dir)))
    except Exception as e:
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
