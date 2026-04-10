"""
CLI entry point for all benchmarks.

Usage:
  python -m scripts.run_benchmark --bench a --config configs/default.yaml
  python -m scripts.run_benchmark --bench b
  python -m scripts.run_benchmark --bench c
  python -m scripts.run_benchmark --bench d
  python -m scripts.run_benchmark --bench all
"""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
import json
import sys
import yaml
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src_workflow.events import EventBus
from src_workflow.vllm_backend import VLLMBackend
from src_workflow.profiler import KVProfiler


async def run_bench_a(backend, event_bus, profiler, cfg):
    from src_workflow.workflows.retry import run_retry_benchmark
    return await run_retry_benchmark(backend, event_bus, cfg["bench_a_retry"])


async def run_bench_b(backend, event_bus, profiler, cfg):
    from src_workflow.workflows.stall import run_concurrency_benchmark
    return await run_concurrency_benchmark(backend, event_bus, cfg["bench_b_concurrency"])


async def run_bench_c(backend, event_bus, profiler, cfg):
    from src_workflow.workflows.branch import run_branch_benchmark
    return await run_branch_benchmark(backend, event_bus, cfg["bench_c_branch"])


async def run_bench_d(backend, event_bus, profiler, cfg):
    from src_workflow.workflows.replay import run_replay_benchmark
    replay_cfg = deepcopy(cfg["bench_d_replay"])
    replay_cfg["max_context_tokens"] = cfg["model"].get("max_model_len")
    return await run_replay_benchmark(backend, event_bus, replay_cfg)


BENCH_MAP = {
    "a": ("bench_a_retry", run_bench_a),
    "b": ("bench_b_concurrency", run_bench_b),
    "c": ("bench_c_branch", run_bench_c),
    "d": ("bench_d_replay", run_bench_d),
}


def apply_runtime_overrides(
    cfg: dict,
    host: str | None = None,
    port: int | None = None,
    model: str | None = None,
) -> dict:
    resolved = deepcopy(cfg)

    if host is not None:
        resolved.setdefault("vllm_server", {})["host"] = host
    if port is not None:
        resolved.setdefault("vllm_server", {})["port"] = port
    if model is not None:
        resolved.setdefault("model", {})["name"] = model

    return resolved


async def _run(bench: str, cfg: dict, run_id: str, output_dir: str):
    event_bus = EventBus()
    backend = VLLMBackend(
        host=cfg["vllm_server"]["host"],
        port=cfg["vllm_server"]["port"],
        model=cfg["model"]["name"],
    )

    # Health check: verify vLLM is reachable before starting
    import aiohttp
    health_url = f"http://{cfg['vllm_server']['host']}:{cfg['vllm_server']['port']}/health"
    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(health_url) as resp:
                if resp.status != 200:
                    print(f"WARNING: vLLM health check returned {resp.status}")
    except Exception as e:
        print(f"ERROR: Cannot connect to vLLM at {health_url}: {e}")
        print("Make sure the vLLM server is running (scripts/start_vllm.sh)")
        sys.exit(1)

    profiler = KVProfiler(
        event_bus, backend, run_id=run_id,
        output_dir=output_dir,
        poll_interval=cfg["profiler"]["poll_interval_sec"],
    )
    profiler.start()

    try:
        if bench == "all":
            all_results = {}
            for b_key, (_, b_fn) in BENCH_MAP.items():
                print(f"\n{'='*60}")
                print(f"Running Benchmark {b_key.upper()}")
                print(f"{'='*60}")
                try:
                    all_results[b_key] = await b_fn(backend, event_bus, profiler, cfg)
                except Exception as e:
                    print(f"Benchmark {b_key} failed: {e}")
                    all_results[b_key] = {"error": str(e)}
            results = all_results
        else:
            _, bench_fn = BENCH_MAP[bench]
            results = await bench_fn(backend, event_bus, profiler, cfg)
    finally:
        profiler.stop()
        summary = profiler.write_summary()
        print("\n" + "=" * 60)
        print("PROFILER SUMMARY")
        print("=" * 60)
        print(json.dumps(summary, indent=2))

    # Save raw results
    out_path = Path(output_dir) / f"{run_id}_raw.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description="CKV-Agent Benchmark Runner")
    parser.add_argument("--bench", required=True,
                        choices=["a", "b", "c", "d", "all"],
                        help="Benchmark to run (a=retry, b=stall, c=branch, d=replay, all)")
    parser.add_argument("--config", default="configs/default.yaml",
                        help="Config YAML file")
    parser.add_argument("--run-id", default=None,
                        help="Run identifier (auto-generated if not set)")
    parser.add_argument("--output-dir", default="results",
                        help="Output directory for results")
    parser.add_argument("--host", default=None,
                        help="Override vLLM host from config")
    parser.add_argument("--port", type=int, default=None,
                        help="Override vLLM port from config")
    parser.add_argument("--model", default=None,
                        help="Override model name from config")
    args = parser.parse_args()

    config_path = PROJECT_ROOT / args.config
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    cfg = apply_runtime_overrides(
        cfg,
        host=args.host,
        port=args.port,
        model=args.model,
    )

    run_id = args.run_id or f"bench_{args.bench}"
    asyncio.run(_run(args.bench, cfg, run_id, args.output_dir))


if __name__ == "__main__":
    main()
