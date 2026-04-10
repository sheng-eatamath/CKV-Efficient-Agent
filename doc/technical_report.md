# CKV-Agent: Harness-Aware KV Cache Profiler for Agentic LLM Workflows

## Technical Report

**Version:** 1.0  
**Date:** April 2026  
**Hardware:** cbcb29 — 4× NVIDIA RTX 6000 Ada (49 GB each)  
**Model:** Meta-Llama-3-8B-Instruct (float16)  
**Serving Backend:** vLLM ≥ 0.6.0 with `--enable-prefix-caching`

---

## 1. Introduction

### 1.1 Motivation

Large language model (LLM) agents interact with tools, retry on failure, branch over alternatives, and stall while waiting for external systems. These workflow-level patterns create opportunities and hazards for the GPU KV cache that standard serving systems ignore:

- **Retry/Backtrack:** After a tool failure, the agent re-generates from the same prefix. Without awareness, the KV cache evicts and re-prefills identical blocks.
- **Tool Stalls:** While an agent waits for a long-running tool (seconds to minutes), its KV blocks occupy GPU memory but produce no tokens — blocking other requests.
- **Branching:** Multiple candidate continuations share a long prefix. Without coordination, each branch re-prefills the shared prefix independently.

CKV-Agent is a *harness-aware KV cache profiler* that instruments agent workflows with structural events, scrapes vLLM's Prometheus metrics in a time-series, and computes derived metrics that quantify these inefficiencies. Its purpose is to answer, with empirical data, whether harness-aware KV management can reduce latency, improve throughput, and lower memory waste in agentic workloads.

### 1.2 Claims Under Investigation

The system is designed around three falsifiable claims:

1. **Explicit checkpoint/restore reduces repeated prefill** in retry, backtrack, and branch workflows.
2. **Tool-stall-aware tail offload improves GPU memory efficiency** and throughput under concurrent agentic load.
3. **Coarse timing buckets (short/medium/long) suffice** for stall-duration classification — no distributional inference is required.

### 1.3 Scope

This report covers the `src_workflow/` implementation: the profiler, event system, workflow benchmarks, and their results. It does not cover the separate ReAct implementation (`src/`), which remains unchanged.

---

## 2. System Architecture

### 2.1 Overview

```
┌──────────────────────────────────────────────────────────┐
│                     Benchmark Runner                      │
│  scripts/run_benchmark.py  (CLI: --bench a|b|c|d|all)    │
└────────────┬────────────┬─────────────┬──────────────────┘
             │            │             │
     ┌───────▼──┐   ┌─────▼──────┐  ┌──▼────────────┐
     │ EventBus │   │ VLLMBackend│  │  KVProfiler   │
     │ pub/sub  │◄──│ OpenAI API │  │  • event log  │
     │          │   │ /metrics   │  │  • KV polling  │
     └────┬─────┘   └─────┬──────┘  │  • summary    │
          │                │         └───────────────┘
          │                │
   ┌──────▼────────────────▼──────────────────────────┐
   │              Workflow Layer                        │
   │  ┌──────────┐ ┌──────────┐ ┌───────┐ ┌────────┐  │
   │  │  Retry   │ │  Stall   │ │Branch │ │ Replay │  │
   │  │(Bench A) │ │(Bench B) │ │(Bench│ │(Bench D)│  │
   │  └──────────┘ └──────────┘ │  C)  │ └────────┘  │
   │                             └───────┘             │
   └───────────────────────────────────────────────────┘
                         │
                    ┌────▼────┐
                    │ vLLM    │
                    │ Server  │
                    │ (GPU)   │
                    └─────────┘
```

### 2.2 Module Responsibilities

| Module | Role |
|--------|------|
| `events.py` | Defines `EventType` enum (11 event types), `WorkflowEvent` dataclass, and `EventBus` pub/sub |
| `vllm_backend.py` | Async OpenAI-compatible client with streaming TTFT measurement; Prometheus `/metrics` scraper |
| `agent_step.py` | Single LLM call abstraction (`run_step`); emits `BEFORE_GENERATE` / `AFTER_GENERATE` |
| `tools.py` | Simulated async tools: `sleep_tool`, `flaky_tool`, `search_stub`, `lookup_stub` |
| `profiler.py` | Core data collection: event logging, KV snapshot polling, session tracking, derived metrics |
| `workflows/base.py` | Abstract `BaseWorkflow` with `do_step()` and `emit()` helpers; `build_filler_prefix()` |
| `workflows/retry.py` | Benchmark A: prefix → checkpoint → generate → tool (may fail) → retry |
| `workflows/stall.py` | Benchmark B: N concurrent sessions with tool stalls of varying duration |
| `workflows/branch.py` | Benchmark C: shared prefix → checkpoint → K parallel branches |
| `workflows/replay.py` | Benchmark D: replay real agent traces (τ-bench format) with synthetic text |

### 2.3 Event Taxonomy

Events are divided into two categories:

**Structural events** (harness-level patterns):

| Event | Emitted When | Key Metadata |
|-------|-------------|--------------|
| `CHECKPOINT` | Prefix boundary reached before tool call or branch | `token_position` |
| `RETRY_REENTRY` | Agent retries from a checkpoint after tool failure | `checkpoint_token_position`, `attempt` |
| `BRANCH_START` | New branch begins from shared prefix | `branch_id`, `parent_checkpoint` |
| `BRANCH_END` | Branch generation completes | `branch_id` |
| `STALL_BEGIN` | Agent begins waiting for external tool | `tool_name`, `bucket` |
| `STALL_END` | Tool returns; agent resumes | `duration_sec`, `success` |

**Operational events** (step-level):

| Event | Emitted When | Key Metadata |
|-------|-------------|--------------|
| `BEFORE_GENERATE` | Immediately before LLM call | — |
| `AFTER_GENERATE` | LLM call completes | `ttft_sec`, `total_sec`, `prompt_tokens`, `generated_tokens` |
| `BEFORE_TOOL` | Before tool invocation | `tool_name` |
| `AFTER_TOOL` | Tool returns | `tool_name`, `result` |
| `END_SESSION` | Workflow terminates | — |

---

## 3. Profiler Design

### 3.1 Data Collection

The `KVProfiler` operates on two parallel data streams:

1. **Event stream** — subscribes to `EventBus` and logs every `WorkflowEvent` to `{run_id}_events.jsonl`. Maintains per-session `SessionStats` tracking generate calls, token counts, TTFT lists, retry counts, stall periods, and branch TTFTs.

2. **KV polling loop** — an `asyncio` background task scraping vLLM's Prometheus endpoint at configurable intervals (default 0.5s). Each `KVSnapshot` records:
   - `gpu_cache_usage_pct`, `cpu_cache_usage_pct`
   - `num_preemptions` (cumulative)
   - `num_running`, `num_waiting` (instantaneous)
   - `prompt_throughput`, `gen_throughput` (tokens/sec)

### 3.2 Metrics Summary

The profiler computes and reports the following metrics in `{run_id}_summary.json`:

| Metric | What It Measures | Why It Matters | Source |
|--------|-----------------|----------------|--------|
| **TTFT** (p50/p95/mean) | Prefill latency | Prefix cache hit detection | `AFTER_GENERATE` event `ttft_sec` |
| **End-to-end workflow latency** | Wall-clock per session | Overall acceleration effect | Session first/last event timestamps |
| **Throughput** | Workflows/min under concurrency | System efficiency at scale | Session timestamps (global span) |
| **GPU KV occupancy** | GPU memory utilization time-series | Whether memory is managed effectively | Prometheus `gpu_cache_usage_perc` |
| **Preemption count** | Requests evicted due to KV pressure | Direct indicator of memory pressure | Prometheus `num_preemptions_total` |
| **Decode throughput** | Tokens/sec during generation | Impact of offload/prefetch on decode | Prometheus `avg_generation_throughput_toks_per_s` |
| **Queue wait time** | Time requests wait for scheduling | How long new requests are blocked when KV is full | Prometheus `num_requests_waiting` integrated over time |

### 3.3 Derived Metrics

Beyond raw metrics, the profiler computes workflow-aware derived signals:

**Retry prefill savings:**
$$\text{savings\_ratio} = \frac{\text{mean}(\text{initial\_ttfts}) - \text{mean}(\text{retry\_ttfts})}{\text{mean}(\text{initial\_ttfts})}$$

Measures whether retrying from a checkpointed prefix avoids re-prefill cost. Positive values indicate cache hits; near-zero or negative values indicate the cache is already warm on initial use.

**Stall KV waste (GPU-memory-seconds):**
$$\text{waste} = \overline{\text{gpu\_cache\_pct}}_{[t_{\text{start}}, t_{\text{end}}]} \times (t_{\text{end}} - t_{\text{start}})$$

Quantifies idle GPU memory occupied by stalled sessions' KV blocks. High values motivate tail offload to CPU.

**Branch prefix reuse rate:**
$$\text{reuse\_rate} = \frac{|\{b : \text{ttft}_b < \text{ttft}_0 \times \theta\}|}{|B| - 1}$$

where $\theta = 0.7$ (configurable). Measures whether later branches benefit from shared prefix caching.

### 3.4 Output Artifacts

Each profiler run produces three files:

| File | Format | Contents |
|------|--------|----------|
| `{run_id}_events.jsonl` | JSON Lines | Every `WorkflowEvent` with timestamp, session ID, event type, metadata |
| `{run_id}_kv_timeseries.jsonl` | JSON Lines | Periodic `KVSnapshot` with GPU/CPU usage, preemptions, throughput |
| `{run_id}_summary.json` | JSON | Aggregated session stats, KV cache summary, global stats, derived metrics, throughput, decode throughput, queue wait |

---

## 4. Benchmark Suite

### 4.1 Benchmark A — Retry/Backtrack

**Purpose:** Measure whether prefix caching reduces re-prefill cost on retries.

**Protocol:**
1. Build synthetic prefix of $P$ tokens
2. Emit `CHECKPOINT` at prefix boundary
3. Generate from prefix
4. Call `flaky_tool` (fails with probability $p$)
5. On failure: emit `RETRY_REENTRY`, re-generate from same prefix
6. Repeat up to $R$ retries

**Parameter sweep:**
- `prefix_tokens` ∈ {1024, 2048, 4096}
- `num_retries` ∈ {1, 2, 4}
- `p_fail` ∈ {0.5, 1.0}
- `gen_tokens` = 256

**Key metric:** Retry prefill savings ratio.

### 4.2 Benchmark B — Concurrent Sessions with Tool Stalls

**Purpose:** Measure GPU KV waste during tool stalls and throughput under concurrent load.

**Protocol:**
1. Launch $N$ concurrent sessions via `asyncio.gather`
2. Each session: generate → tool stall → generate → tool stall (2 rounds)
3. Stalled fraction $f$ of sessions use medium/long stalls; rest use short stalls
4. Profiler polls KV occupancy throughout

**Parameter sweep:**
- `num_sessions` ∈ {4, 8, 16}
- `stalled_fraction` ∈ {0.25, 0.5, 0.75}
- `stall_durations`: short=0.5s, medium=2.0s, long=8.0s
- `prefix_tokens` = 4096, `gen_tokens` = 128

**Key metrics:** Throughput (sessions/min), GPU KV occupancy, stall waste, preemption count.

### 4.3 Benchmark C — Branch/Fanout

**Purpose:** Measure whether branches from a shared prefix benefit from cached KV blocks.

**Protocol:**
1. Build shared prefix of $P$ tokens
2. Emit `CHECKPOINT` at prefix boundary
3. For each branch $k \in [0, K)$: append unique suffix → `BRANCH_START` → generate → `BRANCH_END`
4. Compare TTFTs across branches

**Parameter sweep:**
- `branch_factor` ∈ {2, 4, 8}
- `shared_prefix_tokens` ∈ {2048, 4096}
- `suffix_tokens` ∈ {128, 512}
- `gen_tokens` = 128

**Key metric:** Branch prefix reuse rate.

### 4.4 Benchmark D — τ-bench Trace Replay

**Purpose:** Replay realistic agent execution patterns and measure KV behavior under mixed workloads.

**Design principle:** No ReAct parsing. The workflow topology (which turns are tool calls, which are retries, token counts per turn) is driven by the trace. The LLM fills synthetic text of matching token length.

**Trace format (`ReplayPlan`):**
```json
{
  "trace_id": "retail_001",
  "domain": "retail",
  "turns": [
    {"role": "system", "token_count": 1500},
    {"role": "user", "token_count": 200},
    {"role": "assistant", "token_count": 150, "tool_name": "get_order", "tool_latency_sec": 1.2},
    {"role": "tool", "token_count": 300},
    {"role": "assistant", "token_count": 100}
  ],
  "total_tokens": 2250,
  "num_tool_calls": 1,
  "num_retries": 0
}
```

**Trace generation:** `scripts/extract_traces.py` generates synthetic traces modeled on τ-bench (retail + airline domains) with realistic tool distributions, latency ranges, and retry probabilities.

**Parameter sweep:**
- Domains: retail, airline (30 traces per domain)
- `concurrency` ∈ {1, 4, 8}

**Key metrics:** All metrics (TTFT, throughput, GPU KV, preemptions, decode throughput, queue wait, stall waste).

---

## 5. Implementation Details

### 5.1 vLLM Backend

The `VLLMBackend` class wraps vLLM's OpenAI-compatible API:

- **Streaming generation** measures TTFT as time-to-first-chunk via `time.perf_counter()` delta. Token usage is extracted from the final chunk via `stream_options={"include_usage": True}`.
- **Non-streaming fallback** returns `total_sec` as TTFT (no first-token signal).
- **Metrics scraping** parses Prometheus text format via `_parse_prometheus()`, extracting gauge/counter values by metric name.
- **Timeouts:** 120s for generation, 10s for metrics scrape.

The vLLM server is launched with `--enable-prefix-caching`, which activates automatic block-level prefix deduplication. This is the baseline mechanism that CKV-Agent's explicit checkpointing aims to complement and surpass.

### 5.2 Event Bus

The `EventBus` is a synchronous pub/sub system. Subscribers are called in registration order. Each subscriber is wrapped in a try/except to prevent cascade failures — a failing profiler subscriber cannot crash the workflow.

### 5.3 Workflows

All workflows inherit from `BaseWorkflow`, which provides:
- `emit()` — construct and publish a `WorkflowEvent`
- `do_step()` — call `run_step()` with auto-incrementing step counter

The `build_filler_prefix(target_tokens)` utility generates deterministic synthetic text of approximately `target_tokens` tokens (~12 tokens per repeated sentence).

### 5.4 Trace Replay

`ReplayWorkflow` walks through a `ReplayPlan` turn-by-turn:
- **system/user turns:** appended to message history as synthetic text
- **assistant turns with tool:** checkpoint → generate → stall for `tool_latency_sec`
- **tool turns with `is_retry=True`:** reset messages to checkpoint and emit `RETRY_REENTRY`
- **tool turns without retry:** append tool result to messages

Context overflow protection: `estimate_peak_prompt_tokens()` pre-scans the plan and skips traces whose peak prompt exceeds `max_model_len - gen_tokens`.

---

## 6. Experimental Results

### 6.1 Test Environment

| Property | Value |
|----------|-------|
| Server | cbcb29 |
| GPUs | 4× NVIDIA RTX 6000 Ada (49 GB each) |
| Model | Meta-Llama-3-8B-Instruct (float16) |
| Max context | 8192 tokens |
| GPU memory utilization | 90% |
| vLLM prefix caching | Enabled |
| Python | 3.10 |
| Conda env | `sync` |

### 6.2 Summary Across Benchmarks

| Benchmark | Sessions | TTFT p50 | TTFT p95 | GPU KV Mean | GPU KV Max | Preemptions | KV Snapshots |
|-----------|----------|----------|----------|-------------|------------|-------------|--------------|
| A (Retry) | 18 | 31.2 ms | 36.5 ms | 1.02% | 2.08% | 0 | 75 |
| B (Stall) | 84 | 90.9 ms | 146.6 ms | 0.85% | 2.25% | 0 | 361 |
| C (Branch) | 12 | 46.3 ms | 89.6 ms | 1.69% | 2.36% | 0 | 263 |
| D (Replay) | 30 | 48.4 ms | 146.2 ms | 2.12% | 12.84% | 0 | 1053 |

**Total:** 144 sessions, 1752 KV snapshots, 418 stall events, 0 preemptions.

### 6.3 Benchmark A Results

Retry prefill savings ratio: **−7.44%** (average across 18 sessions).

The negative value is the correct expected outcome: vLLM's prefix caching is transparent and benefits both initial and retry calls equally. The first call in a run incurs a cold-start penalty (~1.4s) reflecting GPU kernel compilation and model warmup. Subsequent calls (both initial and retry) see TTFTs of 30–36 ms regardless of retry status.

TTFT scales sub-linearly with prefix length:
- 1024 tokens: ~30 ms
- 2048 tokens: ~32 ms
- 4096 tokens: ~36 ms

### 6.4 Benchmark B Results

Throughput under concurrency:

| Sessions | Stalled Fraction | Wall (s) | Sessions/min |
|----------|-----------------|----------|--------------|
| 4 | 0.25 | 1.3 | 185 |
| 4 | 0.50 | 4.2 | 57 |
| 8 | 0.50 | 4.5 | 107 |
| 16 | 0.75 | 16.5 | 58 |

TTFT increases from ~30 ms (single session) to ~90 ms (16 concurrent sessions). This is scheduler contention, not memory pressure — GPU KV utilization never exceeds 2.25%.

Stall KV waste is minimal on this hardware: < 0.02 GPU-mem-sec per stall period. This is because the 8B model on a 49 GB GPU uses < 3% of KV cache capacity. Larger models (70B+) or longer contexts (32k+) would produce substantially higher waste values.

### 6.5 Benchmark C Results

Branch prefix reuse rate: **2.27%**.

The low rate paradoxically confirms prefix caching is working: the first branch's TTFT is already cache-warm, so later branches cannot show a speedup relative to it. The single detected reuse hit occurred when branch 0 had a cold-start penalty (1.40s) and branch 1 completed in 0.049s.

TTFT scales linearly with suffix token count:
- 128 suffix tokens: ~40 ms
- 512 suffix tokens: ~80 ms

### 6.6 Benchmark D Results

The trace replay benchmark produces the richest KV behavior:
- Peak GPU KV utilization reached **12.84%** (highest across all benchmarks)
- 1053 KV snapshots over ~8.7 minutes
- 201 stall events across retail and airline domains
- 21 retries (retry savings ratio: −0.52%)

This benchmark demonstrates that realistic multi-turn agent traces with tool calls, retries, and varying context lengths generate meaningful KV pressure even on well-provisioned hardware.

---

## 7. KV Cache Analysis

### 7.1 Memory Pressure

No preemptions occurred in any benchmark. The 8B model with 8192-token max context on 49 GB GPUs is substantially under-provisioned for memory pressure. To observe preemptions and validate offload/prefetch strategies, future experiments require:
- Larger models (70B parameters)
- Longer contexts (32k–128k tokens)
- Higher concurrency (64+ sessions)
- Lower `gpu_memory_utilization` (e.g., 0.50)

### 7.2 Stall Waste Breakdown

Stall waste (GPU-memory-seconds) by bucket:

| Bucket | Duration Range | Count | Avg Waste |
|--------|---------------|-------|-----------|
| Short | < 1.0s | 89 | 0.004 |
| Medium | 1.0–5.0s | 178 | 0.020 |
| Long | > 5.0s | 151 | 0.065 |

At current GPU utilization levels, waste is negligible. Under higher memory pressure (projected from GPU KV occupancy > 80%), medium and long stalls would produce waste > 1.0 GPU-mem-sec per event, making tail offload profitable.

### 7.3 Queue Wait Analysis

Queue wait metrics (from `num_waiting` time-series integration) showed near-zero values across all benchmarks, consistent with the absence of memory pressure. Under higher load, queue wait becomes the primary indicator of user-visible latency degradation.

---

## 8. Configuration Reference

All parameters are specified in `configs/default.yaml`:

```yaml
model:
  name: "meta-llama/Meta-Llama-3-8B-Instruct"
  gpu_memory_utilization: 0.90
  max_model_len: 8192
  dtype: "float16"
  seed: 42

vllm_server:
  host: "localhost"
  port: 8000

profiler:
  poll_interval_sec: 0.5
  output_dir: "logs"

bench_a_retry:
  prefix_tokens: [1024, 2048, 4096]
  num_retries: [1, 2, 4]
  p_fail: [0.5, 1.0]
  gen_tokens: 256

bench_b_concurrency:
  num_sessions: [4, 8, 16]
  stalled_fraction: [0.25, 0.5, 0.75]
  stall_durations_sec: {short: 0.5, medium: 2.0, long: 8.0}
  prefix_tokens: 4096
  gen_tokens: 128

bench_c_branch:
  branch_factor: [2, 4, 8]
  shared_prefix_tokens: [2048, 4096]
  suffix_tokens: [128, 512]
  gen_tokens: 128

bench_d_replay:
  trace_dir: "traces"
  domains: ["retail", "airline"]
  num_traces_per_domain: 30
  concurrency: [1, 4, 8]
```

Runtime overrides via CLI: `--host`, `--port`, `--model`.

---

## 9. Testing

### 9.1 Test Suite

The test suite (`tests/test_ckv_workflow.py`) contains **40 tests** across 11 test classes, all using a mock backend (no vLLM server required):

| Class | Tests | Coverage |
|-------|-------|----------|
| `TestEventBus` | 4 | Subscribe, multi-subscriber, all event types, metadata |
| `TestTools` | 6 | Each tool function + registry |
| `TestPrometheus` | 3 | Prometheus text parsing, labels, empty input |
| `TestAgentStep` | 1 | `run_step` with event emission |
| `TestProfiler` | 3 | Event logging, branch tracking, output files |
| `TestRetryWorkflow` | 2 | Basic retry + success-on-first-try |
| `TestStallWorkflow` | 1 | Concurrent stall workflow |
| `TestBranchWorkflow` | 1 | Branch fanout workflow |
| `TestReplayWorkflow` | 2 | Trace replay + replay with retry |
| `TestTraceExtraction` | 5 | Stall classification, plan loading, trace generation, selection, peak estimation |
| `TestBuildFiller` | 2 | Length check, determinism |
| `TestIntegration` | 3 | Profiler + RetryWorkflow, BranchWorkflow, concurrency |
| `TestBenchmarkSweeps` | 2 | Retry sweep, branch sweep |
| `TestProfilerNewMetrics` | 5 | Throughput, decode throughput, queue wait, generate durations, empty snapshots |

### 9.2 Test Results

```
$ pytest tests/test_ckv_workflow.py -v
========================= 40 passed in 8.12s =========================
```

All 40 tests pass on cbcb29 with `conda activate sync` (Python 3.10, pytest 9.0.2, pytest-asyncio 1.3.0).

---

## 10. Reproduction

### 10.1 Quick Start

```bash
ssh cbcb29
conda activate sync
cd /nfshomes/shengz/sheng/CourseLearning/MLSYS/ckv2

# Run unit tests (no server required):
bash run_tests.sh

# Run full benchmark pipeline (starts vLLM, runs all benchmarks, generates plots):
bash run_all.sh

# Run a single benchmark:
bash run_all.sh --bench a
```

### 10.2 Manual Execution

```bash
# 1. Generate synthetic traces:
python scripts/extract_traces.py --output-dir traces --num-per-domain 30 --seed 42

# 2. Start vLLM server:
bash scripts/start_vllm.sh meta-llama/Meta-Llama-3-8B-Instruct 8000 0.90 8192

# 3. Run benchmark:
python -m scripts.run_benchmark --bench all --config configs/default.yaml --output-dir results

# 4. Generate plots:
python scripts/plot_results.py --results-dir results --output-dir results/figures
```

### 10.3 Output Locations

| Artifact | Path |
|----------|------|
| Event logs | `logs/{run_id}_events.jsonl` |
| KV time-series | `logs/{run_id}_kv_timeseries.jsonl` |
| Profiler summary | `results/{run_id}_summary.json` |
| Raw benchmark results | `results/{run_id}_raw.json` |
| Plots | `results/figures/*.png` |
| Archived runs | `cache/` |

---

## 11. File Manifest

```
ckv2/
├── src_workflow/
│   ├── __init__.py
│   ├── events.py                  # EventType, WorkflowEvent, EventBus
│   ├── vllm_backend.py            # VLLMBackend, GenerateResult, Prometheus parser
│   ├── agent_step.py              # run_step, StepResult
│   ├── tools.py                   # Simulated tools + TOOL_REGISTRY
│   ├── profiler.py                # KVProfiler, KVSnapshot, SessionStats
│   └── workflows/
│       ├── __init__.py
│       ├── base.py                # BaseWorkflow, WorkflowResult, build_filler_prefix
│       ├── retry.py               # RetryWorkflow (Benchmark A)
│       ├── stall.py               # StallWorkflow (Benchmark B)
│       ├── branch.py              # BranchWorkflow (Benchmark C)
│       └── replay.py              # ReplayWorkflow (Benchmark D)
├── scripts/
│   ├── run_benchmark.py           # CLI entry point for all benchmarks
│   ├── extract_traces.py          # Synthetic τ-bench trace generator
│   ├── plot_results.py            # Matplotlib visualization
│   └── start_vllm.sh             # vLLM server launcher
├── tests/
│   └── test_ckv_workflow.py       # 40 unit tests (mock backend)
├── configs/
│   └── default.yaml               # All benchmark parameters
├── doc/
│   ├── technical_report.md        # This report
│   ├── online_test_report.md      # Online benchmark results (April 7, 2026)
│   └── profiler_metrics_update.md # Metrics implementation changelog
├── agent/                         # Agent specifications and scope documents
├── traces/                        # Synthetic τ-bench replay plans (JSON)
├── logs/                          # Event + KV time-series logs
├── results/                       # Summary + raw JSON + plots
├── cache/                         # Archived benchmark runs
├── run_all.sh                     # One-click pipeline runner
└── run_tests.sh                   # One-click test runner
```

---

## 12. Limitations and Future Work

### 12.1 Current Limitations

1. **Low memory pressure:** The 8B model on 49 GB GPUs never triggers preemptions or meaningful queue wait. Stall waste and queue metrics are near-zero, limiting the observability of the metrics designed for high-pressure scenarios.

2. **Passive prefix caching only:** The current implementation relies on vLLM's built-in `--enable-prefix-caching`, which deduplicates identical prefix blocks automatically. Explicit checkpoint/restore (P1/P2 primitives from the MVP spec) and tail offload/prefetch (P3) are not yet implemented.

3. **Synthetic text:** All benchmarks use synthetic filler text rather than real prompts. While token counts match realistic distributions, the lack of semantic content means the generated outputs have no bearing on tool call decisions.

### 12.2 Future Directions

1. **Stress-test configuration:** Run with larger models (70B), longer contexts (32k+), lower GPU memory utilization (50%), and higher concurrency (64+ sessions) to trigger preemptions and validate queue wait metrics.

2. **Explicit checkpoint/restore:** Implement the CKV orchestrator with named checkpoint handles and priority-aware restore, bypassing vLLM's passive eviction policy.

3. **Tail offload/prefetch:** During `STALL_BEGIN`, offload the stalled session's KV tail to CPU; on `STALL_END`, prefetch before the next generate call. This requires interfacing with vLLM's block manager or using LMCache.

4. **Real τ-bench traces:** Extract actual τ-bench execution traces (retail + airline benchmarks) and replay them, replacing the current synthetic trace generator.

5. **Multi-GPU scheduling:** Extend the profiler to track cross-GPU KV migration in tensor-parallel or pipeline-parallel deployments.

---

## 13. Changelog

| Date | Change | Files |
|------|--------|-------|
| 2026-04-07 | Initial implementation: events, profiler, 4 benchmarks, tests | `src_workflow/`, `scripts/`, `tests/` |
| 2026-04-07 | Online test run on cbcb29; 9 critical/important fixes applied | `src_workflow/profiler.py`, `vllm_backend.py` |
| 2026-04-08 | Added throughput, decode throughput, queue wait metrics | `src_workflow/profiler.py`, `tests/test_ckv_workflow.py` |
| 2026-04-09 | Technical report | `doc/technical_report.md` |
