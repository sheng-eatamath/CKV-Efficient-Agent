# Profiler Metrics Update

## Motivation

The existing profiler captured TTFT, end-to-end workflow latency, GPU KV occupancy, and preemption count, but was missing three metrics required for a complete systems paper evaluation:

1. **Throughput** (workflows/min under concurrency)
2. **Decode throughput** (tokens/sec during generation)
3. **Queue wait time** (how long requests wait when KV cache is full)

## Changes

### `src_workflow/profiler.py`

**SessionStats** — added `generate_durations: List[float]` to track per-call total generation time from `total_sec` metadata on `AFTER_GENERATE` events.

**`_on_event()`** — extracts `total_sec` from `AFTER_GENERATE` events into `generate_durations`.

**`write_summary()`** — three new top-level sections in the output JSON:

| Section | Keys | Source |
|---------|------|--------|
| `throughput` | `completed_workflows`, `total_wall_sec`, `workflows_per_min` | Session timestamps |
| `decode_throughput` | `gen_tok_per_s_{mean,max,min}`, `prompt_tok_per_s_mean`, `num_samples` | KV snapshot `gen_throughput` / `prompt_throughput` |
| `queue_wait` | `queue_seconds_total`, `avg_waiting_requests`, `max_waiting_requests`, `pct_time_with_queue` | KV snapshot `num_waiting` time-series integration |

### `tests/test_ckv_workflow.py`

Added `TestProfilerNewMetrics` class with 5 test cases:

- `test_throughput_metric` — verifies `throughput` section with multiple sessions
- `test_decode_throughput_metric` — injects known `gen_throughput` snapshots, checks mean/max/min
- `test_queue_wait_metric` — injects `num_waiting` time-series, validates integral and percentages
- `test_generate_durations_tracked` — confirms `total_sec` is captured in `generate_durations`
- `test_no_snapshots_no_crash` — ensures no crash when KV polling returns no data

## Summary of all profiler metrics (complete)

| Metric from paper | Profiler field | Status |
|---|---|---|
| TTFT | `sessions.*.avg_ttft_sec`, `global.ttft_{p50,p95,mean}` | pre-existing |
| End-to-end workflow latency | `sessions.*.wall_sec` | pre-existing |
| Throughput | `throughput.workflows_per_min` | **new** |
| GPU KV occupancy | `kv_cache.gpu_usage_pct_{mean,max}` | pre-existing |
| Preemption count | `kv_cache.total_preemptions` | pre-existing |
| Decode throughput | `decode_throughput.gen_tok_per_s_{mean,max,min}` | **new** |
| Queue wait time | `queue_wait.{queue_seconds_total,avg_waiting_requests,pct_time_with_queue}` | **new** |

## Non-interference

- No existing fields/logic modified
- No workflow, event, or backend changes
- All 35 pre-existing tests pass unchanged; 5 new tests added (40 total)
- Tested on `cbcb29` with `conda activate sync`: **40/40 passed**
