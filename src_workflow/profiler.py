"""
Harness-Aware KV Cache Profiler.

Subscribes to EventBus to receive workflow events.
Runs a background polling loop to scrape vLLM /metrics.
Produces time-series logs + summary with derived metrics.

Output files per run:
  - {run_id}_events.jsonl
  - {run_id}_kv_timeseries.jsonl
  - {run_id}_summary.json
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, Dict, List

from .events import WorkflowEvent, EventBus, EventType
from .vllm_backend import VLLMBackend


@dataclass
class KVSnapshot:
    timestamp: float
    gpu_cache_usage_pct: float
    cpu_cache_usage_pct: float
    num_preemptions: float
    num_running: float
    num_waiting: float
    prompt_throughput: float
    gen_throughput: float


@dataclass
class SessionStats:
    session_id: str
    total_generate_calls: int = 0
    total_prompt_tokens: int = 0
    total_gen_tokens: int = 0
    ttft_list: List[float] = field(default_factory=list)
    retry_count: int = 0
    tool_stall_sec: float = 0.0
    first_event_ts: float = 0.0
    last_event_ts: float = 0.0
    # Harness-aware tracking
    checkpoint_positions: List[int] = field(default_factory=list)
    retry_ttfts: List[float] = field(default_factory=list)
    initial_ttfts: List[float] = field(default_factory=list)
    branch_ttfts: Dict[int, float] = field(default_factory=dict)
    stall_periods: List[dict] = field(default_factory=list)
    generate_durations: List[float] = field(default_factory=list)


class KVProfiler:
    def __init__(self, event_bus: EventBus, backend: VLLMBackend,
                 run_id: str, output_dir: str = "logs",
                 poll_interval: float = 0.5,
                 branch_reuse_threshold: float = 0.7):
        self.event_bus = event_bus
        self.backend = backend
        self.run_id = run_id
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.poll_interval = poll_interval
        self.branch_reuse_threshold = branch_reuse_threshold

        self._event_log_path = self.output_dir / f"{run_id}_events.jsonl"
        self._kv_log_path = self.output_dir / f"{run_id}_kv_timeseries.jsonl"
        self._summary_path = self.output_dir / f"{run_id}_summary.json"

        self._event_file = None
        self._kv_file = None
        self._poll_task: Optional[asyncio.Task] = None
        self._sessions: Dict[str, SessionStats] = {}
        self._kv_snapshots: List[KVSnapshot] = []

        # State tracking for derived metrics
        # Separate dicts for STALL_BEGIN/END vs BEFORE_TOOL/AFTER_TOOL
        self._stall_start: Dict[str, float] = {}       # STALL_BEGIN/END
        self._stall_begin_info: Dict[str, dict] = {}   # STALL_BEGIN metadata
        self._tool_start: Dict[str, float] = {}        # BEFORE_TOOL/AFTER_TOOL
        self._is_retry: Dict[str, bool] = {}
        self._current_branch: Dict[str, int] = {}

        self.event_bus.subscribe(self._on_event)

    def _get_session(self, sid: str, ts: float) -> SessionStats:
        if sid not in self._sessions:
            self._sessions[sid] = SessionStats(session_id=sid, first_event_ts=ts)
        return self._sessions[sid]

    def _on_event(self, ev: WorkflowEvent):
        record = asdict(ev)
        record["event_type"] = ev.event_type.name
        self._write_jsonl(self._event_file, record)

        ss = self._get_session(ev.session_id, ev.timestamp)
        ss.last_event_ts = ev.timestamp

        if ev.event_type == EventType.CHECKPOINT:
            token_pos = ev.meta.get("token_position", ev.prompt_tokens)
            ss.checkpoint_positions.append(token_pos)

        elif ev.event_type == EventType.RETRY_REENTRY:
            ss.retry_count += 1
            self._is_retry[ev.session_id] = True

        elif ev.event_type == EventType.AFTER_GENERATE:
            ss.total_generate_calls += 1
            if ev.prompt_tokens is not None and ev.prompt_tokens > 0:
                ss.total_prompt_tokens += ev.prompt_tokens
            if ev.generated_tokens is not None and ev.generated_tokens > 0:
                ss.total_gen_tokens += ev.generated_tokens

            total_sec = ev.meta.get("total_sec")
            if total_sec is not None:
                ss.generate_durations.append(total_sec)

            ttft = ev.meta.get("ttft_sec")
            if ttft is not None:
                ss.ttft_list.append(ttft)
                if self._is_retry.get(ev.session_id, False):
                    ss.retry_ttfts.append(ttft)
                    # Clear flag after use — next AFTER_GENERATE is initial unless new RETRY_REENTRY
                    self._is_retry[ev.session_id] = False
                else:
                    ss.initial_ttfts.append(ttft)

                branch_id = self._current_branch.get(ev.session_id)
                if branch_id is not None:
                    ss.branch_ttfts[branch_id] = ttft

        elif ev.event_type == EventType.STALL_BEGIN:
            self._stall_start[ev.session_id] = ev.timestamp
            self._stall_begin_info[ev.session_id] = {
                "tool_name": ev.tool_name,
                "bucket": ev.meta.get("bucket", "unknown"),
                "start_ts": ev.timestamp,
                "gpu_pct_at_start": self._latest_gpu_usage(),
            }

        elif ev.event_type == EventType.STALL_END:
            start = self._stall_start.pop(ev.session_id, None)
            begin_info = self._stall_begin_info.pop(ev.session_id, {})
            if start is not None:
                duration = ev.timestamp - start
                ss.tool_stall_sec += duration
                # Compute GPU-memory-seconds waste from KV snapshots
                mean_gpu = self._mean_gpu_during(start, ev.timestamp)
                waste = mean_gpu * duration
                ss.stall_periods.append({
                    "tool_name": begin_info.get("tool_name"),
                    "bucket": begin_info.get("bucket"),
                    "duration_sec": round(duration, 4),
                    "mean_gpu_pct": round(mean_gpu, 4),
                    "waste_gpu_mem_sec": round(waste, 4),
                })

        elif ev.event_type == EventType.BEFORE_TOOL:
            self._tool_start[ev.session_id] = ev.timestamp

        elif ev.event_type == EventType.AFTER_TOOL:
            # Only record duration — don't add to tool_stall_sec to avoid
            # double-counting when STALL_BEGIN/END is also emitted.
            self._tool_start.pop(ev.session_id, None)

        elif ev.event_type == EventType.BRANCH_START:
            branch_id = ev.meta.get("branch_id", ev.branch_k)
            self._current_branch[ev.session_id] = branch_id

        elif ev.event_type == EventType.BRANCH_END:
            self._current_branch.pop(ev.session_id, None)

    def _latest_gpu_usage(self) -> float:
        """Return the most recent GPU cache usage pct, or 0 if no snapshots."""
        if self._kv_snapshots:
            return self._kv_snapshots[-1].gpu_cache_usage_pct
        return 0.0

    def _mean_gpu_during(self, start_ts: float, end_ts: float) -> float:
        """Mean GPU cache usage during [start_ts, end_ts] from KV snapshots."""
        relevant = [s.gpu_cache_usage_pct for s in self._kv_snapshots
                     if start_ts <= s.timestamp <= end_ts]
        if relevant:
            return sum(relevant) / len(relevant)
        # Fallback: use latest snapshot before start, or overall latest
        return self._latest_gpu_usage()

    async def _poll_loop(self):
        while True:
            try:
                raw = await self.backend.scrape_metrics()
                snap = KVSnapshot(
                    timestamp=time.time(),
                    gpu_cache_usage_pct=raw.get("vllm:gpu_cache_usage_perc", 0),
                    cpu_cache_usage_pct=raw.get("vllm:cpu_cache_usage_perc", 0),
                    num_preemptions=raw.get("vllm:num_preemptions_total", 0),
                    num_running=raw.get("vllm:num_requests_running", 0),
                    num_waiting=raw.get("vllm:num_requests_waiting", 0),
                    prompt_throughput=raw.get(
                        "vllm:avg_prompt_throughput_toks_per_s", 0),
                    gen_throughput=raw.get(
                        "vllm:avg_generation_throughput_toks_per_s", 0),
                )
                self._kv_snapshots.append(snap)
                self._write_jsonl(self._kv_file, asdict(snap))
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Metrics scrape failed: {e}")
            await asyncio.sleep(self.poll_interval)

    def start(self):
        self._event_file = open(self._event_log_path, "w")
        self._kv_file = open(self._kv_log_path, "w")
        self._poll_task = asyncio.create_task(self._poll_loop())

    def stop(self):
        if self._poll_task:
            self._poll_task.cancel()
        if self._event_file and not self._event_file.closed:
            self._event_file.close()
        if self._kv_file and not self._kv_file.closed:
            self._kv_file.close()

    def write_summary(self) -> dict:
        summary = {
            "run_id": self.run_id,
            "num_sessions": len(self._sessions),
            "sessions": {},
            "kv_cache": {},
            "derived_metrics": {},
        }

        all_ttfts = []
        total_retries = 0
        all_retry_savings = []
        all_stall_waste = []
        all_branch_reuse = []

        for sid, ss in self._sessions.items():
            avg_ttft = (sum(ss.ttft_list) / len(ss.ttft_list)
                        if ss.ttft_list else None)
            summary["sessions"][sid] = {
                "generate_calls": ss.total_generate_calls,
                "prompt_tokens": ss.total_prompt_tokens,
                "gen_tokens": ss.total_gen_tokens,
                "retries": ss.retry_count,
                "tool_stall_sec": round(ss.tool_stall_sec, 3),
                "avg_ttft_sec": round(avg_ttft, 4) if avg_ttft else None,
                "wall_sec": round(ss.last_event_ts - ss.first_event_ts, 3),
            }
            all_ttfts.extend(ss.ttft_list)
            total_retries += ss.retry_count

            # Derived: retry prefill savings
            if ss.initial_ttfts and ss.retry_ttfts:
                initial_avg = sum(ss.initial_ttfts) / len(ss.initial_ttfts)
                retry_avg = sum(ss.retry_ttfts) / len(ss.retry_ttfts)
                if initial_avg > 0:
                    savings_ratio = (initial_avg - retry_avg) / initial_avg
                    all_retry_savings.append({
                        "session_id": sid,
                        "initial_ttft_avg": round(initial_avg, 4),
                        "retry_ttft_avg": round(retry_avg, 4),
                        "savings_ratio": round(savings_ratio, 4),
                        "savings_ms": round((initial_avg - retry_avg) * 1000, 2),
                    })

            # Derived: stall KV waste
            for sp in ss.stall_periods:
                all_stall_waste.append(sp)

            # Derived: branch prefix reuse
            if ss.branch_ttfts:
                branch_ids = sorted(ss.branch_ttfts.keys())
                if len(branch_ids) >= 2:
                    first_ttft = ss.branch_ttfts[branch_ids[0]]
                    for bid in branch_ids[1:]:
                        b_ttft = ss.branch_ttfts[bid]
                        reuse_hit = 1 if b_ttft < first_ttft * self.branch_reuse_threshold else 0
                        all_branch_reuse.append({
                            "session_id": sid,
                            "branch_id": bid,
                            "first_branch_ttft": round(first_ttft, 4),
                            "branch_ttft": round(b_ttft, 4),
                            "reuse_hit": reuse_hit,
                        })

        # KV cache summary from snapshots
        if self._kv_snapshots:
            gpu_usages = [s.gpu_cache_usage_pct for s in self._kv_snapshots]
            preemptions = [s.num_preemptions for s in self._kv_snapshots]
            summary["kv_cache"] = {
                "gpu_usage_pct_mean": round(sum(gpu_usages) / len(gpu_usages), 4),
                "gpu_usage_pct_max": round(max(gpu_usages), 4),
                "total_preemptions": int(max(preemptions)) if preemptions else 0,
                "num_snapshots": len(self._kv_snapshots),
            }

        # Global stats
        if all_ttfts:
            all_ttfts.sort()
            n = len(all_ttfts)
            summary["global"] = {
                "total_retries": total_retries,
                "ttft_p50": round(all_ttfts[max(0, (n - 1) // 2)], 4),
                "ttft_p95": round(all_ttfts[max(0, min(int(n * 0.95) - 1, n - 1))], 4),
                "ttft_mean": round(sum(all_ttfts) / n, 4),
            }

        # ── Throughput (workflows/min) ──
        all_wall_secs = [ss.last_event_ts - ss.first_event_ts
                         for ss in self._sessions.values()
                         if ss.last_event_ts > ss.first_event_ts]
        if all_wall_secs:
            total_wall = max(ss.last_event_ts for ss in self._sessions.values()) \
                       - min(ss.first_event_ts for ss in self._sessions.values())
            summary["throughput"] = {
                "completed_workflows": len(self._sessions),
                "total_wall_sec": round(total_wall, 3),
                "workflows_per_min": round(len(self._sessions) / total_wall * 60, 4)
                    if total_wall > 0 else None,
            }

        # ── Decode throughput (tokens/sec during generation) ──
        if self._kv_snapshots:
            gen_tps = [s.gen_throughput for s in self._kv_snapshots
                       if s.gen_throughput > 0]
            prompt_tps = [s.prompt_throughput for s in self._kv_snapshots
                          if s.prompt_throughput > 0]
            summary["decode_throughput"] = {
                "gen_tok_per_s_mean": round(sum(gen_tps) / len(gen_tps), 2)
                    if gen_tps else None,
                "gen_tok_per_s_max": round(max(gen_tps), 2)
                    if gen_tps else None,
                "gen_tok_per_s_min": round(min(gen_tps), 2)
                    if gen_tps else None,
                "prompt_tok_per_s_mean": round(sum(prompt_tps) / len(prompt_tps), 2)
                    if prompt_tps else None,
                "num_samples": len(gen_tps),
            }

        # ── Queue wait time (derived from num_waiting time-series) ──
        if self._kv_snapshots:
            waiting_counts = [s.num_waiting for s in self._kv_snapshots]
            nonzero_waiting = [w for w in waiting_counts if w > 0]
            # Estimate queue-seconds: integrate num_waiting over time
            queue_seconds = 0.0
            for i in range(1, len(self._kv_snapshots)):
                dt = self._kv_snapshots[i].timestamp - self._kv_snapshots[i - 1].timestamp
                queue_seconds += self._kv_snapshots[i - 1].num_waiting * dt
            summary["queue_wait"] = {
                "queue_seconds_total": round(queue_seconds, 3),
                "avg_waiting_requests": round(
                    sum(waiting_counts) / len(waiting_counts), 4)
                    if waiting_counts else 0,
                "max_waiting_requests": max(waiting_counts)
                    if waiting_counts else 0,
                "pct_time_with_queue": round(
                    len(nonzero_waiting) / len(waiting_counts), 4)
                    if waiting_counts else 0,
            }

        # Derived metrics
        summary["derived_metrics"] = {
            "retry_prefill_savings": all_retry_savings,
            "stall_periods": all_stall_waste,
            "branch_prefix_reuse": all_branch_reuse,
        }
        if all_retry_savings:
            avg_savings = sum(r["savings_ratio"] for r in all_retry_savings) / len(all_retry_savings)
            summary["derived_metrics"]["avg_retry_savings_ratio"] = round(avg_savings, 4)
        if all_branch_reuse:
            reuse_rate = sum(r["reuse_hit"] for r in all_branch_reuse) / len(all_branch_reuse)
            summary["derived_metrics"]["branch_reuse_rate"] = round(reuse_rate, 4)

        with open(self._summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        return summary

    @staticmethod
    def _write_jsonl(f, record):
        if f and not f.closed:
            f.write(json.dumps(record) + "\n")
            f.flush()
