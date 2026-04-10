"""
Unit tests for CKV-Agent workflow system.

Tests cover:
  1. EventBus and WorkflowEvent
  2. Tools (sleep, flaky, stubs)
  3. Profiler (event logging, session tracking, derived metrics)
  4. Workflow classes (RetryWorkflow, StallWorkflow, BranchWorkflow, ReplayWorkflow)
  5. Trace extraction

Runs WITHOUT a vLLM server — uses a mock backend.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

import pytest

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src_workflow.events import EventBus, WorkflowEvent, EventType
from src_workflow.vllm_backend import VLLMBackend, GenerateResult, _parse_prometheus
from src_workflow.agent_step import run_step, StepResult
from src_workflow.tools import sleep_tool, flaky_tool, search_stub, lookup_stub, TOOL_REGISTRY
from src_workflow.profiler import KVProfiler, KVSnapshot, SessionStats
from src_workflow.workflows.base import BaseWorkflow, WorkflowResult, build_filler_prefix
from src_workflow.workflows.retry import RetryWorkflow, run_retry_benchmark
from src_workflow.workflows.stall import StallWorkflow, run_concurrency_benchmark
from src_workflow.workflows.branch import BranchWorkflow, run_branch_benchmark
from src_workflow.workflows.replay import (
    ReplayWorkflow, ReplayPlan, ReplayTurn, load_replay_plan,
    classify_stall_bucket, estimate_peak_prompt_tokens, run_replay_benchmark,
    select_replay_plans,
)


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def event_bus():
    return EventBus()


@pytest.fixture
def mock_backend():
    """Mock VLLMBackend that returns dummy results without a real server."""
    backend = MagicMock(spec=VLLMBackend)

    async def mock_generate(messages, max_tokens=256, temperature=0.0):
        return GenerateResult(
            text="This is a mock response from the LLM.",
            prompt_tokens=len(str(messages)) // 4,
            completion_tokens=max_tokens,
            ttft_sec=0.05 + len(str(messages)) / 100000,  # simulate longer prefix = longer TTFT
            total_sec=0.1 + max_tokens * 0.002,
        )

    async def mock_scrape():
        return {
            "vllm:gpu_cache_usage_perc": 0.45,
            "vllm:cpu_cache_usage_perc": 0.10,
            "vllm:num_preemptions_total": 0,
            "vllm:num_requests_running": 1,
            "vllm:num_requests_waiting": 0,
            "vllm:avg_prompt_throughput_toks_per_s": 1000,
            "vllm:avg_generation_throughput_toks_per_s": 50,
        }

    backend.generate = AsyncMock(side_effect=mock_generate)
    backend.scrape_metrics = AsyncMock(side_effect=mock_scrape)
    return backend


@pytest.fixture
def tmp_output_dir():
    with tempfile.TemporaryDirectory() as d:
        yield d


# ============================================================
# 1. EventBus Tests
# ============================================================

class TestEventBus:
    def test_subscribe_and_emit(self, event_bus):
        received = []
        event_bus.subscribe(lambda ev: received.append(ev))

        ev = WorkflowEvent(
            event_type=EventType.BEFORE_GENERATE,
            session_id="test_1",
            step_id=0,
        )
        event_bus.emit(ev)

        assert len(received) == 1
        assert received[0].event_type == EventType.BEFORE_GENERATE
        assert received[0].session_id == "test_1"

    def test_multiple_subscribers(self, event_bus):
        r1, r2 = [], []
        event_bus.subscribe(lambda ev: r1.append(ev))
        event_bus.subscribe(lambda ev: r2.append(ev))

        event_bus.emit(WorkflowEvent(
            event_type=EventType.CHECKPOINT,
            session_id="s1", step_id=0,
        ))

        assert len(r1) == 1
        assert len(r2) == 1

    def test_all_event_types(self, event_bus):
        received = []
        event_bus.subscribe(lambda ev: received.append(ev.event_type))

        for et in EventType:
            event_bus.emit(WorkflowEvent(
                event_type=et, session_id="s", step_id=0,
            ))

        assert len(received) == len(EventType)
        for et in EventType:
            assert et in received

    def test_event_meta(self, event_bus):
        received = []
        event_bus.subscribe(lambda ev: received.append(ev))

        event_bus.emit(WorkflowEvent(
            event_type=EventType.CHECKPOINT,
            session_id="s1", step_id=0,
            meta={"token_position": 4096, "label": "before_tool"},
        ))

        assert received[0].meta["token_position"] == 4096
        assert received[0].meta["label"] == "before_tool"


# ============================================================
# 2. Tools Tests
# ============================================================

class TestTools:
    @pytest.mark.asyncio
    async def test_sleep_tool(self):
        t0 = time.time()
        success, msg = await sleep_tool(0.1)
        elapsed = time.time() - t0
        assert success is True
        assert elapsed >= 0.09
        assert "slept" in msg

    @pytest.mark.asyncio
    async def test_flaky_tool_always_fail(self):
        success, msg = await flaky_tool(p_fail=1.0, latency_sec=0.01)
        assert success is False
        assert "error" in msg

    @pytest.mark.asyncio
    async def test_flaky_tool_always_succeed(self):
        success, msg = await flaky_tool(p_fail=0.0, latency_sec=0.01)
        assert success is True
        assert "success" in msg

    @pytest.mark.asyncio
    async def test_search_stub(self):
        success, msg = await search_stub("test query", latency_sec=0.01)
        assert success is True
        assert "test query" in msg

    @pytest.mark.asyncio
    async def test_lookup_stub(self):
        success, msg = await lookup_stub("test title", latency_sec=0.01)
        assert success is True
        assert "test title" in msg

    def test_tool_registry(self):
        assert "sleep" in TOOL_REGISTRY
        assert "flaky" in TOOL_REGISTRY
        assert "search" in TOOL_REGISTRY
        assert "lookup" in TOOL_REGISTRY


# ============================================================
# 3. VLLMBackend Prometheus Parsing Test
# ============================================================

class TestPrometheus:
    def test_parse_prometheus(self):
        text = """# HELP vllm:gpu_cache_usage_perc GPU cache usage
# TYPE vllm:gpu_cache_usage_perc gauge
vllm:gpu_cache_usage_perc 0.85
vllm:cpu_cache_usage_perc 0.12
vllm:num_preemptions_total 3
vllm:num_requests_running 5
vllm:num_requests_waiting 2
"""
        result = _parse_prometheus(text)
        assert result["vllm:gpu_cache_usage_perc"] == 0.85
        assert result["vllm:cpu_cache_usage_perc"] == 0.12
        assert result["vllm:num_preemptions_total"] == 3.0
        assert result["vllm:num_requests_running"] == 5.0

    def test_parse_prometheus_with_labels(self):
        text = 'vllm:request_e2e_time_seconds_bucket{le="0.5"} 10\n'
        result = _parse_prometheus(text)
        assert "vllm:request_e2e_time_seconds_bucket" in result

    def test_parse_empty(self):
        result = _parse_prometheus("")
        assert result == {}


# ============================================================
# 4. Agent Step Tests
# ============================================================

class TestAgentStep:
    @pytest.mark.asyncio
    async def test_run_step(self, event_bus, mock_backend):
        received = []
        event_bus.subscribe(lambda ev: received.append(ev))

        result = await run_step(
            session_id="test_step",
            step_id=0,
            messages=[{"role": "user", "content": "hello"}],
            backend=mock_backend,
            event_bus=event_bus,
            max_tokens=64,
        )

        assert isinstance(result, StepResult)
        assert result.text == "This is a mock response from the LLM."
        assert result.ttft_sec > 0

        # Check events emitted
        types = [ev.event_type for ev in received]
        assert EventType.BEFORE_GENERATE in types
        assert EventType.AFTER_GENERATE in types


# ============================================================
# 5. Profiler Tests
# ============================================================

class TestProfiler:
    @pytest.mark.asyncio
    async def test_profiler_event_logging(self, event_bus, mock_backend, tmp_output_dir):
        profiler = KVProfiler(event_bus, mock_backend, run_id="test_prof",
                              output_dir=tmp_output_dir, poll_interval=10.0)
        profiler.start()

        # Emit some events
        event_bus.emit(WorkflowEvent(
            event_type=EventType.CHECKPOINT,
            session_id="s1", step_id=0,
            meta={"token_position": 4096},
        ))
        event_bus.emit(WorkflowEvent(
            event_type=EventType.AFTER_GENERATE,
            session_id="s1", step_id=1,
            prompt_tokens=4096, generated_tokens=100,
            meta={"ttft_sec": 0.5},
        ))
        event_bus.emit(WorkflowEvent(
            event_type=EventType.RETRY_REENTRY,
            session_id="s1", step_id=2,
            retry_reason="tool_fail",
        ))
        event_bus.emit(WorkflowEvent(
            event_type=EventType.AFTER_GENERATE,
            session_id="s1", step_id=3,
            prompt_tokens=4096, generated_tokens=100,
            meta={"ttft_sec": 0.1},
        ))
        event_bus.emit(WorkflowEvent(
            event_type=EventType.END_SESSION,
            session_id="s1", step_id=4,
        ))

        profiler.stop()
        summary = profiler.write_summary()

        assert summary["num_sessions"] == 1
        assert "s1" in summary["sessions"]
        assert summary["sessions"]["s1"]["retries"] == 1

        # Check derived metrics
        savings = summary["derived_metrics"]["retry_prefill_savings"]
        assert len(savings) == 1
        assert savings[0]["savings_ratio"] > 0  # 0.5 -> 0.1, big improvement

    @pytest.mark.asyncio
    async def test_profiler_branch_tracking(self, event_bus, mock_backend, tmp_output_dir):
        profiler = KVProfiler(event_bus, mock_backend, run_id="test_branch",
                              output_dir=tmp_output_dir, poll_interval=10.0)
        profiler.start()

        # Simulate branch workflow
        event_bus.emit(WorkflowEvent(
            event_type=EventType.CHECKPOINT,
            session_id="b1", step_id=0,
            meta={"token_position": 4096},
        ))

        for branch_id in range(3):
            event_bus.emit(WorkflowEvent(
                event_type=EventType.BRANCH_START,
                session_id="b1", step_id=branch_id + 1,
                meta={"branch_id": branch_id},
            ))
            event_bus.emit(WorkflowEvent(
                event_type=EventType.AFTER_GENERATE,
                session_id="b1", step_id=branch_id + 1,
                meta={"ttft_sec": 0.5 - branch_id * 0.15},
            ))
            event_bus.emit(WorkflowEvent(
                event_type=EventType.BRANCH_END,
                session_id="b1", step_id=branch_id + 1,
                meta={"branch_id": branch_id},
            ))

        profiler.stop()
        summary = profiler.write_summary()

        reuse = summary["derived_metrics"]["branch_prefix_reuse"]
        assert len(reuse) > 0

    @pytest.mark.asyncio
    async def test_profiler_output_files(self, event_bus, mock_backend, tmp_output_dir):
        profiler = KVProfiler(event_bus, mock_backend, run_id="test_files",
                              output_dir=tmp_output_dir, poll_interval=10.0)
        profiler.start()
        event_bus.emit(WorkflowEvent(
            event_type=EventType.END_SESSION,
            session_id="s1", step_id=0,
        ))
        profiler.stop()
        profiler.write_summary()

        assert (Path(tmp_output_dir) / "test_files_events.jsonl").exists()
        assert (Path(tmp_output_dir) / "test_files_kv_timeseries.jsonl").exists()
        assert (Path(tmp_output_dir) / "test_files_summary.json").exists()


# ============================================================
# 6. Workflow Tests (with mock backend)
# ============================================================

class TestRetryWorkflow:
    @pytest.mark.asyncio
    async def test_retry_workflow_basic(self, event_bus, mock_backend):
        events = []
        event_bus.subscribe(lambda ev: events.append(ev))

        config = {"prefix_tokens": 1024, "num_retries": 2,
                  "p_fail": 1.0, "gen_tokens": 64}
        wf = RetryWorkflow("retry_test", mock_backend, event_bus, config)
        result = await wf.run()

        assert isinstance(result, WorkflowResult)
        assert result.session_id == "retry_test"
        assert len(result.ttfts) == 3  # initial + 2 retries (all fail at p=1.0)

        # Check structural events
        types = [ev.event_type for ev in events]
        assert EventType.CHECKPOINT in types
        assert EventType.RETRY_REENTRY in types
        assert EventType.STALL_BEGIN in types
        assert EventType.STALL_END in types
        assert EventType.END_SESSION in types

    @pytest.mark.asyncio
    async def test_retry_workflow_success_first(self, event_bus, mock_backend):
        config = {"prefix_tokens": 1024, "num_retries": 3,
                  "p_fail": 0.0, "gen_tokens": 64}
        wf = RetryWorkflow("retry_success", mock_backend, event_bus, config)
        result = await wf.run()

        # Should succeed on first try
        assert len(result.ttfts) == 1
        assert result.retries == 0


class TestStallWorkflow:
    @pytest.mark.asyncio
    async def test_stall_workflow(self, event_bus, mock_backend):
        events = []
        event_bus.subscribe(lambda ev: events.append(ev))

        config = {"prefix_tokens": 512, "gen_tokens": 32,
                  "stall_sec": 0.05, "num_rounds": 2, "stall_bucket": "short"}
        wf = StallWorkflow("stall_test", mock_backend, event_bus, config)
        result = await wf.run()

        assert len(result.ttfts) == 2

        types = [ev.event_type for ev in events]
        assert types.count(EventType.STALL_BEGIN) == 2
        assert types.count(EventType.STALL_END) == 2
        assert EventType.END_SESSION in types


class TestBranchWorkflow:
    @pytest.mark.asyncio
    async def test_branch_workflow(self, event_bus, mock_backend):
        events = []
        event_bus.subscribe(lambda ev: events.append(ev))

        config = {"shared_prefix_tokens": 1024, "branch_factor": 3,
                  "suffix_tokens": 64, "gen_tokens": 32}
        wf = BranchWorkflow("branch_test", mock_backend, event_bus, config)
        result = await wf.run()

        assert len(result.ttfts) == 3

        types = [ev.event_type for ev in events]
        assert EventType.CHECKPOINT in types
        assert types.count(EventType.BRANCH_START) == 3
        assert types.count(EventType.BRANCH_END) == 3
        assert EventType.END_SESSION in types


class TestReplayWorkflow:
    @pytest.mark.asyncio
    async def test_replay_workflow(self, event_bus, mock_backend):
        message_lengths = []
        original_generate = mock_backend.generate.side_effect

        async def record_generate(messages, max_tokens=256, temperature=0.0):
            message_lengths.append(len(messages))
            return await original_generate(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        mock_backend.generate = AsyncMock(side_effect=record_generate)

        plan = ReplayPlan(
            trace_id="test_trace",
            domain="retail",
            turns=[
                ReplayTurn(role="system", token_count=500),
                ReplayTurn(role="user", token_count=100),
                ReplayTurn(role="assistant", token_count=100,
                           tool_name="get_order", tool_latency_sec=0.05),
                ReplayTurn(role="tool", token_count=200, tool_success=True),
                ReplayTurn(role="assistant", token_count=100),
            ],
            total_tokens=1000,
            num_tool_calls=1,
            num_retries=0,
        )

        events = []
        event_bus.subscribe(lambda ev: events.append(ev))

        config = {"gen_tokens": 32}
        wf = ReplayWorkflow("replay_test", mock_backend, event_bus, config, plan)
        result = await wf.run()

        assert result.extra["trace_id"] == "test_trace"
        assert len(result.ttfts) >= 1
        assert message_lengths == [2, 4]

        types = [ev.event_type for ev in events]
        assert EventType.CHECKPOINT in types
        assert EventType.STALL_BEGIN in types
        assert EventType.END_SESSION in types

    @pytest.mark.asyncio
    async def test_replay_with_retry(self, event_bus, mock_backend):
        message_lengths = []
        original_generate = mock_backend.generate.side_effect

        async def record_generate(messages, max_tokens=256, temperature=0.0):
            message_lengths.append(len(messages))
            return await original_generate(
                messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )

        mock_backend.generate = AsyncMock(side_effect=record_generate)

        plan = ReplayPlan(
            trace_id="retry_trace",
            domain="airline",
            turns=[
                ReplayTurn(role="system", token_count=500),
                ReplayTurn(role="user", token_count=100),
                ReplayTurn(role="assistant", token_count=100,
                           tool_name="update_booking", tool_latency_sec=0.05),
                ReplayTurn(role="tool", token_count=100, tool_success=False, is_retry=True),
                ReplayTurn(role="assistant", token_count=100),
            ],
        )

        events = []
        event_bus.subscribe(lambda ev: events.append(ev))

        wf = ReplayWorkflow("replay_retry", mock_backend, event_bus,
                            {"gen_tokens": 32}, plan)
        result = await wf.run()

        assert result.retries == 1
        assert message_lengths == [2, 2]
        types = [ev.event_type for ev in events]
        assert EventType.RETRY_REENTRY in types


# ============================================================
# 7. Trace Extraction Tests
# ============================================================

class TestTraceExtraction:
    def test_classify_stall_bucket(self):
        assert classify_stall_bucket(0.3) == "short"
        assert classify_stall_bucket(2.5) == "medium"
        assert classify_stall_bucket(10.0) == "long"

    def test_load_replay_plan(self, tmp_path):
        plan_data = {
            "trace_id": "test_001",
            "domain": "retail",
            "turns": [
                {"role": "system", "token_count": 500},
                {"role": "user", "token_count": 100},
            ],
            "total_tokens": 600,
            "num_tool_calls": 0,
            "num_retries": 0,
        }
        path = tmp_path / "test_001.json"
        with open(path, "w") as f:
            json.dump(plan_data, f)

        plan = load_replay_plan(str(path))
        assert plan.trace_id == "test_001"
        assert plan.domain == "retail"
        assert len(plan.turns) == 2
        assert plan.total_tokens == 600

    def test_generate_synthetic_traces(self, tmp_path):
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "extract_traces", PROJECT_ROOT / "scripts" / "extract_traces.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.generate_synthetic_traces(str(tmp_path), num_per_domain=5)
        files = list(tmp_path.glob("*.json"))
        assert len(files) == 10  # 5 retail + 5 airline

    def test_select_replay_plans_respects_domain_and_limit(self, tmp_path):
        for domain in ("retail", "airline"):
            for idx in range(3):
                plan_data = {
                    "trace_id": f"{domain}_{idx:03d}",
                    "domain": domain,
                    "turns": [{"role": "user", "token_count": 100}],
                }
                path = tmp_path / f"{domain}_{idx:03d}.json"
                with open(path, "w") as f:
                    json.dump(plan_data, f)

        selected = select_replay_plans(
            tmp_path,
            domains=["retail"],
            num_traces_per_domain=2,
        )

        assert [plan.trace_id for plan in selected] == ["retail_000", "retail_001"]

    def test_estimate_peak_prompt_tokens_retry_resets_to_checkpoint(self):
        plan = ReplayPlan(
            trace_id="retry_peak",
            domain="retail",
            turns=[
                ReplayTurn(role="system", token_count=1000),
                ReplayTurn(role="user", token_count=500),
                ReplayTurn(role="assistant", token_count=200, tool_name="lookup"),
                ReplayTurn(role="tool", token_count=300, tool_success=False, is_retry=True),
                ReplayTurn(role="assistant", token_count=150),
            ],
        )

        assert estimate_peak_prompt_tokens(plan) == 1500


# ============================================================
# 8. Build Filler Prefix Test
# ============================================================

class TestBuildFiller:
    def test_filler_length(self):
        text = build_filler_prefix(1024)
        # Rough check: ~12 tokens per sentence repeat
        assert len(text) > 100

    def test_filler_deterministic(self):
        a = build_filler_prefix(2048)
        b = build_filler_prefix(2048)
        assert a == b


# ============================================================
# 9. Integration: Profiler + Workflow
# ============================================================

class TestIntegration:
    @pytest.mark.asyncio
    async def test_retry_with_profiler(self, event_bus, mock_backend, tmp_output_dir):
        profiler = KVProfiler(event_bus, mock_backend, run_id="int_retry",
                              output_dir=tmp_output_dir, poll_interval=10.0)
        profiler.start()

        config = {"prefix_tokens": 2048, "num_retries": 2,
                  "p_fail": 1.0, "gen_tokens": 64}
        wf = RetryWorkflow("int_retry_1", mock_backend, event_bus, config)
        await wf.run()

        profiler.stop()
        summary = profiler.write_summary()

        assert summary["num_sessions"] == 1
        assert summary["sessions"]["int_retry_1"]["retries"] == 2
        assert summary["sessions"]["int_retry_1"]["generate_calls"] == 3

    @pytest.mark.asyncio
    async def test_branch_with_profiler(self, event_bus, mock_backend, tmp_output_dir):
        profiler = KVProfiler(event_bus, mock_backend, run_id="int_branch",
                              output_dir=tmp_output_dir, poll_interval=10.0)
        profiler.start()

        config = {"shared_prefix_tokens": 2048, "branch_factor": 4,
                  "suffix_tokens": 128, "gen_tokens": 32}
        wf = BranchWorkflow("int_branch_1", mock_backend, event_bus, config)
        await wf.run()

        profiler.stop()
        summary = profiler.write_summary()

        assert summary["num_sessions"] == 1
        assert summary["sessions"]["int_branch_1"]["generate_calls"] == 4

    @pytest.mark.asyncio
    async def test_concurrency_with_profiler(self, event_bus, mock_backend, tmp_output_dir):
        profiler = KVProfiler(event_bus, mock_backend, run_id="int_conc",
                              output_dir=tmp_output_dir, poll_interval=10.0)
        profiler.start()

        config = {
            "num_sessions": [2],
            "stalled_fraction": [0.5],
            "stall_durations_sec": {"short": 0.01, "medium": 0.05, "long": 0.1},
            "prefix_tokens": 512,
            "gen_tokens": 32,
        }
        results = await run_concurrency_benchmark(mock_backend, event_bus, config)

        profiler.stop()
        summary = profiler.write_summary()

        assert len(results) == 1
        assert results[0]["num_sessions"] == 2
        assert summary["num_sessions"] == 2


# ============================================================
# 10. Benchmark Sweep Tests (mock, fast)
# ============================================================

class TestBenchmarkSweeps:
    @pytest.mark.asyncio
    async def test_retry_sweep(self, event_bus, mock_backend):
        config = {
            "prefix_tokens": [512],
            "num_retries": [1],
            "p_fail": [1.0],
            "gen_tokens": 32,
        }
        results = await run_retry_benchmark(mock_backend, event_bus, config)
        assert len(results) == 1
        assert results[0]["prefix_tokens"] == 512
        assert len(results[0]["ttfts"]) == 2  # initial + 1 retry

    @pytest.mark.asyncio
    async def test_branch_sweep(self, event_bus, mock_backend):
        config = {
            "branch_factor": [2],
            "shared_prefix_tokens": [1024],
            "suffix_tokens": [64],
            "gen_tokens": 32,
        }
        results = await run_branch_benchmark(mock_backend, event_bus, config)
        assert len(results) == 1
        assert len(results[0]["branch_ttfts"]) == 2


# ============================================================
# 11. New Profiler Metrics Tests (throughput, decode, queue)
# ============================================================

class TestProfilerNewMetrics:
    @pytest.mark.asyncio
    async def test_throughput_metric(self, event_bus, mock_backend, tmp_output_dir):
        """Throughput section appears in summary with workflows_per_min."""
        profiler = KVProfiler(event_bus, mock_backend, run_id="test_tp",
                              output_dir=tmp_output_dir, poll_interval=10.0)
        profiler.start()

        for i in range(3):
            sid = f"tp_{i}"
            event_bus.emit(WorkflowEvent(
                event_type=EventType.AFTER_GENERATE,
                session_id=sid, step_id=0,
                prompt_tokens=100, generated_tokens=50,
                meta={"ttft_sec": 0.1, "total_sec": 0.5},
            ))
            event_bus.emit(WorkflowEvent(
                event_type=EventType.END_SESSION,
                session_id=sid, step_id=1,
            ))

        profiler.stop()
        summary = profiler.write_summary()

        assert "throughput" in summary
        assert summary["throughput"]["completed_workflows"] == 3
        assert summary["throughput"]["workflows_per_min"] is not None
        assert summary["throughput"]["workflows_per_min"] > 0

    @pytest.mark.asyncio
    async def test_decode_throughput_metric(self, event_bus, mock_backend, tmp_output_dir):
        """Decode throughput section summarizes gen_throughput from KV snapshots."""
        profiler = KVProfiler(event_bus, mock_backend, run_id="test_dec",
                              output_dir=tmp_output_dir, poll_interval=10.0)
        profiler.start()

        # Inject KV snapshots with known gen_throughput values
        now = time.time()
        for i, gt in enumerate([40.0, 50.0, 60.0]):
            profiler._kv_snapshots.append(KVSnapshot(
                timestamp=now + i,
                gpu_cache_usage_pct=0.5,
                cpu_cache_usage_pct=0.1,
                num_preemptions=0,
                num_running=1,
                num_waiting=0,
                prompt_throughput=1000.0,
                gen_throughput=gt,
            ))

        event_bus.emit(WorkflowEvent(
            event_type=EventType.AFTER_GENERATE,
            session_id="dec_s", step_id=0,
            meta={"ttft_sec": 0.1, "total_sec": 0.5},
        ))

        profiler.stop()
        summary = profiler.write_summary()

        assert "decode_throughput" in summary
        dt = summary["decode_throughput"]
        assert dt["gen_tok_per_s_mean"] == 50.0
        assert dt["gen_tok_per_s_max"] == 60.0
        assert dt["gen_tok_per_s_min"] == 40.0
        assert dt["prompt_tok_per_s_mean"] == 1000.0
        assert dt["num_samples"] == 3

    @pytest.mark.asyncio
    async def test_queue_wait_metric(self, event_bus, mock_backend, tmp_output_dir):
        """Queue wait section computes queue-seconds and waiting stats."""
        profiler = KVProfiler(event_bus, mock_backend, run_id="test_qw",
                              output_dir=tmp_output_dir, poll_interval=10.0)
        profiler.start()

        now = time.time()
        # 3 snapshots: 1s apart, num_waiting = [0, 2, 3]
        for i, nw in enumerate([0, 2, 3]):
            profiler._kv_snapshots.append(KVSnapshot(
                timestamp=now + i,
                gpu_cache_usage_pct=0.5,
                cpu_cache_usage_pct=0.1,
                num_preemptions=0,
                num_running=1,
                num_waiting=nw,
                prompt_throughput=1000.0,
                gen_throughput=50.0,
            ))

        event_bus.emit(WorkflowEvent(
            event_type=EventType.AFTER_GENERATE,
            session_id="qw_s", step_id=0,
            meta={"ttft_sec": 0.1, "total_sec": 0.5},
        ))

        profiler.stop()
        summary = profiler.write_summary()

        assert "queue_wait" in summary
        qw = summary["queue_wait"]
        # queue_seconds = 0*1 + 2*1 = 2.0
        assert abs(qw["queue_seconds_total"] - 2.0) < 0.1
        assert qw["max_waiting_requests"] == 3
        # 2 out of 3 have nonzero waiting
        assert abs(qw["pct_time_with_queue"] - 2 / 3) < 0.01

    @pytest.mark.asyncio
    async def test_generate_durations_tracked(self, event_bus, mock_backend, tmp_output_dir):
        """generate_durations list is populated from total_sec metadata."""
        profiler = KVProfiler(event_bus, mock_backend, run_id="test_gd",
                              output_dir=tmp_output_dir, poll_interval=10.0)
        profiler.start()

        event_bus.emit(WorkflowEvent(
            event_type=EventType.AFTER_GENERATE,
            session_id="gd_s", step_id=0,
            prompt_tokens=100, generated_tokens=50,
            meta={"ttft_sec": 0.05, "total_sec": 0.42},
        ))
        event_bus.emit(WorkflowEvent(
            event_type=EventType.AFTER_GENERATE,
            session_id="gd_s", step_id=1,
            prompt_tokens=200, generated_tokens=60,
            meta={"ttft_sec": 0.08, "total_sec": 0.55},
        ))

        profiler.stop()
        ss = profiler._sessions["gd_s"]
        assert ss.generate_durations == [0.42, 0.55]

    @pytest.mark.asyncio
    async def test_no_snapshots_no_crash(self, event_bus, mock_backend, tmp_output_dir):
        """Summary generates cleanly when there are no KV snapshots."""
        profiler = KVProfiler(event_bus, mock_backend, run_id="test_empty",
                              output_dir=tmp_output_dir, poll_interval=10.0)
        profiler.start()

        event_bus.emit(WorkflowEvent(
            event_type=EventType.AFTER_GENERATE,
            session_id="e_s", step_id=0,
            meta={"ttft_sec": 0.1, "total_sec": 0.5},
        ))

        profiler.stop()
        summary = profiler.write_summary()

        # These sections should not exist or be empty when no snapshots
        assert summary.get("decode_throughput") is None or summary["decode_throughput"] == {}
        assert summary.get("queue_wait") is None or summary["queue_wait"] == {}
