"""
ReplayWorkflow — Benchmark D.

Replays real agent execution traces (from τ-bench or similar).
Extracts structural patterns (tool calls, retries, stalls) and
replays them through the harness with synthetic text of matching token length.
"""
from __future__ import annotations

import asyncio
import json
import uuid
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from .base import BaseWorkflow, WorkflowResult, build_filler_prefix
from ..events import EventBus, EventType
from ..vllm_backend import VLLMBackend


@dataclass
class ReplayTurn:
    role: str              # "system" | "user" | "assistant" | "tool"
    token_count: int
    tool_name: Optional[str] = None
    tool_latency_sec: float = 0.0
    tool_success: bool = True
    is_retry: bool = False


@dataclass
class ReplayPlan:
    trace_id: str
    domain: str            # "retail" | "airline"
    turns: List[ReplayTurn] = field(default_factory=list)
    total_tokens: int = 0
    num_tool_calls: int = 0
    num_retries: int = 0


def load_replay_plan(path: str) -> ReplayPlan:
    with open(path) as f:
        data = json.load(f)
    turns = [ReplayTurn(**t) for t in data.get("turns", [])]
    return ReplayPlan(
        trace_id=data.get("trace_id", Path(path).stem),
        domain=data.get("domain", "unknown"),
        turns=turns,
        total_tokens=data.get("total_tokens", sum(t.token_count for t in turns)),
        num_tool_calls=data.get("num_tool_calls", sum(1 for t in turns if t.tool_name)),
        num_retries=data.get("num_retries", sum(1 for t in turns if t.is_retry)),
    )


def classify_stall_bucket(duration_sec: float) -> str:
    if duration_sec < 1.0:
        return "short"
    elif duration_sec < 5.0:
        return "medium"
    return "long"


def select_replay_plans(
    trace_dir: str | Path,
    domains: Optional[List[str]] = None,
    num_traces_per_domain: Optional[int] = None,
) -> List[ReplayPlan]:
    trace_root = Path(trace_dir)
    if not trace_root.exists():
        return []

    requested_domains = list(domains) if domains else None
    plans_by_domain: dict[str, List[ReplayPlan]] = {}

    for path in sorted(trace_root.glob("*.json")):
        try:
            plan = load_replay_plan(str(path))
        except Exception:
            continue

        if requested_domains and plan.domain not in requested_domains:
            continue
        plans_by_domain.setdefault(plan.domain, []).append(plan)

    domain_order = requested_domains or sorted(plans_by_domain)
    selected: List[ReplayPlan] = []
    for domain in domain_order:
        domain_plans = sorted(plans_by_domain.get(domain, []), key=lambda item: item.trace_id)
        if num_traces_per_domain is not None:
            domain_plans = domain_plans[:num_traces_per_domain]
        selected.extend(domain_plans)

    return selected


def estimate_peak_prompt_tokens(plan: ReplayPlan) -> int:
    """Estimate the largest prompt prefix seen by any generation in the replay."""
    current_token_count = 0
    checkpoint_token_count = 0
    peak_prompt_tokens = 0

    for turn in plan.turns:
        if turn.role in ("system", "user"):
            current_token_count += turn.token_count
        elif turn.role == "assistant":
            peak_prompt_tokens = max(peak_prompt_tokens, current_token_count)
            if turn.tool_name:
                checkpoint_token_count = current_token_count
            current_token_count += turn.token_count
        elif turn.role == "tool":
            if turn.is_retry:
                current_token_count = checkpoint_token_count
            else:
                current_token_count += turn.token_count

    return peak_prompt_tokens


class ReplayWorkflow(BaseWorkflow):
    def __init__(self, session_id: str, backend: VLLMBackend,
                 event_bus: EventBus, config: dict,
                 replay_plan: ReplayPlan):
        super().__init__(session_id, backend, event_bus, config)
        self.replay_plan = replay_plan

    async def run(self) -> WorkflowResult:
        gen_tokens = self.config.get("gen_tokens", 128)
        t0 = time.time()
        ttfts = []
        messages = []
        current_token_count = 0
        last_checkpoint_pos = 0
        checkpoint_messages: list[dict] = []
        checkpoint_token_count = 0
        retries = 0

        for turn in self.replay_plan.turns:
            filler = build_filler_prefix(turn.token_count)

            if turn.role in ("system", "user"):
                messages.append({"role": turn.role, "content": filler})
                current_token_count += turn.token_count

            elif turn.role == "assistant":
                if turn.tool_name:
                    # Checkpoint before tool call
                    checkpoint_messages = list(messages)
                    checkpoint_token_count = current_token_count
                    self.emit(EventType.CHECKPOINT,
                              meta={"token_position": checkpoint_token_count})
                    last_checkpoint_pos = checkpoint_token_count

                    result = await self.do_step(messages, max_tokens=gen_tokens)
                    ttfts.append(result.ttft_sec)
                    messages.append({"role": "assistant", "content": filler})
                    current_token_count += turn.token_count

                    # Tool stall
                    bucket = classify_stall_bucket(turn.tool_latency_sec)
                    self.emit(EventType.STALL_BEGIN, tool_name=turn.tool_name,
                              meta={"bucket": bucket})
                    await asyncio.sleep(turn.tool_latency_sec)
                    self.emit(EventType.STALL_END, tool_name=turn.tool_name,
                              meta={"duration_sec": turn.tool_latency_sec})
                else:
                    result = await self.do_step(messages, max_tokens=gen_tokens)
                    ttfts.append(result.ttft_sec)
                    messages.append({"role": "assistant", "content": filler})
                    current_token_count += turn.token_count

            elif turn.role == "tool":
                if turn.is_retry:
                    retries += 1
                    messages = list(checkpoint_messages)
                    current_token_count = checkpoint_token_count
                    self.emit(EventType.RETRY_REENTRY,
                              meta={"checkpoint_token_position": last_checkpoint_pos})
                else:
                    messages.append(
                        {"role": "user", "content": f"Tool result: {filler}"}
                    )
                    current_token_count += turn.token_count

        self.emit(EventType.END_SESSION)
        wall = time.time() - t0

        return WorkflowResult(
            session_id=self.session_id,
            wall_sec=wall,
            ttfts=ttfts,
            retries=retries,
            steps=len(ttfts),
            extra={
                "trace_id": self.replay_plan.trace_id,
                "domain": self.replay_plan.domain,
                "total_tokens": self.replay_plan.total_tokens,
                "num_tool_calls": self.replay_plan.num_tool_calls,
            },
        )


async def run_replay_benchmark(
    backend: VLLMBackend,
    event_bus: EventBus,
    config: dict,
) -> list[dict]:
    """Sweep traces at various concurrency levels for Benchmark D."""
    trace_dir = Path(config.get("trace_dir", "traces"))
    domains = config.get("domains")
    num_traces_per_domain = config.get("num_traces_per_domain")
    concurrency_list = config.get("concurrency", [1, 4])
    gen_tokens = config.get("gen_tokens", 128)
    max_context_tokens = config.get("max_context_tokens")

    selected_plans = select_replay_plans(
        trace_dir,
        domains=domains,
        num_traces_per_domain=num_traces_per_domain,
    )

    skipped_trace_ids: list[str] = []
    plans = selected_plans
    if max_context_tokens is not None:
        safe_plans = []
        for plan in selected_plans:
            peak_prompt_tokens = estimate_peak_prompt_tokens(plan)
            if peak_prompt_tokens + gen_tokens <= max_context_tokens:
                safe_plans.append(plan)
            else:
                skipped_trace_ids.append(plan.trace_id)
        plans = safe_plans

    if not plans:
        return [{
            "error": "no trace files found",
            "trace_dir": str(trace_dir),
            "domains": domains,
            "num_traces_per_domain": num_traces_per_domain,
            "skipped_trace_ids": skipped_trace_ids,
        }]

    results = []
    for conc in concurrency_list:
        sem = asyncio.Semaphore(conc)
        wf_results = []

        async def _run_one(plan: ReplayPlan):
            async with sem:
                sid = f"replay_{plan.trace_id}_{uuid.uuid4().hex[:4]}"
                wf = ReplayWorkflow(sid, backend, event_bus,
                                    {"gen_tokens": gen_tokens}, plan)
                return await wf.run()

        t0 = time.time()
        batch_results = await asyncio.gather(
            *[_run_one(p) for p in plans]
        )
        wall = time.time() - t0

        results.append({
            "concurrency": conc,
            "num_traces": len(plans),
            "num_skipped_traces": len(skipped_trace_ids),
            "skipped_trace_ids": skipped_trace_ids,
            "wall_sec": round(wall, 3),
            "throughput_traces_per_min": round(len(plans) / wall * 60, 2),
            "trace_results": [
                {"session_id": wr.session_id, "trace_id": wr.extra.get("trace_id"),
                 "ttfts": wr.ttfts, "retries": wr.retries, "wall_sec": wr.wall_sec}
                for wr in batch_results
            ],
        })
    return results
