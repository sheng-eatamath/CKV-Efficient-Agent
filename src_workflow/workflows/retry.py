"""
RetryWorkflow — Benchmark A.

Workflow:
  1. Build long prefix
  2. CHECKPOINT
  3. Generate from prefix
  4. Tool call (may fail)
  5. On failure: RETRY_REENTRY → re-generate from same prefix
  6. Measure: does retry TTFT benefit from prefix caching?
"""
from __future__ import annotations

import uuid
import time

from .base import BaseWorkflow, WorkflowResult, build_filler_prefix
from ..events import EventType
from ..tools import flaky_tool


class RetryWorkflow(BaseWorkflow):
    async def run(self) -> WorkflowResult:
        prefix_tokens = self.config.get("prefix_tokens", 4096)
        num_retries = self.config.get("num_retries", 2)
        p_fail = self.config.get("p_fail", 0.5)
        gen_tokens = self.config.get("gen_tokens", 256)

        filler = build_filler_prefix(prefix_tokens)
        question = filler + "\nSummarize the above text in one sentence."
        messages = [{"role": "user", "content": question}]

        t0 = time.time()
        ttfts = []

        # Emit checkpoint at the prefix boundary
        self.emit(EventType.CHECKPOINT, meta={"token_position": prefix_tokens})

        for attempt in range(1 + num_retries):
            is_retry = attempt > 0

            if is_retry:
                self.emit(EventType.RETRY_REENTRY,
                          meta={"checkpoint_token_position": prefix_tokens,
                                "attempt": attempt})

            result = await self.do_step(messages, max_tokens=gen_tokens)
            ttfts.append(result.ttft_sec)

            # Simulate tool call
            self.emit(EventType.STALL_BEGIN, tool_name="flaky",
                      meta={"bucket": "short"})
            success, obs = await flaky_tool(p_fail=p_fail, latency_sec=0.1)
            self.emit(EventType.STALL_END, tool_name="flaky",
                      meta={"duration_sec": 0.1, "success": success})

            if success:
                break

        self.emit(EventType.END_SESSION)
        wall = time.time() - t0

        return WorkflowResult(
            session_id=self.session_id,
            wall_sec=wall,
            ttfts=ttfts,
            retries=max(0, attempt),
            steps=attempt + 1,
            extra={
                "prefix_tokens": prefix_tokens,
                "p_fail": p_fail,
                "initial_ttft": ttfts[0] if ttfts else None,
                "retry_ttfts": ttfts[1:] if len(ttfts) > 1 else [],
            },
        )


async def run_retry_benchmark(backend, event_bus, config: dict) -> list[dict]:
    """Run full parameter sweep for Benchmark A."""
    results = []
    prefix_list = config.get("prefix_tokens", [4096])
    retries_list = config.get("num_retries", [2])
    pfail_list = config.get("p_fail", [0.5])
    gen_tokens = config.get("gen_tokens", 256)

    for pt in prefix_list:
        for nr in retries_list:
            for pf in pfail_list:
                sid = f"retry_{pt}_{nr}_{pf}_{uuid.uuid4().hex[:6]}"
                wf_config = {
                    "prefix_tokens": pt,
                    "num_retries": nr,
                    "p_fail": pf,
                    "gen_tokens": gen_tokens,
                }
                wf = RetryWorkflow(sid, backend, event_bus, wf_config)
                wr = await wf.run()
                results.append({
                    "session_id": wr.session_id,
                    "prefix_tokens": pt,
                    "num_retries": nr,
                    "p_fail": pf,
                    "ttfts": wr.ttfts,
                    "retries": wr.retries,
                    "wall_sec": wr.wall_sec,
                    "extra": wr.extra,
                })
    return results
