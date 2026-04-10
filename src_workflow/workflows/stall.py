"""
StallWorkflow — Benchmark B.

Multiple concurrent sessions each do: generate → tool stall → generate → ...
Measures GPU KV occupancy waste during tool stalls.
"""
from __future__ import annotations

import asyncio
import random
import uuid
import time

from .base import BaseWorkflow, WorkflowResult, build_filler_prefix
from ..events import EventBus, EventType
from ..vllm_backend import VLLMBackend
from ..tools import sleep_tool


class StallWorkflow(BaseWorkflow):
    async def run(self) -> WorkflowResult:
        prefix_tokens = self.config.get("prefix_tokens", 4096)
        gen_tokens = self.config.get("gen_tokens", 128)
        stall_sec = self.config.get("stall_sec", 2.0)
        num_rounds = self.config.get("num_rounds", 3)
        stall_bucket = self.config.get("stall_bucket", "medium")

        filler = build_filler_prefix(prefix_tokens)
        messages = [{"role": "user", "content": filler}]
        t0 = time.time()
        ttfts = []

        for step in range(num_rounds):
            result = await self.do_step(messages, max_tokens=gen_tokens)
            ttfts.append(result.ttft_sec)
            messages.append({"role": "assistant", "content": result.text})

            # Tool stall
            self.emit(EventType.STALL_BEGIN, tool_name="sleep",
                      meta={"bucket": stall_bucket})
            await sleep_tool(stall_sec)
            self.emit(EventType.STALL_END, tool_name="sleep",
                      meta={"duration_sec": stall_sec})

            messages.append({"role": "user", "content": "Continue."})

        self.emit(EventType.END_SESSION)
        wall = time.time() - t0

        return WorkflowResult(
            session_id=self.session_id,
            wall_sec=wall,
            ttfts=ttfts,
            steps=num_rounds,
            extra={"stall_sec": stall_sec, "stall_bucket": stall_bucket},
        )


async def run_concurrency_benchmark(
    backend: VLLMBackend,
    event_bus: EventBus,
    config: dict,
) -> list[dict]:
    """Sweep concurrency × stall fraction for Benchmark B."""
    results = []
    ns_list = config.get("num_sessions", [4, 8])
    sf_list = config.get("stalled_fraction", [0.5])
    stall_durations = config.get("stall_durations_sec",
                                  {"short": 0.5, "medium": 2.0, "long": 8.0})
    prefix_tokens = config.get("prefix_tokens", 4096)
    gen_tokens = config.get("gen_tokens", 128)

    for ns in ns_list:
        for sf in sf_list:
            n_stalled = int(ns * sf)
            tasks = []
            session_ids = []

            for i in range(ns):
                sid = f"stall_{ns}_{sf}_{i}_{uuid.uuid4().hex[:4]}"
                session_ids.append(sid)

                if i < n_stalled:
                    stall = random.choice(
                        [stall_durations["medium"], stall_durations["long"]])
                    bucket = "long" if stall > 5 else "medium"
                else:
                    stall = stall_durations["short"]
                    bucket = "short"

                wf_config = {
                    "prefix_tokens": prefix_tokens,
                    "gen_tokens": gen_tokens,
                    "stall_sec": stall,
                    "stall_bucket": bucket,
                    "num_rounds": 2,
                }
                wf = StallWorkflow(sid, backend, event_bus, wf_config)
                tasks.append(wf.run())

            t0 = time.time()
            wf_results = await asyncio.gather(*tasks)
            wall = time.time() - t0

            results.append({
                "num_sessions": ns,
                "stalled_fraction": sf,
                "wall_sec": round(wall, 3),
                "throughput_sessions_per_min": round(ns / wall * 60, 2),
                "session_results": [
                    {"session_id": wr.session_id, "ttfts": wr.ttfts,
                     "wall_sec": wr.wall_sec}
                    for wr in wf_results
                ],
            })
    return results
