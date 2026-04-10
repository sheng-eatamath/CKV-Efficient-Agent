"""
BranchWorkflow — Benchmark C.

Workflow:
  1. Build shared prefix
  2. CHECKPOINT at prefix boundary
  3. For each branch k: BRANCH_START → generate with unique suffix → BRANCH_END
  4. Measure: do later branches benefit from shared prefix cache?
"""
from __future__ import annotations

import uuid
import time

from .base import BaseWorkflow, WorkflowResult, build_filler_prefix
from ..events import EventBus, EventType
from ..vllm_backend import VLLMBackend


class BranchWorkflow(BaseWorkflow):
    async def run(self) -> WorkflowResult:
        shared_prefix_tokens = self.config.get("shared_prefix_tokens", 4096)
        branch_factor = self.config.get("branch_factor", 4)
        suffix_tokens = self.config.get("suffix_tokens", 128)
        gen_tokens = self.config.get("gen_tokens", 128)

        prefix_text = build_filler_prefix(shared_prefix_tokens)
        t0 = time.time()
        branch_ttfts = []

        # Checkpoint at prefix boundary
        self.emit(EventType.CHECKPOINT,
                  meta={"token_position": shared_prefix_tokens})

        for branch_id in range(branch_factor):
            suffix = f"\n[Branch {branch_id}] " + (
                f"Option {branch_id}: analyze from perspective {branch_id}. " * (suffix_tokens // 10)
            )
            messages = [{"role": "user", "content": prefix_text + suffix}]

            self.emit(EventType.BRANCH_START,
                      meta={"branch_id": branch_id,
                            "parent_checkpoint": shared_prefix_tokens})

            result = await self.do_step(messages, max_tokens=gen_tokens)
            branch_ttfts.append(result.ttft_sec)

            self.emit(EventType.BRANCH_END,
                      meta={"branch_id": branch_id})

        self.emit(EventType.END_SESSION)
        wall = time.time() - t0

        return WorkflowResult(
            session_id=self.session_id,
            wall_sec=wall,
            ttfts=branch_ttfts,
            steps=branch_factor,
            extra={
                "shared_prefix_tokens": shared_prefix_tokens,
                "branch_factor": branch_factor,
                "suffix_tokens": suffix_tokens,
                "branch_ttfts": branch_ttfts,
            },
        )


async def run_branch_benchmark(
    backend: VLLMBackend,
    event_bus: EventBus,
    config: dict,
) -> list[dict]:
    """Sweep branch_factor × prefix × suffix for Benchmark C."""
    results = []
    bf_list = config.get("branch_factor", [2, 4])
    pt_list = config.get("shared_prefix_tokens", [2048, 8192])
    st_list = config.get("suffix_tokens", [128, 512])
    gen_tokens = config.get("gen_tokens", 128)

    for bf in bf_list:
        for pt in pt_list:
            for st in st_list:
                sid = f"branch_{bf}_{pt}_{st}_{uuid.uuid4().hex[:6]}"
                wf_config = {
                    "shared_prefix_tokens": pt,
                    "branch_factor": bf,
                    "suffix_tokens": st,
                    "gen_tokens": gen_tokens,
                }
                wf = BranchWorkflow(sid, backend, event_bus, wf_config)
                wr = await wf.run()
                results.append({
                    "session_id": wr.session_id,
                    "branch_factor": bf,
                    "shared_prefix_tokens": pt,
                    "suffix_tokens": st,
                    "branch_ttfts": wr.ttfts,
                    "wall_sec": wr.wall_sec,
                    "extra": wr.extra,
                })
    return results
