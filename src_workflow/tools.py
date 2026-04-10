"""
Simulated tools for agent workloads.
All tools are async. Each returns (success: bool, result: str).
"""
from __future__ import annotations

import asyncio
import random


async def sleep_tool(duration_sec: float = 1.0) -> tuple[bool, str]:
    await asyncio.sleep(duration_sec)
    return True, f"slept {duration_sec:.1f}s"


async def flaky_tool(p_fail: float = 0.5,
                     latency_sec: float = 0.1) -> tuple[bool, str]:
    await asyncio.sleep(latency_sec)
    if random.random() < p_fail:
        return False, "tool_error: simulated failure"
    return True, "tool_ok: success"


async def search_stub(query: str = "",
                      latency_sec: float = 1.0) -> tuple[bool, str]:
    await asyncio.sleep(latency_sec)
    return True, f"search_result: [paragraph about '{query}' ...]"


async def lookup_stub(title: str = "",
                      latency_sec: float = 0.5) -> tuple[bool, str]:
    await asyncio.sleep(latency_sec)
    return True, f"lookup_result: [sentence from '{title}' ...]"


TOOL_REGISTRY = {
    "sleep": sleep_tool,
    "flaky": flaky_tool,
    "search": search_stub,
    "lookup": lookup_stub,
}
