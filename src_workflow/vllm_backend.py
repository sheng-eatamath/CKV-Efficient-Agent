"""
Thin wrapper around vLLM's OpenAI-compatible API.

Responsibilities:
  - Send chat completion requests (streaming for TTFT measurement)
  - Return usage stats
  - Expose /metrics scraping for profiler
"""
from __future__ import annotations

import logging
import time
import aiohttp
from dataclasses import dataclass
from typing import Optional
from openai import AsyncOpenAI

log = logging.getLogger(__name__)


@dataclass
class GenerateResult:
    text: str
    prompt_tokens: Optional[int]
    completion_tokens: Optional[int]
    ttft_sec: float
    total_sec: float


class VLLMBackend:
    def __init__(self, host: str = "localhost", port: int = 8000,
                 model: str = "meta-llama/Meta-Llama-3-8B-Instruct"):
        self.base_url = f"http://{host}:{port}/v1"
        self.metrics_url = f"http://{host}:{port}/metrics"
        self.model = model
        self.client = AsyncOpenAI(
            base_url=self.base_url, api_key="dummy",
            timeout=120.0,
        )

    async def generate(self, messages: list[dict], max_tokens: int = 256,
                       temperature: float = 0.0) -> GenerateResult:
        t0 = time.perf_counter()
        ttft = None
        full_text = ""

        prompt_tokens = None
        completion_tokens = None

        stream = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True,
            stream_options={"include_usage": True},
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                if ttft is None:
                    ttft = time.perf_counter() - t0
                full_text += chunk.choices[0].delta.content
            # vLLM sends usage in the final chunk (no choices)
            if hasattr(chunk, "usage") and chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
                completion_tokens = chunk.usage.completion_tokens

        total = time.perf_counter() - t0

        return GenerateResult(
            text=full_text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            ttft_sec=ttft if ttft is not None else total,
            total_sec=total,
        )

    async def generate_non_streaming(self, messages: list[dict],
                                     max_tokens: int = 256,
                                     temperature: float = 0.0) -> GenerateResult:
        t0 = time.perf_counter()
        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=False,
        )
        total = time.perf_counter() - t0
        choice = resp.choices[0]
        usage = resp.usage
        return GenerateResult(
            text=choice.message.content or "",
            prompt_tokens=usage.prompt_tokens if usage else None,
            completion_tokens=usage.completion_tokens if usage else None,
            ttft_sec=total,
            total_sec=total,
        )

    async def scrape_metrics(self) -> dict:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(self.metrics_url) as resp:
                text = await resp.text()
        return _parse_prometheus(text)


def _parse_prometheus(text: str) -> dict:
    metrics = {}
    for line in text.splitlines():
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        if len(parts) >= 2:
            name = parts[0].split("{")[0]
            try:
                metrics[name] = float(parts[-1])
            except ValueError:
                pass
    return metrics
