"""
Single LLM call step — no ReAct parsing.
The workflow harness decides what to do with the output.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
from .events import EventBus, WorkflowEvent, EventType
from .vllm_backend import VLLMBackend, GenerateResult


@dataclass
class StepResult:
    text: str
    prompt_tokens: Optional[int]
    gen_tokens: Optional[int]
    ttft_sec: float
    total_sec: float


async def run_step(
    session_id: str,
    step_id: int,
    messages: list[dict],
    backend: VLLMBackend,
    event_bus: EventBus,
    max_tokens: int = 256,
) -> StepResult:
    event_bus.emit(WorkflowEvent(
        event_type=EventType.BEFORE_GENERATE,
        session_id=session_id,
        step_id=step_id,
    ))

    result: GenerateResult = await backend.generate(
        messages, max_tokens=max_tokens
    )

    event_bus.emit(WorkflowEvent(
        event_type=EventType.AFTER_GENERATE,
        session_id=session_id,
        step_id=step_id,
        prompt_tokens=result.prompt_tokens or 0,
        generated_tokens=result.completion_tokens or 0,
        meta={"ttft_sec": result.ttft_sec, "total_sec": result.total_sec},
    ))

    return StepResult(
        text=result.text,
        prompt_tokens=result.prompt_tokens,
        gen_tokens=result.completion_tokens,
        ttft_sec=result.ttft_sec,
        total_sec=result.total_sec,
    )
