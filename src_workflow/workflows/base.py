"""
Base workflow class.
All workflow types inherit from this and implement run().
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Optional

from ..events import EventBus, WorkflowEvent, EventType
from ..vllm_backend import VLLMBackend
from ..agent_step import run_step, StepResult


@dataclass
class WorkflowResult:
    session_id: str
    wall_sec: float = 0.0
    ttfts: List[float] = field(default_factory=list)
    retries: int = 0
    steps: int = 0
    extra: dict = field(default_factory=dict)


class BaseWorkflow(ABC):
    def __init__(self, session_id: str, backend: VLLMBackend,
                 event_bus: EventBus, config: dict):
        self.session_id = session_id
        self.backend = backend
        self.event_bus = event_bus
        self.config = config
        self._step_counter = 0

    def emit(self, event_type: EventType, **kwargs):
        ev = WorkflowEvent(
            event_type=event_type,
            session_id=self.session_id,
            step_id=self._step_counter,
            **kwargs,
        )
        self.event_bus.emit(ev)

    async def do_step(self, messages: list[dict],
                      max_tokens: int = 256) -> StepResult:
        result = await run_step(
            session_id=self.session_id,
            step_id=self._step_counter,
            messages=messages,
            backend=self.backend,
            event_bus=self.event_bus,
            max_tokens=max_tokens,
        )
        self._step_counter += 1
        return result

    @abstractmethod
    async def run(self) -> WorkflowResult:
        raise NotImplementedError


def build_filler_prefix(target_tokens: int) -> str:
    """Build a synthetic text of approximately target_tokens tokens.
    ~4 tokens per word on average, ~10 words per sentence."""
    sentence = "The quick brown fox jumps over the lazy dog near the river. "
    # ~12 tokens per sentence
    repeats = max(1, target_tokens // 12)
    return sentence * repeats
