"""
P0: Workflow Events — structural + operational signals for CKV-Agent.

Structural events (harness-level):
  CHECKPOINT, RETRY_REENTRY, BRANCH_START, BRANCH_END, STALL_BEGIN, STALL_END

Operational events (step-level):
  BEFORE_GENERATE, AFTER_GENERATE, BEFORE_TOOL, AFTER_TOOL, END_SESSION
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional, Callable, List

log = logging.getLogger(__name__)


class EventType(Enum):
    # Structural (harness-level)
    CHECKPOINT = auto()
    RETRY_REENTRY = auto()
    BRANCH_START = auto()
    BRANCH_END = auto()
    STALL_BEGIN = auto()
    STALL_END = auto()

    # Operational (step-level)
    BEFORE_GENERATE = auto()
    AFTER_GENERATE = auto()
    BEFORE_TOOL = auto()
    AFTER_TOOL = auto()
    END_SESSION = auto()


@dataclass
class WorkflowEvent:
    event_type: EventType
    session_id: str
    step_id: int
    timestamp: float = field(default_factory=time.time)
    prompt_tokens: int = 0
    generated_tokens: int = 0
    tool_name: Optional[str] = None
    branch_k: int = 0
    retry_reason: Optional[str] = None
    meta: dict = field(default_factory=dict)


class EventBus:
    """Simple publish-subscribe bus. Profiler and policy engine subscribe."""

    def __init__(self):
        self._subscribers: List[Callable[[WorkflowEvent], None]] = []

    def subscribe(self, fn: Callable[[WorkflowEvent], None]):
        self._subscribers.append(fn)

    def emit(self, event: WorkflowEvent):
        for fn in self._subscribers:
            try:
                fn(event)
            except Exception:
                log.exception("Subscriber %s failed on event %s",
                              fn, event.event_type.name)
