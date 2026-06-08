from __future__ import annotations

"""Small runtime primitives used by the DIETA orchestrator.

These primitives provide the run-level controls used by the biomedical pipeline:
run-scoped context, tool call envelopes, guard checks, retry policy metadata and
handoff records.  They are deterministic and do not change scientific outputs.
"""

import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional


@dataclass(frozen=True)
class ToolCallEnvelope:
    call_id: str
    tool_name: str
    agent_id: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)

    @classmethod
    def open(cls, tool_name: str, agent_id: str, **arguments: Any) -> "ToolCallEnvelope":
        return cls(call_id=f"tool-{uuid.uuid4().hex[:12]}", tool_name=tool_name, agent_id=agent_id, arguments=arguments)


@dataclass(frozen=True)
class HandoffRecord:
    source_agent: str
    target_agent: str
    reason: str
    state_keys: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetryPolicy:
    max_attempts: int = 1
    backoff_seconds: float = 0.0
    retry_on_status: List[str] = field(default_factory=lambda: ["failed"])


@dataclass
class GuardCheck:
    name: str
    passed: bool
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)


class GuardrailSet:
    def __init__(self, checks: Optional[List[Callable[[Any], GuardCheck]]] = None):
        self._checks = checks or []

    def add(self, fn: Callable[[Any], GuardCheck]) -> Callable[[Any], GuardCheck]:
        self._checks.append(fn)
        return fn

    def run(self, state: Any) -> List[GuardCheck]:
        return [fn(state) for fn in self._checks]

    def require(self, state: Any) -> None:
        failed = [c for c in self.run(state) if not c.passed]
        if failed:
            joined = "; ".join(f"{c.name}: {c.message}" for c in failed)
            raise RuntimeError(f"Guardrail check failed: {joined}")


def as_dict(obj: Any) -> Dict[str, Any]:
    if hasattr(obj, "__dataclass_fields__"):
        return asdict(obj)
    if isinstance(obj, dict):
        return obj
    return {"value": obj}
