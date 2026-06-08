from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Dict, List, Optional
from .runtime_primitives import HandoffRecord, RetryPolicy
import time
from .state import DIETAAgentState
from .tools import GLOBAL_TOOL_REGISTRY, Tool

@dataclass
class AgentCard:
    """Metadata card for a DIETA role-specialized processing unit."""
    agent_id: str
    name: str
    role: str
    instructions: str
    input_artifacts: List[str]
    output_artifacts: List[str]
    tools: List[str]
    handoffs: List[str] = field(default_factory=list)
    guardrails: List[str] = field(default_factory=list)
    output_contract: str = ''

@dataclass
class AgentContext:
    state: DIETAAgentState
    dry_run: bool = False
    max_records: Optional[int] = None
    extra_args: List[str] = field(default_factory=list)

@dataclass
class AgentResult:
    agent_id: str
    status: str
    message: str = ''
    artifacts: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    handoff_to: Optional[str] = None

class Agent:
    """Minimal DIETA runtime agent.

    An Agent has instructions, registered tools, guardrails, typed
    input/output artifacts, and explicit handoffs.
    """
    def __init__(self, card: AgentCard, run_fn: Callable[[AgentContext], AgentResult]):
        self.card = card
        self.run_fn = run_fn

    def run(self, ctx: AgentContext) -> AgentResult:
        started = time.time()
        result = self.run_fn(ctx)
        result.metrics.setdefault('runtime_seconds', time.time() - started)
        ctx.state.add_decision(self.card.agent_id, result.status, {
            'message': result.message,
            'handoff_to': result.handoff_to,
            'artifacts': result.artifacts,
            'metrics': result.metrics,
        })
        return result

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self.card)

def agent(card: AgentCard):
    """Decorator for registering a DIETA agent run function."""
    def wrapper(func: Callable[[AgentContext], AgentResult]) -> Agent:
        return Agent(card=card, run_fn=func)
    return wrapper

def handoff(target_agent_id: str) -> str:
    """Declare a typed handoff target."""
    return target_agent_id
