from __future__ import annotations
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional
from .agent import Agent, AgentContext, AgentResult
from .state import DIETAAgentState

START = '__start__'
END = '__end__'

class StateGraph:
    """Small deterministic state graph for explicit workflow orchestration."""
    def __init__(self):
        self.nodes: Dict[str, Agent] = {}
        self.edges: Dict[str, List[str]] = {START: []}
        self.conditional_edges: Dict[str, Callable[[DIETAAgentState, AgentResult], str]] = {}

    def add_node(self, agent: Agent):
        self.nodes[agent.card.agent_id] = agent
        self.edges.setdefault(agent.card.agent_id, [])
        return self

    def add_edge(self, source: str, target: str):
        self.edges.setdefault(source, []).append(target)
        return self

    def add_conditional_edge(self, source: str, router: Callable[[DIETAAgentState, AgentResult], str]):
        self.conditional_edges[source] = router
        return self

    def compile(self) -> 'CompiledStateGraph':
        return CompiledStateGraph(self.nodes, self.edges, self.conditional_edges)

class CompiledStateGraph:
    def __init__(self, nodes, edges, conditional_edges):
        self.nodes = nodes
        self.edges = edges
        self.conditional_edges = conditional_edges

    def invoke(self, state: DIETAAgentState, *, dry_run: bool = False, max_records: Optional[int] = None) -> DIETAAgentState:
        cursor = self.edges.get(START, [END])[0] if self.edges.get(START) else END
        last_result: Optional[AgentResult] = None
        while cursor != END:
            agent = self.nodes[cursor]
            ctx = AgentContext(state=state, dry_run=dry_run, max_records=max_records)
            last_result = agent.run(ctx)
            if cursor in self.conditional_edges:
                cursor = self.conditional_edges[cursor](state, last_result)
            else:
                nexts = self.edges.get(cursor, [END])
                cursor = nexts[0] if nexts else END
        return state

    def draw_mermaid(self) -> str:
        lines = ['flowchart TD']
        for src, tgts in self.edges.items():
            for tgt in tgts:
                lines.append(f'  {src.replace("__", "")} --> {tgt.replace("__", "")}')
        for src in self.conditional_edges:
            lines.append(f'  {src} -. conditional .-> feedback_or_end')
        return '\n'.join(lines)
