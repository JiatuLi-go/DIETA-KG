from __future__ import annotations
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import hashlib, json, time


def sha256_file(path: str | Path) -> Optional[str]:
    p = Path(path)
    if not p.exists() or not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

@dataclass
class ArtifactRef:
    """Typed state artifact exchanged between DIETA agents."""
    name: str
    path: str
    artifact_type: str
    producer_agent: str
    schema_name: str
    required: bool = True
    sha256: Optional[str] = None

    def refresh(self) -> 'ArtifactRef':
        self.sha256 = sha256_file(self.path)
        return self

    def exists(self) -> bool:
        return Path(self.path).exists()

@dataclass
class DIETAAgentState:
    """Shared state object for the controlled multi-agent workflow."""
    run_id: str
    base_dir: str
    active_routes: List[str] = field(default_factory=list)
    artifacts: Dict[str, ArtifactRef] = field(default_factory=dict)
    decisions: List[Dict[str, Any]] = field(default_factory=list)
    audit_events: List[Dict[str, Any]] = field(default_factory=list)
    feedback_pending: bool = False
    ablation_variant: str = 'full'
    ablation_config: Dict[str, Any] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)

    def put_artifact(self, ref: ArtifactRef) -> None:
        self.artifacts[ref.name] = ref.refresh()

    def get_artifact(self, name: str) -> Optional[ArtifactRef]:
        return self.artifacts.get(name)

    def add_decision(self, agent_id: str, decision: str, details: Dict[str, Any] | None = None) -> None:
        self.decisions.append({
            'agent_id': agent_id,
            'decision': decision,
            'details': details or {},
            'timestamp': time.time(),
        })

    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['elapsed_seconds'] = time.time() - self.started_at
        return data

    def write_snapshot(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding='utf-8')
