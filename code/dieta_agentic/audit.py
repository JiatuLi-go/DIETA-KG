from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import json, time, uuid

class JsonlAuditLogger:
    """JSONL audit trace compatible with runtime inspection."""
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event: Dict[str, Any]) -> None:
        event = dict(event)
        event.setdefault('event_id', str(uuid.uuid4()))
        event.setdefault('timestamp', time.time())
        with self.path.open('a', encoding='utf-8') as f:
            f.write(json.dumps(event, ensure_ascii=False) + '\n')
