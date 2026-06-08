from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import functools, subprocess, sys
from pathlib import Path

@dataclass
class Tool:
    name: str
    description: str
    func: Callable[..., Any]
    input_schema: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)
    deterministic: bool = False
    side_effects: List[str] = field(default_factory=list)

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

class ToolRegistry:
    """Registry of tools available to role-specialized DIETA agents."""
    def __init__(self):
        self._tools: Dict[str, Tool] = {}

    def register(self, tool: Tool) -> Tool:
        self._tools[tool.name] = tool
        return tool

    def get(self, name: str) -> Tool:
        return self._tools[name]

    def names(self) -> List[str]:
        return sorted(self._tools)

    def describe(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                'description': t.description,
                'input_schema': t.input_schema,
                'output_schema': t.output_schema,
                'deterministic': t.deterministic,
                'side_effects': t.side_effects,
            }
            for name, t in sorted(self._tools.items())
        }

GLOBAL_TOOL_REGISTRY = ToolRegistry()

def tool(name: Optional[str] = None, description: str = '', **meta):
    """Decorator used to register a callable tool with runtime metadata."""
    def wrapper(func: Callable[..., Any]):
        t = Tool(
            name=name or func.__name__,
            description=description or (func.__doc__ or '').strip(),
            func=func,
            input_schema=meta.get('input_schema', {}),
            output_schema=meta.get('output_schema', {}),
            deterministic=meta.get('deterministic', False),
            side_effects=meta.get('side_effects', []),
        )
        GLOBAL_TOOL_REGISTRY.register(t)
        @functools.wraps(func)
        def inner(*args, **kwargs):
            return func(*args, **kwargs)
        inner.tool = t
        return inner
    return wrapper


def run_python_script(script_path: str | Path, args: Optional[List[str]] = None, cwd: Optional[str | Path] = None) -> subprocess.CompletedProcess:
    cmd = [sys.executable, str(script_path)] + list(args or [])
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, text=True, capture_output=True)
