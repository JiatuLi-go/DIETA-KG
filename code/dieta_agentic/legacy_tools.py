from __future__ import annotations
from pathlib import Path
from typing import List, Optional
from .tools import tool, run_python_script

CORE_DIR = Path(__file__).resolve().parents[1] / 'dieta_core'


def _run(script_name: str, limit: Optional[int] = None):
    args: List[str] = []
    if limit is not None:
        args += ['--limit', str(limit)]
    return run_python_script(CORE_DIR / script_name, args=args, cwd=CORE_DIR)

@tool('semantic_triage_tool', description='Run ontology-constrained semantic triage over PubMed abstracts.', side_effects=['results/semantic_triage/*.xlsx'])
def semantic_triage_tool(limit: Optional[int] = None):
    return _run('semantic_triage_agent.py', limit)

@tool('seed_entity_tool', description='Run route-conditioned pretrained NER seed entity generation.', side_effects=['results/seed_entity/*.json'])
def seed_entity_tool(limit: Optional[int] = None):
    return _run('seed_entity_builder.py', limit)

@tool('candidate_entity_tool', description='Run multi-prompt candidate entity extraction and self-consistency aggregation.', side_effects=['results/candidate_extraction/candidate_entity*'])
def candidate_entity_tool(limit: Optional[int] = None):
    return _run('candidate_entity_agent.py', limit)

@tool('candidate_relation_tool', description='Run route-conditioned candidate relation extraction and relation voting.', side_effects=['results/candidate_extraction/candidate_relation*'])
def candidate_relation_tool(limit: Optional[int] = None):
    return _run('candidate_relation_agent.py', limit)

@tool('critic_quality_tool', description='Run RUQA Critic Agent for entropy-aware quality adjudication and feedback decisions.', side_effects=['results/quality_decision/*'])
def critic_quality_tool(limit: Optional[int] = None):
    return _run('quality_decision_agent.py', limit)

@tool('grounding_alignment_tool', description='Run type-aware entity grounding and open-world alignment decisions.', side_effects=['results/grounding_alignment/*'])
def grounding_alignment_tool(limit: Optional[int] = None):
    return _run('grounding_alignment_agent.py', limit)

@tool('feedback_refinement_tool', description='Run feedback controller for boundary ambiguity and re-extraction requests.', side_effects=['results/grounding_alignment/*'])
def feedback_refinement_tool(limit: Optional[int] = None):
    return _run('feedback_refinement_agent.py', None)
