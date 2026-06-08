from __future__ import annotations
from pathlib import Path
from typing import Optional
import time, uuid

from .agent import AgentCard, AgentResult, agent, handoff
from .graph import StateGraph, START, END
from .state import DIETAAgentState, ArtifactRef
from .tools import GLOBAL_TOOL_REGISTRY
from . import legacy_tools  # registers legacy scientific tools
from . import contract_tools  # registers output-contract and ablation tools

CORE_DIR = Path(__file__).resolve().parents[1] / 'dieta_core'
RESULTS_DIR = CORE_DIR / 'results'


def _tool_result_to_agent_result(agent_id: str, cp, outputs, handoff_to: Optional[str] = None) -> AgentResult:
    status = 'completed' if getattr(cp, 'returncode', 0) == 0 else 'failed'
    msg = (getattr(cp, 'stderr', '') or getattr(cp, 'stdout', '') or '')[-1200:]
    return AgentResult(agent_id=agent_id, status=status, message=msg, artifacts=outputs, metrics={'return_code': getattr(cp, 'returncode', None)}, handoff_to=handoff_to)

@agent(AgentCard(
    agent_id='triage_agent',
    name='Triage Agent',
    role='Ontology-constrained semantic router',
    instructions='Activate task-ontology routing spaces T1-T6 before candidate generation.',
    input_artifacts=['data/diabetes_diet_papers*.csv', 'prompts/router.txt'],
    output_artifacts=['results/semantic_triage/semantic_triage_latest.xlsx'],
    tools=['semantic_triage_tool'],
    handoffs=[handoff('seed_entity_agent')],
    guardrails=['schema route set must be T1-T6', 'out-of-scope documents do not enter extraction'],
    output_contract='Excel file with pmid/title/abstract and T1-T6 route indicators.'
))
def triage_agent_run(ctx):
    if ctx.dry_run:
        return AgentResult('triage_agent', 'dry_run', 'Would run semantic triage.', handoff_to='seed_entity_agent')
    cp = GLOBAL_TOOL_REGISTRY.get('semantic_triage_tool')(ctx.max_records)
    out = 'results/semantic_triage/semantic_triage_latest.xlsx'
    ctx.state.put_artifact(ArtifactRef('semantic_triage', str(RESULTS_DIR/'semantic_triage/semantic_triage_latest.xlsx'), 'xlsx', 'triage_agent', 'semantic_triage_schema'))
    return _tool_result_to_agent_result('triage_agent', cp, [out], 'seed_entity_agent')

@agent(AgentCard(
    agent_id='seed_entity_agent',
    name='Seed Entity Agent',
    role='Route-conditioned pretrained entity seeding',
    instructions='Use entity-type-specific pretrained models to provide structured priors for candidate generation.',
    input_artifacts=['results/semantic_triage/semantic_triage_latest.xlsx'],
    output_artifacts=['results/seed_entity/seed_entity_latest.json'],
    tools=['seed_entity_tool'],
    handoffs=[handoff('candidate_entity_agent')],
    guardrails=['Food/Disease/Nutrient models are type-specific', 'DietaryPattern remains LLM-mediated due to missing robust pretrained NER'],
    output_contract='JSON keyed by PMID with seed_entities and matched_clusters.'
))
def seed_entity_agent_run(ctx):
    if ctx.dry_run:
        return AgentResult('seed_entity_agent', 'dry_run', 'Would run seed entity generation.', handoff_to='candidate_entity_agent')
    cp = GLOBAL_TOOL_REGISTRY.get('seed_entity_tool')(ctx.max_records)
    out = 'results/seed_entity/seed_entity_latest.json'
    ctx.state.put_artifact(ArtifactRef('seed_entity', str(RESULTS_DIR/'seed_entity/seed_entity_latest.json'), 'json', 'seed_entity_agent', 'seed_entity_schema'))
    return _tool_result_to_agent_result('seed_entity_agent', cp, [out], 'candidate_entity_agent')

@agent(AgentCard(
    agent_id='candidate_entity_agent',
    name='Candidate Entity Agent',
    role='Multi-path candidate entity generator',
    instructions='Execute route-specific NER prompts and aggregate entity self-consistency votes.',
    input_artifacts=['results/seed_entity/seed_entity_latest.json'],
    output_artifacts=['results/candidate_extraction/candidate_entity_extraction_latest.json', 'results/candidate_extraction/candidate_entity_extraction_latest.xlsx'],
    tools=['candidate_entity_tool'],
    handoffs=[handoff('candidate_relation_agent')],
    guardrails=['output entity types must remain Food/Nutrient/Disease/DietaryPattern', 'confidence derives from prompt-vote consistency'],
    output_contract='JSON keyed by PMID with entities and raw_votes; Excel flattened view for audit.'
))
def candidate_entity_agent_run(ctx):
    if ctx.dry_run:
        return AgentResult('candidate_entity_agent', 'dry_run', 'Would run candidate entity extraction.', handoff_to='candidate_relation_agent')
    cp = GLOBAL_TOOL_REGISTRY.get('candidate_entity_tool')(ctx.max_records)
    out = 'results/candidate_extraction/candidate_entity_extraction_latest.json'
    ctx.state.put_artifact(ArtifactRef('candidate_entities', str(RESULTS_DIR/'candidate_extraction/candidate_entity_extraction_latest.json'), 'json', 'candidate_entity_agent', 'candidate_entity_schema'))
    return _tool_result_to_agent_result('candidate_entity_agent', cp, [out], 'candidate_relation_agent')

@agent(AgentCard(
    agent_id='candidate_relation_agent',
    name='Candidate Relation Agent',
    role='Route-conditioned relation generator',
    instructions='Construct type-permissible entity pairs and generate candidate relations under legal relation schema.',
    input_artifacts=['results/candidate_extraction/candidate_entity_extraction_latest.json'],
    output_artifacts=['results/candidate_extraction/candidate_relation_extraction_latest.json', 'results/candidate_extraction/candidate_relation_extraction_latest.xlsx'],
    tools=['candidate_relation_tool'],
    handoffs=[handoff('critic_agent')],
    guardrails=['relation must belong to seven formal GCDI relations', 'relation direction must match route schema'],
    output_contract='JSON keyed by PMID/row_id with candidate triples, relation votes and topic.'
))
def candidate_relation_agent_run(ctx):
    if ctx.dry_run:
        return AgentResult('candidate_relation_agent', 'dry_run', 'Would run candidate relation extraction.', handoff_to='critic_agent')
    cp = GLOBAL_TOOL_REGISTRY.get('candidate_relation_tool')(ctx.max_records)
    out = 'results/candidate_extraction/candidate_relation_extraction_latest.json'
    ctx.state.put_artifact(ArtifactRef('candidate_relations', str(RESULTS_DIR/'candidate_extraction/candidate_relation_extraction_latest.json'), 'json', 'candidate_relation_agent', 'candidate_relation_schema'))
    return _tool_result_to_agent_result('candidate_relation_agent', cp, [out], 'critic_agent')

@agent(AgentCard(
    agent_id='critic_agent',
    name='Critic Agent',
    role='Entropy-aware quality adjudicator',
    instructions='Calculate RUQA retention score, produce reflection reports and keep/revise/discard decisions.',
    input_artifacts=['results/candidate_extraction/candidate_entity_extraction_latest.json', 'results/candidate_extraction/candidate_relation_extraction_latest.json'],
    output_artifacts=['results/quality_decision/quality_decision_latest.json', 'results/quality_decision/quality_decision_latest.xlsx'],
    tools=['critic_quality_tool'],
    handoffs=[handoff('grounding_agent')],
    guardrails=['s^(k)(t)=lambda_reflect(k)Q^(k)(t)-lambda U^(0)(t)', 'low-quality candidates are not inserted into the KG'],
    output_contract='JSON/Excel with final_score, quality_tier, reflection report and retained triples.'
))
def critic_agent_run(ctx):
    if ctx.dry_run:
        return AgentResult('critic_agent', 'dry_run', 'Would run RUQA quality decision.', handoff_to='grounding_agent')
    cp = GLOBAL_TOOL_REGISTRY.get('critic_quality_tool')(ctx.max_records)
    out = 'results/quality_decision/quality_decision_latest.json'
    ctx.state.put_artifact(ArtifactRef('quality_decision', str(RESULTS_DIR/'quality_decision/quality_decision_latest.json'), 'json', 'critic_agent', 'quality_decision_schema'))
    return _tool_result_to_agent_result('critic_agent', cp, [out], 'grounding_agent')


@agent(AgentCard(
    agent_id='quality_contract_agent',
    name='Step4 Contract Guard Agent',
    role='Legacy output-format guard for Critic/Step4 outputs',
    instructions='Normalize quality_decision_latest.xlsx to the locked Step4 contract without changing RUQA formulas or retained-triple logic.',
    input_artifacts=['results/quality_decision/quality_decision_latest.json', 'results/quality_decision/quality_decision_latest.xlsx'],
    output_artifacts=['results/quality_decision/quality_decision_latest.xlsx', 'results/contracts/output_contract_report.json'],
    tools=['quality_contract_tool', 'ablation_snapshot_tool'],
    handoffs=[handoff('grounding_agent')],
    guardrails=['only tabular export columns may be normalized', 'scientific JSON and scoring components are not rewritten'],
    output_contract='Step4 Excel columns remain compatible with legacy quality_decision exports.'
))
def quality_contract_agent_run(ctx):
    if ctx.dry_run:
        return AgentResult('quality_contract_agent', 'dry_run', 'Would normalize Step4 quality output and snapshot ablation state.', handoff_to='grounding_agent')
    report = GLOBAL_TOOL_REGISTRY.get('quality_contract_tool')()
    GLOBAL_TOOL_REGISTRY.get('ablation_snapshot_tool')(ctx.state.ablation_variant, 'quality_contract_agent')
    ctx.state.add_decision('quality_contract_agent', 'normalized_step4_contract', report if isinstance(report, dict) else {'report': str(report)})
    return AgentResult('quality_contract_agent', 'completed', 'Step4 output contract checked/normalized.', artifacts=['results/quality_decision/quality_decision_latest.xlsx'], metrics={'rows': report.get('rows') if isinstance(report, dict) else None}, handoff_to='grounding_agent')

@agent(AgentCard(
    agent_id='grounding_agent',
    name='Grounding Agent',
    role='Type-aware semantic anchoring agent',
    instructions='Align retained entities with HumanDO/SapBERT, USDA-derived FAISS index, MeSH/ONS and open-world retention logic.',
    input_artifacts=['results/quality_decision/quality_decision_latest.json'],
    output_artifacts=['results/grounding_alignment/grounding_alignment_latest.json', 'results/grounding_alignment/grounding_alignment_latest.xlsx', 'results/grounding_alignment/feedback_refinement_requests.json'],
    tools=['grounding_alignment_tool'],
    handoffs=[handoff('feedback_controller'), handoff('__end__')],
    guardrails=['type-specific grounding channel', 'open-world concepts are retained rather than force-merged'],
    output_contract='Grounded triples with alignment status, standard IDs/names where available and feedback flags.'
))
def grounding_agent_run(ctx):
    if ctx.dry_run:
        return AgentResult('grounding_agent', 'dry_run', 'Would run grounding alignment.', handoff_to='feedback_controller')
    cp = GLOBAL_TOOL_REGISTRY.get('grounding_alignment_tool')(ctx.max_records)
    out = 'results/grounding_alignment/grounding_alignment_latest.json'
    feedback_path = RESULTS_DIR/'grounding_alignment/feedback_refinement_requests.json'
    disable_feedback = bool(ctx.state.ablation_config.get('disable_feedback_loop') or ctx.state.ablation_config.get('disable_grounding_feedback'))
    ctx.state.feedback_pending = feedback_path.exists() and not disable_feedback
    ctx.state.put_artifact(ArtifactRef('grounding_alignment', str(RESULTS_DIR/'grounding_alignment/grounding_alignment_latest.json'), 'json', 'grounding_agent', 'grounding_alignment_schema'))
    if feedback_path.exists():
        ctx.state.put_artifact(ArtifactRef('feedback_requests', str(feedback_path), 'json', 'grounding_agent', 'feedback_request_schema', required=False))
        if disable_feedback:
            ctx.state.add_decision('grounding_agent', 'feedback_requests_recorded_but_not_executed', {'path': str(feedback_path), 'variant': ctx.state.ablation_variant})
    return _tool_result_to_agent_result('grounding_agent', cp, [out], 'feedback_controller' if ctx.state.feedback_pending else 'final_contract_guard_agent')

@agent(AgentCard(
    agent_id='feedback_controller',
    name='Feedback Controller',
    role='Boundary-ambiguity and revision controller',
    instructions='Consume grounding feedback requests, call Extraction/Critic/Grounding tools for selected cases and merge updates.',
    input_artifacts=['results/grounding_alignment/feedback_refinement_requests.json'],
    output_artifacts=['results/grounding_alignment/grounding_alignment_latest.json', 'results/grounding_alignment/grounding_alignment_latest.xlsx'],
    tools=['feedback_refinement_tool'],
    handoffs=[handoff('__end__')],
    guardrails=['feedback is bounded and non-recursive beyond is_retry protection', 'output schema remains grounding_alignment schema'],
    output_contract='Updated grounding_alignment_latest JSON/XLSX with feedback_completed flag.'
))
def feedback_controller_run(ctx):
    if ctx.dry_run:
        return AgentResult('feedback_controller', 'dry_run', 'Would run feedback refinement if requests exist.', handoff_to=END)
    cp = GLOBAL_TOOL_REGISTRY.get('feedback_refinement_tool')()
    out = 'results/grounding_alignment/grounding_alignment_latest.json'
    ctx.state.put_artifact(ArtifactRef('grounding_alignment_after_feedback', str(RESULTS_DIR/'grounding_alignment/grounding_alignment_latest.json'), 'json', 'feedback_controller', 'grounding_alignment_schema'))
    return _tool_result_to_agent_result('feedback_controller', cp, [out], END)



@agent(AgentCard(
    agent_id='final_contract_guard_agent',
    name='Final Contract Guard Agent',
    role='Legacy output-format guard for final Step5 outputs',
    instructions='Normalize grounding_alignment_latest.xlsx to the locked step5 after-feedback contract while preserving JSON graph facts.',
    input_artifacts=['results/grounding_alignment/grounding_alignment_latest.json', 'results/grounding_alignment/grounding_alignment_latest.xlsx'],
    output_artifacts=['results/grounding_alignment/grounding_alignment_latest.xlsx', 'results/contracts/output_contract_report.json'],
    tools=['grounding_contract_tool', 'ablation_snapshot_tool'],
    handoffs=[handoff('ablation_trace_agent')],
    guardrails=['final xlsx columns must match the legacy step5 file exactly', 'do not alter candidate generation or RUQA scoring'],
    output_contract='Step5 Excel columns: pmid, subject, relation, object, source_cluster, quality_tier, Score_t, head_status, head_standard, head_similarity, tail_status, tail_standard, tail_similarity, needs_feedback, feedback_completed.'
))
def final_contract_guard_agent_run(ctx):
    if ctx.dry_run:
        return AgentResult('final_contract_guard_agent', 'dry_run', 'Would normalize Step5 grounding output contract.', handoff_to='ablation_trace_agent')
    report = GLOBAL_TOOL_REGISTRY.get('grounding_contract_tool')()
    GLOBAL_TOOL_REGISTRY.get('ablation_snapshot_tool')(ctx.state.ablation_variant, 'final_contract_guard_agent')
    ctx.state.add_decision('final_contract_guard_agent', 'normalized_step5_contract', report if isinstance(report, dict) else {'report': str(report)})
    return AgentResult('final_contract_guard_agent', 'completed', 'Final output contract checked/normalized.', artifacts=['results/grounding_alignment/grounding_alignment_latest.xlsx'], metrics={'rows': report.get('rows') if isinstance(report, dict) else None}, handoff_to='ablation_trace_agent')

@agent(AgentCard(
    agent_id='ablation_trace_agent',
    name='Ablation Trace Agent',
    role='Intermediate evidence and score-component trace exporter',
    instructions='Export vote-level, entropy-level and score-component traces for component ablations without changing final legacy outputs.',
    input_artifacts=['results/candidate_extraction/candidate_relation_extraction_latest.json', 'results/quality_decision/quality_decision_latest.json'],
    output_artifacts=['results/ablation/<variant>/ablation_trace_latest.jsonl', 'results/ablation/<variant>/ablation_config.json'],
    tools=['ablation_trace_tool'],
    handoffs=[handoff('__end__')],
    guardrails=['trace export is additive only', 'final step4/step5 file contracts are unchanged'],
    output_contract='JSONL rows contain variant_id, candidate relation votes, raw_votes, entropy, Q_t, U_t, Score_t and feedback_rounds where available.'
))
def ablation_trace_agent_run(ctx):
    if ctx.dry_run:
        return AgentResult('ablation_trace_agent', 'dry_run', 'Would export ablation trace.', handoff_to=END)
    report = GLOBAL_TOOL_REGISTRY.get('ablation_trace_tool')(ctx.state.ablation_variant)
    ctx.state.add_decision('ablation_trace_agent', 'exported_ablation_trace', report if isinstance(report, dict) else {'report': str(report)})
    return AgentResult('ablation_trace_agent', 'completed', 'Ablation trace exported.', artifacts=[f"results/ablation/{ctx.state.ablation_variant}/ablation_trace_latest.jsonl"], metrics={'trace_rows': report.get('trace_rows') if isinstance(report, dict) else None}, handoff_to=END)

def build_dieta_graph() -> StateGraph:
    graph = StateGraph()
    agents = [
        triage_agent_run, seed_entity_agent_run, candidate_entity_agent_run,
        candidate_relation_agent_run, critic_agent_run, grounding_agent_run,
        feedback_controller_run, quality_contract_agent_run, final_contract_guard_agent_run, ablation_trace_agent_run,
    ]
    for a in agents:
        graph.add_node(a)
    graph.add_edge(START, 'triage_agent')
    graph.add_edge('triage_agent', 'seed_entity_agent')
    graph.add_edge('seed_entity_agent', 'candidate_entity_agent')
    graph.add_edge('candidate_entity_agent', 'candidate_relation_agent')
    graph.add_edge('candidate_relation_agent', 'critic_agent')
    graph.add_edge('critic_agent', 'quality_contract_agent')
    graph.add_edge('quality_contract_agent', 'grounding_agent')
    def grounding_router(state, result):
        return 'feedback_controller' if state.feedback_pending else 'final_contract_guard_agent'
    graph.add_conditional_edge('grounding_agent', grounding_router)
    graph.add_edge('feedback_controller', 'final_contract_guard_agent')
    graph.add_edge('final_contract_guard_agent', 'ablation_trace_agent')
    graph.add_edge('ablation_trace_agent', END)
    return graph


def new_state(base_dir: str | None = None, *, ablation_variant: str = 'full', ablation_config: dict | None = None) -> DIETAAgentState:
    return DIETAAgentState(run_id=f"dieta-{uuid.uuid4().hex[:12]}", base_dir=str(base_dir or CORE_DIR), ablation_variant=ablation_variant, ablation_config=ablation_config or {})


def list_agent_cards():
    return [a.to_dict() for a in [
        triage_agent_run, seed_entity_agent_run, candidate_entity_agent_run,
        candidate_relation_agent_run, critic_agent_run, grounding_agent_run,
        feedback_controller_run, quality_contract_agent_run, final_contract_guard_agent_run, ablation_trace_agent_run,
    ]]
