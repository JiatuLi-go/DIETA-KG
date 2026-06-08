from __future__ import annotations
from pathlib import Path
from .tools import tool
from .output_contracts import normalize_grounding_final, normalize_quality_step4, write_contract_report
from .ablation import export_ablation_trace, load_ablation_config, snapshot_stage_outputs

CORE_DIR = Path(__file__).resolve().parents[1] / 'dieta_core'

@tool('quality_contract_tool', description='Normalize Step4 quality_decision exports to legacy columns and write a contract report.', side_effects=['results/quality_decision/quality_decision_latest.xlsx', 'results/contracts/output_contract_report.json'])
def quality_contract_tool():
    return normalize_quality_step4(CORE_DIR)

@tool('grounding_contract_tool', description='Normalize Step5 grounding_alignment exports to legacy after-feedback columns and write a contract report.', side_effects=['results/grounding_alignment/grounding_alignment_latest.xlsx', 'results/contracts/output_contract_report.json'])
def grounding_contract_tool():
    result = normalize_grounding_final(CORE_DIR)
    write_contract_report(CORE_DIR)
    return result

@tool('ablation_trace_tool', description='Export score-component and vote-level intermediate traces for component ablation analysis.', side_effects=['results/ablation/*/ablation_trace_latest.jsonl'])
def ablation_trace_tool(variant_id: str = 'full'):
    cfg = load_ablation_config(variant_id)
    return export_ablation_trace(CORE_DIR, cfg)

@tool('ablation_snapshot_tool', description='Snapshot legacy latest outputs into a per-variant ablation audit folder without changing final file names.', side_effects=['results/ablation/*/stage_outputs/*'])
def ablation_snapshot_tool(variant_id: str = 'full', stage_name: str = 'stage'):
    cfg = load_ablation_config(variant_id)
    return snapshot_stage_outputs(CORE_DIR, cfg, stage_name)
