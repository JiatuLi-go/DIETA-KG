#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from dieta_agentic.dieta_workflow import build_dieta_graph, new_state, list_agent_cards
from dieta_agentic.ablation import load_ablation_config
from dieta_agentic.tools import GLOBAL_TOOL_REGISTRY


def main():
    parser = argparse.ArgumentParser(description='Run DIETA as an explicit controlled multi-agent workflow.')
    parser.add_argument('--dry-run', action='store_true', help='Print/trace the agent graph without executing underlying tools.')
    parser.add_argument('--list-agents', action='store_true', help='List published AgentCards.')
    parser.add_argument('--list-tools', action='store_true', help='List registered tool metadata.')
    parser.add_argument('--max-records', type=int, default=None, help='Optional max records forwarded to legacy agents that support --limit.')
    parser.add_argument('--state-out', default='agent_run_state.json', help='Where to write final state snapshot.')
    parser.add_argument('--variant', default='full', help='Ablation/run variant id, e.g. full, no_uncertainty, no_feedback, no_relation_consistency, no_routing.')
    parser.add_argument('--ablation-config', default=None, help='Optional JSON file with explicit ablation switches.')
    parser.add_argument('--mermaid-out', default='workflow_graph.mmd', help='Where to write Mermaid graph.')
    args = parser.parse_args()

    if args.list_agents:
        print(json.dumps(list_agent_cards(), ensure_ascii=False, indent=2))
        return
    if args.list_tools:
        print(json.dumps(GLOBAL_TOOL_REGISTRY.describe(), ensure_ascii=False, indent=2))
        return

    graph = build_dieta_graph()
    compiled = graph.compile()
    Path(args.mermaid_out).write_text(compiled.draw_mermaid(), encoding='utf-8')
    cfg = load_ablation_config(args.variant, args.ablation_config)
    # Forward variant switches to the underlying scientific modules.
    # Subprocess-based tools inherit these values.
    os.environ.update(cfg.to_env())
    state = new_state(ablation_variant=cfg.variant_id, ablation_config=cfg.__dict__)
    state = compiled.invoke(state, dry_run=args.dry_run, max_records=args.max_records)
    state.write_snapshot(args.state_out)
    print(json.dumps({'run_id': state.run_id, 'dry_run': args.dry_run, 'variant': state.ablation_variant, 'state_out': args.state_out, 'mermaid_out': args.mermaid_out}, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
