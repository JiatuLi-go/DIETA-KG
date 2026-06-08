# Runtime architecture

The workflow is implemented as a bounded state graph. Each agent has a declared task scope, input artifacts, output artifacts, tool access and handoff target. This makes the pipeline inspectable without turning it into an open-ended autonomous system.

## Components

- `AgentCard`: metadata for a role-specialized processing unit.
- `ToolRegistry`: registry of callable tools used by agents.
- `DIETAAgentState`: run-level state object storing artifacts, decisions and ablation metadata.
- `StateGraph`: deterministic graph runner for sequential and conditional handoffs.
- `Contract Guard`: final-table schema normalizer for Step4 and Step5 exports.
- `Ablation Trace Exporter`: additive trace writer for score components and vote-level diagnostics.

## Design boundary

Agents do not select arbitrary tools or modify the schema at runtime. This boundary is intentional: biomedical graph construction requires deterministic schema control, traceable evidence admission and stable public outputs.
