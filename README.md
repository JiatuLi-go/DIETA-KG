# DIETA: ontology-constrained multi-agent workflow for biomedical knowledge graph construction

DIETA (Domain-Informed Evidence Triage and Assessment) is a controlled multi-agent workflow for converting biomedical literature into auditable knowledge graph facts. The implementation is designed for clinical nutrition evidence, where relation direction, entity type, grounding quality and schema validity must be controlled before graph insertion.

The repository separates two layers:

- `code/dieta_core/`: scientific modules for semantic triage, candidate entity extraction, candidate relation extraction, RUQA quality adjudication, grounding alignment and feedback refinement.
- `code/dieta_agentic/`: runtime layer for agent cards, registered tools, typed artifacts, state-graph execution, bounded handoffs, output-contract guards and ablation traces.

The runtime layer does not change the public result tables. Its role is to make the workflow explicit, auditable and easier to run under different ablation settings.

## Workflow

```text
Triage Agent
→ Seed Entity Agent
→ Candidate Entity Agent
→ Candidate Relation Agent
→ Critic Agent
→ Step4 Contract Guard
→ Grounding Agent
→ Feedback Controller, when feedback requests exist
→ Step5 Contract Guard
→ Ablation Trace Exporter
```

## Locked output contracts

The final public outputs keep the same file names and column order as the earlier DIETA scripts.

### Step 4: quality decision

Path:

```text
code/dieta_core/results/quality_decision/quality_decision_latest.xlsx
```

Columns:

```text
pmid, subject, subject_type, relation, object, object_type,
P_entity_head, P_entity_tail, P_relation, Q_t, U_t, Score_t,
quality_tier, lambda_reflect, Q_t_penalized, feedback_rounds, source_cluster
```

### Step 5: grounding after feedback

Path:

```text
code/dieta_core/results/grounding_alignment/grounding_alignment_latest.xlsx
```

Columns:

```text
pmid, subject, relation, object, source_cluster, quality_tier, Score_t,
head_status, head_standard, head_similarity, tail_status, tail_standard,
tail_similarity, needs_feedback, feedback_completed
```

Run the contract check before committing results:

```bash
cd code
python contract_smoke_test.py
python verify_output_contracts.py --show-contracts
```

## Installation

A minimal runtime environment requires Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

The full grounding pipeline also needs the biomedical dependencies listed under `pyproject.toml` optional group `biomedical`.

## Configuration

Copy the environment template and fill in the API endpoint and model settings:

```bash
cp .env.example .env
```

Required project resources should be placed under:

```text
code/dieta_core/data/
code/dieta_core/prompts/
code/dieta_core/本体/
code/dieta_core/grounding_data/
```

Raw corpora, caches, indexes and generated result tables are excluded from version control by `.gitignore`.

## Running the workflow

List the declared agents and tools:

```bash
cd code
python run_dieta_agentic.py --list-agents
python run_dieta_agentic.py --list-tools
```

Inspect the workflow without running the scientific modules:

```bash
python run_dieta_agentic.py --dry-run --variant full
```

Run the full workflow:

```bash
python run_dieta_agentic.py --variant full
```

Run a small subset when the underlying modules support `--limit`:

```bash
python run_dieta_agentic.py --variant full --max-records 10
```

## Ablation variants

The runner forwards ablation switches to the scientific modules through environment variables. Final Step4 and Step5 tables keep the locked formats; additional trace files are written under `code/dieta_core/results/ablation/<variant>/`.

```bash
python run_dieta_agentic.py --variant no_uncertainty
python run_dieta_agentic.py --variant no_feedback
python run_dieta_agentic.py --variant no_relation_consistency
python run_dieta_agentic.py --variant no_routing
```

Ablation traces include score components, vote information and feedback-round metadata when available.

## Score calculation

The Critic Agent follows the RUQA scoring pathway:

```text
Q(t) = P_entity_head × P_entity_tail × P_relation
s^(k)(t) = lambda_reflect(k) × Q^(k)(t) - lambda × U^(0)(t)
```

`U^(0)(t)` is the initial normalized relation-distribution entropy. The uncertainty term can be disabled only through the explicit ablation variant.

## Validation commands

```bash
cd code
python contract_smoke_test.py
PYTHONPATH=code pytest -q
python run_dieta_agentic.py --dry-run --variant full
```

`docs/VALIDATION.md` records the checks expected for a clean release.
## Copyright and use

Copyright (c) 2026 Huixin Yu and collaborators.

This repository is made publicly available for academic inspection and peer-review transparency. Unless otherwise stated, no permission is granted to reproduce, redistribute, sublicense, or create derivative works from the source code or associated materials before formal publication of the corresponding manuscript. A formal open-source license may be assigned after manuscript acceptance or publication.
