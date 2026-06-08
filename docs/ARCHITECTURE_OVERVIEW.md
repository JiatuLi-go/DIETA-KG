# Architecture overview

DIETA is a bounded multi-agent workflow for ontology-constrained graph population. The architecture separates scientific processing from runtime orchestration.

## Scientific stages

1. Semantic triage activates task routes.
2. Seed entity generation applies type-specific pretrained NER models.
3. Candidate entity extraction performs route-conditioned multi-prompt NER.
4. Candidate relation extraction generates legal relation candidates over type-permissible entity pairs.
5. The Critic stage applies RUQA scoring and retention decisions.
6. Grounding aligns retained entities to type-specific resources and retains valid open-world concepts.
7. Feedback refinement reruns selected boundary-ambiguity cases.
8. Contract guards normalize final public tables to the locked Step4 and Step5 layouts.

## Runtime layer

The runtime layer provides a small in-repository implementation of:

- agent metadata cards;
- registered tool calls;
- typed artifact references;
- deterministic state-graph execution;
- conditional handoff from grounding to feedback;
- audit snapshots and output-contract checks.

The runtime layer does not replace the scientific modules. It provides a reproducible execution surface and keeps final output formats stable.
