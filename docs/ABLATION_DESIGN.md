# Ablation design

Ablation variants are selected through `run_dieta_agentic.py --variant <name>`. The runner exports the selected variant as environment variables before calling the scientific modules, so subprocess-based stages receive the same configuration.

## Variants

| Variant | Runtime effect |
|---|---|
| `full` | Default DIETA workflow. |
| `no_uncertainty` | Sets the uncertainty penalty weight to zero. |
| `no_feedback` | Skips the feedback controller and limits feedback iteration. |
| `no_relation_consistency` | Disables NER/RE vote-threshold filtering by setting `TAU=0` and `DELTA=0`. |
| `no_routing` | Activates all six route slots rather than applying semantic triage filtering. |

## Additive trace outputs

Trace files are written under:

```text
code/dieta_core/results/ablation/<variant>/
```

The trace can contain:

```text
variant_id, stage, pmid, topic, subject, relation, object,
votes, raw_votes, vote_entropy,
P_entity_head, P_entity_tail, P_relation,
Q_t, U_t, lambda_reflect, Q_t_penalized, Score_t,
quality_tier, feedback_rounds
```

The final Step4 and Step5 files are not extended with ablation-only columns.
