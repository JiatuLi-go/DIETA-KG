# Output contracts

The runtime can create additional intermediate artifacts, but the public Step4 and Step5 files must keep the historical layout.

## Step4 quality decision

`code/dieta_core/results/quality_decision/quality_decision_latest.xlsx`

```text
pmid, subject, subject_type, relation, object, object_type,
P_entity_head, P_entity_tail, P_relation, Q_t, U_t, Score_t,
quality_tier, lambda_reflect, Q_t_penalized, feedback_rounds, source_cluster
```

## Step5 grounding after feedback

`code/dieta_core/results/grounding_alignment/grounding_alignment_latest.xlsx`

```text
pmid, subject, relation, object, source_cluster, quality_tier, Score_t,
head_status, head_standard, head_similarity, tail_status, tail_standard,
tail_similarity, needs_feedback, feedback_completed
```

## Verification

```bash
cd code
python contract_smoke_test.py
python verify_output_contracts.py --show-contracts
```

A contract guard may rewrite the Excel table headers and column order. It does not rewrite candidate generation, scoring, grounding decisions or the JSON graph facts.
