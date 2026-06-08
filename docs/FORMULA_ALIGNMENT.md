# Formula alignment

The quality-decision stage uses the same retention score implemented in the manuscript methods.

## Entity and relation quality

```text
Q(t) = p(e_h) × p(e_t) × p(r)
```

## Initial uncertainty

```text
U^(0)(t) = normalized Shannon entropy over the first-round relation-vote distribution
```

## Feedback-round calibration

```text
lambda_reflect(0) = 1
lambda_reflect(k) = 1 - beta_decay × (k / K_max),  k > 0
```

## Final retention score

```text
s^(k)(t) = lambda_reflect(k) × Q^(k)(t) - lambda × U^(0)(t)
```

The reference helper is `code/dieta_agentic/ruqa_formula.py`. The original scientific implementation remains in `code/dieta_core/quality_decision_agent.py`.
