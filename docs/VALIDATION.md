# Validation checklist

Run these checks before uploading a release or submitting code for inspection.

```bash
cd code
python contract_smoke_test.py
python verify_output_contracts.py --show-contracts
PYTHONPATH=code pytest -q
python run_dieta_agentic.py --dry-run --variant full
```

Expected properties:

1. Step4 Excel header matches the locked quality-decision contract.
2. Step5 Excel header matches the locked grounding-after-feedback contract.
3. Formula helper tests pass.
4. Agent cards and tool registry are importable.
5. The dry-run writes a workflow graph and final run-state snapshot without executing scientific modules.

When the full workflow is run against a live model endpoint, row values may differ across reruns because the remote model service can change. The contract checks validate schema, file path and formula pathway stability.
