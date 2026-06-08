# Change log

## Contract-locked release

- Preserves the historical Step4 and Step5 Excel output contracts.
- Adds a typed runtime layer with agent cards, registered tools, state graph execution, bounded handoffs and contract guards.
- Adds runtime ablation variants with environment-variable propagation to subprocess stages.
- Writes ablation traces under `results/ablation/<variant>/` without changing final result tables.
- Keeps raw data, model caches, vector indexes and generated outputs outside the repository by default.
