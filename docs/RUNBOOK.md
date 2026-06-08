# Runbook

## 1. Prepare resources

Place files in the expected project directories:

```text
code/dieta_core/data/
code/dieta_core/prompts/
code/dieta_core/本体/
code/dieta_core/grounding_data/
```

## 2. Configure runtime

```bash
cp .env.example .env
```

Set API endpoint, model name and sampling flag in `.env` or your shell environment.

## 3. Inspect runtime declarations

```bash
cd code
python run_dieta_agentic.py --list-agents
python run_dieta_agentic.py --list-tools
```

## 4. Run checks

```bash
python contract_smoke_test.py
PYTHONPATH=code pytest -q
```

## 5. Run workflow

```bash
python run_dieta_agentic.py --variant full
```

## 6. Export ablation trace from existing results

```bash
python export_ablation_trace.py --variant full
```

## 7. Clean before upload

```bash
python sanitize_release.py
```

Review the repository manually after cleaning. Generated results and private resources should not be committed.
