# Contributing

This project is currently in early development. Keep the core engine conservative and deterministic.

## Development setup

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
python -m pip install -e ".[dev]"
pytest -q
ruff check src tests
```

## Design rules

1. Never mutate the source dataset during profiling.
2. A suspicious value is not automatically an incorrect value.
3. Ambiguous transformations must require review.
4. Research structure (subjects, visits, specimens, events) must not be flattened into generic duplicate logic.
5. AI may propose structured actions later, but deterministic code executes them.
6. Add a regression test for every bug fix or new detector.
