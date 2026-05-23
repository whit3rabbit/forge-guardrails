# Parity Fixtures

`fixtures/python_golden.json` is generated from the Python reference submodule
and consumed by `tests/parity_tests.rs`.

Regenerate after intentional parity behavior changes:

```bash
uv run --project forge python tests/parity/generate_fixtures.py
```

Normal Rust test runs use the checked-in JSON and do not invoke Python.
