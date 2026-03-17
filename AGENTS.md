# AGENTS.md

## Codex Working Conventions

### Type Annotations

- When editing Python code, add explicit type annotations for new or modified functions, methods, and data structures.
- Prefer precise types over broad `dict`, `list`, or `Any` when the concrete shape is known.
- Add return type annotations for all functions and methods, including `-> None` for procedures.
- When a structured payload appears in multiple places, extract a reusable type alias instead of repeating loose container types.
- Keep runtime behavior unchanged when adding or refining type annotations unless the task explicitly requires a behavioral change.

### Docstrings

- Add docstrings for all data structures, functions, and classes that are added or modified.
- Use Google-style docstrings.
- Include type information inside docstrings as well as in Python signatures.
- In `Args`, use the form `name (type): description`.
- In `Returns`, use the form `Type: description`.
- In `Raises`, describe the concrete exception type and the condition.
- For data structures such as `dataclass` or `BaseModel`, document fields under `Attributes` using the form `name (type): description`.

### Consistency

- Keep docstrings and type annotations consistent with each other.
- If code formatting or pre-commit hooks rewrite docstring layout, keep the formatter's result and do not fight it manually.
- Prefer incremental, local improvements over unrelated large-scale annotation changes.
