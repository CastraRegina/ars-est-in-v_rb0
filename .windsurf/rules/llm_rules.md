---
trigger: always_on
globs: .py
---
# LLM RULES (STRICT)

## ABSOLUTE

- English only.
- US-ASCII only.
- Use only " ' and -.
- Python only (python3 mandatory).
- After ANY code-affecting change, run the FULL test suite AFTER the change:
  Use `PYTHONPATH=./src ./venv/bin/python -m pytest -q`.
- Non-code-only edits (comments/docstrings/whitespace) do NOT require running tests.
- Every response that changes code MUST explicitly state that the full test suite was run and passed.

## DO NOT (HALLUCINATION PREVENTION)

- Do NOT invent APIs, functions, classes, modules, files, or paths.
- Do NOT assume behavior not stated in code or tests.
- Do NOT guess library capabilities.
- Do NOT change public behavior without tests.
- Do NOT silence errors with try/except unless required.
- Do NOT add dependencies without approval.
- Do NOT claim performance gains without measurements.
- Do NOT modify tests to hide failures.
- Do NOT refactor for style preferences or minor optimizations.
- Refactor only when: fixing bugs, improving testability, or explicitly requested.

## SYMBOL VERIFICATION

- Before referencing any function, class, file, or path, locate it via search/read and cite with `@filepath#L-L`.
- If not found, state that explicitly—never fabricate.

## LIBRARIES

- Allowed: stdlib, numpy, fontTools, shapely, svgwrite, pillow.
- Import only what is used.

## STYLE

- PEP 8.
- Indent 4 spaces, no tabs.
- Line length: prefer 79, max 120 if needed.
- Imports: stdlib -> third-party -> local; absolute imports.
- Naming: snake_case, CamelCase, UPPER_CASE.
- Imports MUST be at top of file, never inline or inside functions.
- Use `from __future__ import annotations` for forward references instead of string quotes.

## DOCS & TYPES

- Docstrings required for public APIs.
- Public API = any module, class, function, or method without a leading underscore.
- PEP 257 + Google style.
- Triple double-quotes only.
- Type hints required for all params, returns, and variables with significant business logic.
- Docstrings MUST match actual implementation; update when logic changes.

## QUALITY

- Code must be flake8-, pylint-, and mypy-clean.
- Lint cleanliness must be maintained; pylint MUST be runnable explicitly.
- Required pylint command:
  `PYTHONPATH=./src ./venv/bin/python -m pylint src`
- Match existing project style exactly.
- Code must match docstrings and type hints.
- Handle edge cases.
- Prefer simple, idiomatic Python.
- pylint disable comments (`# pylint: disable=`) require user approval.
- Explain why the disable is needed and suggest alternatives.
- Only proceed if user explicitly approves.

## DUPLICATION

- Detect and eliminate duplicate code.
- Refactor duplicated logic into shared functions/classes.
- Prefer DRY principle over copy-paste.
- Refactoring is allowed for eliminating duplication or when requested.

## LAYOUT

- src/ for code.
- tests/ for tests.
- src/examples/qndai/ for AI verify/check/temp files only.
- Descriptive filenames required.
- Naming patterns: `verify_<feature>.py`, `check_<component>.py`, `test_<behavior>.py`.

## TEST EXECUTION

- Use project venv only.
- python3 mandatory.
- Preferred command:
  `PYTHONPATH=./src ./venv/bin/python -m pytest -q`
- Plain pytest without -q is acceptable.
- Import errors = wrong environment.

## PERFORMANCE

- Any performance claim requires benchmark, before/after code, timing, percent gain.

## REVIEW ORDER

1. Correctness.
2. Tests.
3. Clarity.
4. Performance.
