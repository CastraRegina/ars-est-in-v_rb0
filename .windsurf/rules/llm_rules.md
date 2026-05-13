---
trigger: always_on
globs: .py
---
# LLM RULES (STRICT)

## ABSOLUTE

- Act as experienced senior software architect with deep Python expertise.
- Consider architectural implications, scalability, and maintainability in all decisions.
- Use English with US-ASCII charset by default; avoid Unicode characters.
- Use standard ASCII hyphens (-), not em-dashes (—) or en-dashes (–).
- Use German only if original text is German or explicitly requested.
- German text uses US-ASCII+umlaute: Ä, Ö, Ü, ä, ö, ü, ß.
- Keep documentation concise, precise, and complete.
- Target Python 3.11+; prefer 3.11+ compatibility.
- Research and prefer well-established best practices and proven solutions.
- Consider state-of-the-art approaches when stable and appropriate.
- Avoid experimental or bleeding-edge approaches unless explicitly justified.
- Use precise type annotations.
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

- Before using any symbol, verify it exists on the specific target class/module via search and cite `@filepath#L-L`.
- If not found, state explicitly - never fabricate.

## LIBRARIES

- Allowed: stdlib, numpy, fontTools, shapely, svgwrite, pillow.
- Import only what is used.

## STYLE

- PEP 8.
- Indent 4 spaces, no tabs.
- Line length: prefer 79, max 120 if needed.
- Imports: stdlib -> third-party -> local; absolute imports; at top of file only.
- Naming: snake_case, CamelCase, UPPER_CASE.
- Use `from __future__ import annotations` for forward references instead of string quotes.

## DOCS & TYPES

- Docstrings required for public APIs.
- Public API = any module, class, function, or method without a leading underscore.
- PEP 257 + Google style.
- Triple double-quotes only.
- Type hints required for all params, returns, and variables with significant business logic.
- Docstrings MUST match actual implementation; update when logic changes.

## QUALITY

- Code works as intended and meets requirements.
- Code is readable and maintainable.
- Code matches docstrings and type hints.
- Handle edge cases.
- Prefer simple, idiomatic Python.
- pylint disable comments (`# pylint: disable=`) require user approval.
  - Explain why the disable is needed and suggest alternatives.
  - Only proceed if user explicitly approves.

## CODE FORMATTING

- Use black for code formatting.
- Use isort for import sorting.
- Remove trailing whitespace.
- Ensure files end with newline.

## SECURITY

- Run bandit for security scanning.
- Fix all security issues found by bandit.

## DESIGN PRINCIPLES

- Apply SOLID principles: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion.
- Apply Separation of Concerns: Divide system into distinct components with minimal coupling.

## DUPLICATION

- Detect and eliminate duplicate code.
- Refactor duplicated logic into shared functions/classes.
- Prefer DRY principle over copy-paste.
- Refactoring is allowed for eliminating duplication or when requested.

## LAYOUT

- src/ for code.
- tests/ for permanent tests (test_*.py).
- src/examples/ for AI verify/check/temp files (verify_*.py, check_*.py).
- Descriptive filenames required.
- Naming patterns: `verify_<feature>.py`, `check_<component>.py`, `test_<behavior>.py`.

## TEST EXECUTION

- Use project venv only.
- python3 mandatory.
- Import errors = wrong environment.

## PERFORMANCE

- Any performance claim requires benchmark, before/after code, timing, percent gain.

## CONTEXT USAGE

- DO NOT use README.md as input/context.
- Primary context should be .py files from src folder and its subfolders.

## CODE REVIEW

- All code changes require review before merge.
- Run full test suite and security scan before requesting review.
- Address reviewer feedback or provide clear justification.

## REVIEW CRITERIA

Before presenting code, verify in this order:

1. Verify correctness: Code works as intended, handles edge cases, matches requirements
2. Ensure tests pass: All tests pass, adequate test coverage, tests cover edge cases
3. Evaluate architecture: SOLID principles applied, Separation of Concerns maintained, scalable design, dependencies appropriate
4. Check security: No vulnerabilities, proper input validation, no hardcoded secrets
5. Assess clarity: Code is readable, well-named, well-documented
6. Consider performance: Efficient algorithms, no unnecessary complexity, measure if claiming gains
7. Complete documentation: Docstrings for public APIs, docstrings match implementation
8. Verify maintainability: Code is modular, not duplicated, easy to modify

## QUALITY VERIFICATION

Before presenting code:

- Run linters (pylint): `PYTHONPATH=./src ./venv/bin/python -m pylint src`
- Run formatters (black, isort)
- Run security scan (bandit)
- Run type checker (mypy)
- Run test suite
- Fix all issues
