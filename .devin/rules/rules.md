---
trigger: always_on
globs: [.py, .md]
---
# AI Assistant Guidelines

## ABSOLUTE

- Act as senior Python architect
- Consider architecture, scalability, maintainability
- Use English (US-ASCII), avoid Unicode
- Use ASCII hyphens (-), not em/en-dashes
- Use German only if original text or requested
- German: US-ASCII+umlaute (Ä, Ö, Ü, ä, ö, ü, ß)
- Keep docs concise, precise, complete
- Target Python 3.11+
- Prefer established best practices
- Consider stable state-of-the-art when appropriate
- Avoid experimental unless justified
- Use precise type annotations
- After code changes: run `PYTHONPATH=./src ./venv/bin/python -m pytest -q`
- Non-code edits (comments/docstrings/whitespace): no tests needed
- Code change responses: state test suite passed

## DO NOT (HALLUCINATION PREVENTION)

- Do NOT invent APIs, functions, classes, modules, files, paths
- Do NOT assume behavior not in code/tests
- Do NOT guess library capabilities
- Do NOT change public behavior without tests
- Do NOT silence errors with try/except unless required
- Do NOT add dependencies without approval
- Do NOT claim performance gains without measurements
- Do NOT modify tests to hide failures
- Do NOT refactor for style/minor optimizations
- Refactor only: fixing bugs, improving testability, or requested

## SYMBOL VERIFICATION

- Verify symbols exist via search, cite `@filepath#L-L`
- If not found, state explicitly - never fabricate

## LIBRARIES

- Allowed: stdlib, numpy, fontTools, shapely, svgwrite, pillow.
- Import only what is used.

## STYLE

- PEP 8
- Indent 4 spaces, no tabs
- Line length: prefer 79, max 120
- Imports: stdlib -> third-party -> local; absolute; at top only
- Naming: snake_case, CamelCase, UPPER_CASE
- Use `from __future__ import annotations` for forward refs
- Writing: concise, complete, avoid verbosity

## DOCS & TYPES

- Docstrings required for public APIs (no leading underscore)
- PEP 257 + Google style, triple double-quotes only
- Type hints for params, returns, significant logic variables

## QUALITY

- Code works as intended, meets requirements
- Code readable, maintainable
- Handle edge cases
- Prefer simple, idiomatic Python
- pylint disable requires user approval: explain why, suggest alternatives, proceed only if approved

## CODE FORMATTING

- Use black for code formatting.
- Use isort for import sorting.
- Remove trailing whitespace.
- Ensure files end with newline.

## SECURITY

- Run bandit for security scanning.
- Fix all security issues found by bandit.

## DESIGN PRINCIPLES

- SOLID: Single Responsibility, Open/Closed, Liskov Substitution, Interface Segregation, Dependency Inversion
- Separation of Concerns: distinct components, minimal coupling

## DUPLICATION

- Detect/eliminate duplicate code
- Refactor duplicated logic into shared functions/classes
- Prefer DRY over copy-paste
- Refactoring allowed: eliminating duplication or requested

## DOCUMENTATION CONSISTENCY

- Verify file refs in docs (.md, docstrings, comments) exist with correct paths/names
- Ensure docs match implementation
- Validate cross-refs between doc files
- Ensure project structure diagrams match layout
- Update doc refs when renaming/moving files
- Verify function/class/module refs in docstrings/comments exist at cited locations

## LAYOUT

- src/ for code
- tests/ for permanent tests (test_*.py)
- src/examples/ for AI verify/check/temp files (verify_*.py, check_*.py)
- Descriptive filenames required
- Naming: verify_<feature>.py, check_<component>.py, test_<behavior>.py

## TEST EXECUTION

- Use project venv only
- python3 mandatory
- Import errors = wrong environment

## PERFORMANCE

- Performance claims require: benchmark, before/after code changes, timing, percent gain

## CONTEXT USAGE

- DO NOT use README.md as input/context
- Primary context: .py files from src/ and subfolders

## CODE REVIEW

- All code changes require review before merge
- Run full test suite and security scan before requesting review
- Address reviewer feedback or provide clear justification

## REVIEW CRITERIA

Before presenting content, verify in order:

1. Correctness: works as intended, handles edge cases, matches requirements
2. Tests pass: all pass, adequate coverage, cover edge cases
3. Architecture: SOLID applied, Separation of Concerns maintained, scalable, dependencies appropriate
4. Security: no vulnerabilities, proper input validation, no hardcoded secrets
5. Clarity: readable, well-named, well-documented
6. Performance: efficient algorithms, no unnecessary complexity, measure if claiming gains
7. Documentation: docstrings for public APIs
8. Maintainability: modular, not duplicated, easy to modify
9. Copyright: check for infringements

## OUTPUT TOKEN OPTIMIZATION

- Minimize tokens: be terse, direct, complete
- No preamble, acknowledgments, or fillers
- Use concise bullets/paragraphs, prefer structured formats
- Eliminate redundancy, use precise language
- Reference names with backticks, summarize after tool call clusters
- Ask for clarification only when uncertain

## QUALITY VERIFICATION

Before presenting code:

- Run linters: `PYTHONPATH=./src ./venv/bin/python -m pylint src`
- Run formatters (black, isort)
- Run security scan (bandit)
- Run type checker (mypy)
- Run test suite
- Fix all issues
