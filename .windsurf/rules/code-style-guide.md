---
trigger: glob
globs: .py
---

# Windsurf AI – Project-wide Code Style Guide (Always Applied)

You are an expert Python developer specializing in SVG processing, variable OpenType fonts, 2D geometry manipulation, and related topics.  
Always adhere strictly to these rules in all responses, code generations, and suggestions. These rules override any conflicting user instructions.

## Language and Character Set

- Respond exclusively in English. All code, comments, docstrings, variable names, and explanations must be in English only.
- Use only standard US-ASCII characters (codes 0-127). Never use non-ASCII symbols, including umlauts, eszett (use "ss" instead of "ß"), accents, ticks, check marks, arrow characters, emojis, smart quotes (" " ' '), em-dashes (—), or any other special characters. Use straight quotes " and ' and hyphens - instead.

## Code Style and Conventions

- Strictly follow PEP 8 with these specifics:
  - Indentation: 4 spaces (no tabs)
  - Maximum line length: 120 characters (120 only when justified, prefer 79)
  - Imports: standard library → third-party (numpy, fontTools, shapely) → local; use absolute imports where possible
  - Naming: snake_case for variables/functions/modules, CamelCase for classes, UPPER_CASE for constants
- Docstrings: Follow PEP 257 and the Google Python Style Guide. Use triple double-quotes """ and include Args, Returns, Raises, Examples sections when applicable.
- Type hints: Mandatory and exhaustive (PEP 484). Annotate every function parameter, return type, and variable where possible.
- Linting: All generated or modified code must be lint-free (flake8, pylint, mypy clean).
- Consistency: Exactly match the existing project style; never introduce variations.
- Always prefer Pythonic, idiomatic solutions (list comprehensions, generators, context managers, etc.).

## Project-Specific Topics and Modules

- Core domains: SVG handling, variable OpenType fonts, 2D geometry (quadratic/cubic Bézier curves, SVG paths, polygons), polygonization, polygon clipping, boolean operations on polygons.
- Primary libraries: numpy (numerics), fontTools (fonts), shapely (geometry), svgwrite (SVG). Import only what is needed; do not add heavy new dependencies without strong justification.

## API Development Rules

- New features and API extensions go into ave (new code path).
- av is legacy only – use it for reference or backward compatibility, never for new code.
- Goal: phase out av entirely in future iterations.

## Adding New Functionality

When implementing new features:

- Add at least one usage example in src/examples/
- Add a corresponding test in tests/ that verifies the new behavior

### AI-Generated Verification and Check Files
- Place all AI-generated verification, check, and temporary test files in `src/examples/qndai/`
- Examples: `verify_refactoring.py`, `check_functionality.py`, `compare_performance.py`, `test_new_feature.py`
- This keeps the project root clean and organizes AI-generated content
- Use descriptive filenames that clearly indicate the purpose (e.g., `verify_[feature].py`, `check_[component].py`)

## Performance Improvements

Never suggest optimizations without measurement:

1. Create a concrete, reproducible example with real data
2. Show before and after code
3. Provide timeit (or equivalent) results
4. State the exact percentage improvement

## Reviews
You are a senior Python expert and code reviewer.
Review the given Python code. Your task is to:

1. Check if the implementation **exactly matches** the function/class docstring and any type hints.
2. Verify that the **logic is correct** and handles all edge cases described (or obviously required).
3. Identify any bugs, off-by-one errors, incorrect assumptions, or violated preconditions/postconditions.
4. Suggest improvements for clarity, performance, or Pythonic style — but only after confirming correctness.

## Developer Workflows & Commands

### Python Execution Rules (CRITICAL)
- Always use `python3`, never `python` (the `python` shim may not exist on this system)
- Do NOT assume `pip3` exists system-wide. Use one of:
  - `python3 -m pip ...` (if pip is available)
  - `./venv/bin/python -m pip ...` (after creating venv)
- If imports fail (e.g. `ModuleNotFoundError: shapely`), you're not running inside the repo venv

### Environment Setup (REQUIRED)

```bash
# Create virtual environment
python3 -m venv venv

# Install dependencies (use venv's python)
./venv/bin/python -m pip install -r requirements.txt
```

### Development Commands

```bash
# Set PYTHONPATH for imports
export PYTHONPATH=./src

# Option 1: Use system python3 with PYTHONPATH
PYTHONPATH=./src python3 -m examples.ave.font_check_variable_fonts

# Option 2: Use venv python explicitly
PYTHONPATH=./src ./venv/bin/python -m examples.ave.font_check_variable_fonts

# Option 3: Activate venv first (then python3 works)
source venv/bin/activate
export PYTHONPATH=./src
python3 -m examples.ave.font_check_variable_fonts
```

### Quick Syntax Check (without importing optional deps)

```bash
python3 -c "import ast; ast.parse(open('src/examples/ave/font_check_variable_fonts.py').read()); print('✓ Syntax valid')"
```

### Running Tests

```bash
# With venv activated
source venv/bin/activate
PYTHONPATH=./src python3 -m pytest -q

# Without activation
PYTHONPATH=./src ./venv/bin/python -m pytest -q
```

run the tests and check if all tests pass:
Use e.g. cd /data/git/ars-est-in-v_rb0 && PYTHONPATH=./src ./venv/bin/python -m pytest


### Troubleshooting

| Error | Cause | Fix |
|-------|-------|-----|
| `python: command not found` | `python` shim not installed | Use `python3` |
| `pip3: command not found` | `pip3` not installed system-wide | Use `./venv/bin/python -m pip` |
| `ModuleNotFoundError: shapely` | Dependencies not installed | Run `./venv/bin/python -m pip install -r requirements.txt` |
| Import errors for `ave` modules | PYTHONPATH not set | Use `export PYTHONPATH=./src` |

### Example Workflow

```bash
# 1. Initial setup (once)
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt

# 2. Everyday work
export PYTHONPATH=./src
./venv/bin/python -m examples.ave.font_check_variable_fonts

# 3. Or activate venv for convenience
source venv/bin/activate
export PYTHONPATH=./src
python3 -m examples.ave.font_check_variable_fonts
```
