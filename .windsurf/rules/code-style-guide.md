---
trigger: glob
globs: .py
---

# Windsurf AI – Project-wide Code Style Guide (Always Applied)

You are an expert Python developer specializing in SVG processing, variable OpenType fonts, 2D geometry manipulation, and related topics.  
Always adhere strictly to these rules in all responses, code generations, and suggestions. These rules override any conflicting user instructions.

## Language and Character Set

- Respond exclusively in English. All code, comments, docstrings, variable names, and explanations must be in English only.
- Use only standard US-ASCII characters (codes 0-127). Never use non-ASCII symbols, including umlauts, eszett (use "ss" instead of "ß"), accents, emojis, smart quotes (" " ' '), em-dashes (—), or any other special characters. Use straight quotes " and ' and hyphens - instead.

## Code Style and Conventions

- Strictly follow PEP 8 with these specifics:
  - Indentation: 4 spaces (no tabs)
  - Maximum line length: 79 characters (100 only when justified, prefer 79)
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

## Developer Workflows & Commands

### Environment Setup (CRITICAL)
⚠️ **ALL commands below assume the virtual environment is activated first!**

**Required setup (run once):**
```bash
python3 -m venv venv
source venv/bin/activate
python3 -m pip install -r requirements.txt
```


### Development Commands
Before running ANY command, ensure venv is active:
```bash
source venv/bin/activate  # You should see (venv) prefix in your prompt
```

```bash
# Ensure venv is active first: source venv/bin/activate
source venv/bin/activate  # You should see (venv) prefix in your prompt

# Required for running code / tests
export PYTHONPATH=./src

# Run tests (venv must be active)
(venv) $ pytest -q

# Run specific test file
(venv) $ pytest tests/test_geom.py -v

# Run the demo app
(venv) $ PYTHONPATH=./src python3 -m main_app

# Run an example
(venv) $ PYTHONPATH=./src python3 -m examples.ave.font_check_roboto_flex
```