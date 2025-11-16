## Quick summary for AI coding agents

This repo renders and manipulates fonts and SVGs using Python. Key high-level components live under `src/av` (older API) and `src/ave` (newer API). Read `README.md` for extensive project notes; this file captures the essentials an AI needs to be productive immediately.

## Big picture
- Purpose: create SVG pages, convert font glyphs to SVG paths and geometry, and export SVG/SVGZ outputs.
- Major components:
  - `src/ave/` — newer API (AvSvgPage, AvFont, AvGlyph, AvLetter, AvSvgPath). Look at `src/ave/page.py`, `src/ave/glyph.py`, `src/ave/svgpath.py`.
  - `src/av/` — existing/older API with similar responsibilities (`src/av/page.py`, `src/av/glyph.py`, `src/av/helper.py`). Tests and examples import these modules, so maintain compatibility where needed.
  - `src/examples/` — runnable examples that are used by tests.

## Important patterns & conventions (do not invent alternatives)
- Coordinate systems: SVG canvas is built with a viewBox scaled so that often "1" equals viewbox width. Many helpers flip the y-axis so internal coordinates are left-to-right, bottom-to-top (see `AvSvgPage` / `AvPageSvg`). Preserve this convention when adding geometry.
- SVG handling: use `svgwrite` and the Inkscape extension for layers. Code expects layer groups named `main`, `debug` etc.; prefer using the helper methods that add to layers rather than manually composing groups.
- Font handling: fonts are manipulated via `fontTools`. Variable fonts are instantiated with `instancer` when needed. Glyph geometry is fetched via `SVGPathPen` then converted to absolute SVG via `AvSvgPath.convert_relative_to_absolute`.
- Path math: `src/ave/svgpath.py` contains utilities to beautify, convert relative→absolute, and apply affine transforms to path strings; use those helpers when transforming glyph paths.
- Tests: many tests simply run example scripts to smoke-test functionality. Do not assume heavy test coverage — add focused unit tests for new behavior.

## Developer workflows & commands
- Create venv and install requirements:
  - python3 -m venv venv
  - . venv/bin/activate
  - python3 -m pip install -r requirements.txt
- Important environment detail: the project expects `src` on `PYTHONPATH`. Set it before running code or tests:
  - export PYTHONPATH=./src
  - pytest -q   # runs tests under ./tests (tests import modules from examples and src)
  - PYTHONPATH=./src python3 -m main_app   # runs the simple main that writes an example SVG
  - PYTHONPATH=./src python3 -m examples.ave.font_check_roboto_flex  # run an example module

## Files & locations to inspect for changes
- `README.md` — high level and detailed developer notes (long, authoritative source). Use it as the default reference for configuration and environment.
- `requirements.txt` — the full dependency list (heavy on font/svg toolchain). Avoid adding big new dependencies without justification.
- `src/ave/*.py` — preferred new code path.
- `src/av/*.py` — legacy API used by tests/examples; ensure backward compatibility if you change public behaviour.
- `src/examples/*` and `tests/*` — use these for smoke tests and to mirror expected usage patterns.

## Integration points & runtime assumptions
- External resources: fonts are expected under `fonts/` (large variable TTFs). Examples reference concrete filenames — avoid hardcoding new fonts without updating examples/tests.
- Output: SVG and compressed SVGZ are produced by `save_as(...)` helpers. Use the same `include_debug_layer` and `pretty` flags when adding examples for consistency.

## What reviewers should check when an AI edits code
- Does the change preserve the viewbox / y-axis flip semantics? (Look at page construction in `page.py`.)
- Are paths kept as absolute coordinates before transforms? Use `AvSvgPath.convert_relative_to_absolute` when needed.
- If you touch font handling, ensure variable-font instantiation flow using `fontTools.varLib.instancer` remains intact and that `unitsPerEm`-based scaling is used.
- Keep imports relative to `src` layout and do not assume package installation — tests and examples run with `PYTHONPATH=./src`.

## Minimal examples to show how to run or transform code
- Transform a glyph to page coordinates (concept):
  - ttfont = TTFont("fonts/<...>.ttf")
  - avfont = AvFont(ttfont, AvGlyphFactory())
  - glyph = avfont.fetch_glyph('A')
  - letter = AvLetter(glyph, xpos, ypos, scale)
  - path = AvSvgPath.transform_path_string(glyph.svg_path_string(), letter.trafo)

## If something is unclear
- To extend the API or to add new features use `ave` (new). Use `av` (legacy) only for reference.
  In future iterations we can phase out `av` entirely.
- When adding new functionality, ensure that at least one example script in `src/examples/` uses it, and that a corresponding test in `tests/` verifies the behavior.
- If a new dependency is needed, justify it and add it to `requirements.txt` and `README.md` with a short note.
