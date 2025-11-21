#!/bin/bash

# set flags for the script
# -e: exit immediately if a command in the script exits with a non-zero status
# -u: treat unset variables as an error when substituting
# -o pipefail: if any command in a pipeline fails (i.e. exits with a non-zero status), the entire pipeline will exit with a non-zero status
set -euo pipefail

# remove python cache files
find . -type f -name "*.pyc" -delete

# remove python extension cache (global)
rm -rf ~/.cache/pycache/

# remove VS Code Python Extension Cache (per extension)
rm -rf ~/.vscode/extensions/ms-python.python-*/pythonFiles/lib/

# remove caches of various tools
find . \
  \( -name "__pycache__" \
  -o -name ".pytest_cache" \
  -o -name ".mypy_cache" \
  -o -name ".ruff_cache" \
  -o -name ".coverage" \
  -o -name "htmlcov" \
  -o -name "build" \
  -o -name "dist" \
  -o -name "*.egg-info" \) \
  -exec rm -rf {} +

# remove VS Code workspace cache
rm -rf .vscode/.pytest_cache



