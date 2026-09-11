#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
exec "$PYTHON_BIN" "$ROOT/tools/hilti_workflow/run_hilti_batch.py" "$@"
