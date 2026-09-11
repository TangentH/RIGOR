#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$REPO_ROOT/tools/hilti_workflow/run_hilti_rosbag_to_reconstruction.py" "$@"
