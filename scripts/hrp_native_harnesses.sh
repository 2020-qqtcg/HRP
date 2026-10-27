#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT="${1:?Usage: hrp_native_harnesses.sh OUTPUT [native_experiment arguments]}"
shift
exec bash "$PROJECT_ROOT/scripts/eval_native_harnesses.sh" "$OUTPUT" --mode hrp "$@"
