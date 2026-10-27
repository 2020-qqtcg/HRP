#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
exec uv run --project environment/training --locked --no-sync python -u -m hrp_miniwob.benchmarks.miniwob_harnesses "$@"
