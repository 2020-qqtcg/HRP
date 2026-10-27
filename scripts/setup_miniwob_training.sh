#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
if [[ ! -f environment/vendor/verl/pyproject.toml ]]; then
    echo "The vendored verl source is required at environment/vendor/verl." >&2
    exit 1
fi
uv sync --project environment/training --locked
uv run --project environment/training hrp-miniwob fetch
uv run --project environment/training playwright install --with-deps chromium
uv run --project environment/training --locked --no-sync \
    python -c 'from hrp_miniwob.v2.assets import verify_assets; verify_assets(fetch=True)'
