#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> SentinelAI devcontainer post-create"

git config core.autocrlf input 2>/dev/null || true

echo "==> uv sync (project deps)"
uv sync

echo "==> CPU-only PyTorch (avoid CUDA wheels)"
uv pip install --python "$UV_PROJECT_ENVIRONMENT/bin/python" \
  torch --index-url https://download.pytorch.org/whl/cpu

echo "==> Download SKAB dataset"
uv run python -m sentinelai.data.download

echo "==> Smoke test"
uv run pytest -q

echo "==> Jupyter kernel"
uv run python -m ipykernel install --user --name sentinelai --display-name "SentinelAI (uv)"

echo ""
echo "Dev container ready."
echo "  uv run pytest"
echo "  uv run jupyter lab notebooks/"
