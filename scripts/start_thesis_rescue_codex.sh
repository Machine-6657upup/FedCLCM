#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROMPT_FILE="$ROOT_DIR/docs/thesis_rescue/11_BOOTSTRAP_PROMPT.txt"
CODEX_BIN="${CODEX_BIN:-}"
TORCH_BIN="/home/huangtu/miniconda3/envs/torch/bin"

if [ -d "$TORCH_BIN" ]; then
  export PATH="$TORCH_BIN:$PATH"
fi

if [ -z "$CODEX_BIN" ]; then
  if [ -x "$TORCH_BIN/codex" ]; then
    CODEX_BIN="$TORCH_BIN/codex"
  elif command -v codex >/dev/null 2>&1; then
    CODEX_BIN="$(command -v codex)"
  fi
fi

if [ -z "$CODEX_BIN" ]; then
  echo "codex command not found. Make sure Codex CLI is installed, or set CODEX_BIN explicitly."
  exit 1
fi

if [ ! -f "$PROMPT_FILE" ]; then
  echo "Bootstrap prompt file not found: $PROMPT_FILE"
  exit 1
fi

cd "$ROOT_DIR"
exec "$CODEX_BIN" "$(cat "$PROMPT_FILE")"
