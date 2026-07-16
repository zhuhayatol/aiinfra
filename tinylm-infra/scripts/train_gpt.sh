#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(
    cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
    pwd
)"

PROJECT_ROOT="$(
    cd -- "$SCRIPT_DIR/.."
    pwd
)"

cd "$PROJECT_ROOT"

echo "================================"
echo "TinyLM training"
echo "script_dir: $SCRIPT_DIR"
echo "project root: $PROJECT_ROOT"
echo "python: $(which python)"
echo "device information:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
echo "================================"

python -m tiny_lm.train.train_gpt2

echo "Training finished."