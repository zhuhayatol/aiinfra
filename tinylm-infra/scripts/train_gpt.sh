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

NUM_GPUS="${NUM_GPUS:-1}"

echo "================================"
echo "TinyLM training"
echo "script_dir: $SCRIPT_DIR"
echo "project root: $PROJECT_ROOT"
echo "python: $(which python)"
echo "num_gpus: $NUM_GPUS"
echo "device information:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "================================"

if [ "$NUM_GPUS" -gt 1 ]; then
    echo "Launching DDP with $NUM_GPUS GPUs..."
    torchrun --standalone --nproc_per_node="$NUM_GPUS" -m tiny_lm.train.train_gpt2 "$@"
else
    echo "Launching single-process training..."
    python -m tiny_lm.train.train_gpt2 "$@"
fi

echo "Training finished."


#单卡：
# bash scripts/train_Gpt.sh
# 两卡 DDP：
# NUM_GPUS=2 bash scripts/train_Gpt.sh
# 四卡 DDP：
# NUM_GPUS=4 bash scripts/train_Gpt.sh