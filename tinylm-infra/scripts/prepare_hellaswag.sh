#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python -m tiny_lm.eval.prepare_hellaswag "$@"