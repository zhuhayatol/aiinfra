#!/usr/bin/env bash
set -e

# 切换到项目根目录
cd "$(dirname "$0")/.."

python -m tiny_lm.generation.generate "$@"