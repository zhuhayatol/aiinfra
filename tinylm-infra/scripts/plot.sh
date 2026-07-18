#!/usr/bin/env bash
set -e

cd "$(dirname "$0")/.."

python -m tiny_lm.utils.plot "$@"