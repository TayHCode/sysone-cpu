#!/usr/bin/env bash
set -eu
cd -- "$(dirname -- "$0")/.."
export PYTHONDONTWRITEBYTECODE=1
exec python3 formal/run.py "$@"
