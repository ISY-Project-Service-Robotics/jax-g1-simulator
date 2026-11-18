#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
PLAYGROUND_ROOT="${MUJOCO_PLAYGROUND_ROOT:-}"
PYTHON="${SIMULATOR_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

if [[ ! -x "$PYTHON" ]]; then
  printf 'Python executable not found: %s\nSet SIMULATOR_PYTHON to the training environment Python.\n' "$PYTHON" >&2
  exit 1
fi

if [[ -n "$PLAYGROUND_ROOT" && ! -d "$PLAYGROUND_ROOT/mujoco_playground" ]]; then
  printf 'MuJoCo Playground source not found: %s\nSet MUJOCO_PLAYGROUND_ROOT to its checkout.\n' "$PLAYGROUND_ROOT" >&2
  exit 1
fi

export PYTHONPATH="$PROJECT_ROOT${PLAYGROUND_ROOT:+:$PLAYGROUND_ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export JAX_PLATFORMS="${JAX_PLATFORMS:-cpu}"

exec "$PYTHON" -m g1_simulator
