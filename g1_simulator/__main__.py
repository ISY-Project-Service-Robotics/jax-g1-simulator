"""Start the G1 simulator desktop application."""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from g1_simulator.ui import run

if __name__ == "__main__":
  run()
