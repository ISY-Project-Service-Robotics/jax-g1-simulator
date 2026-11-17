from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from g1_simulator.checkpoints import discover_checkpoints, read_network_config


class CheckpointTests(unittest.TestCase):
  def test_discovery_sorts_only_complete_numeric_checkpoints(self) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
      root = Path(temporary_directory)
      for name in ("0002", "0001", "incomplete"):
        checkpoint = root / name
        checkpoint.mkdir()
        if name != "incomplete":
          (checkpoint / "commit_success.txt").write_text("ok")
          (checkpoint / "ppo_network_config.json").write_text("{}")
      (root / "config.json").write_text("{}")

      self.assertEqual(
          [checkpoint.name for checkpoint in discover_checkpoints(root)],
          ["0001", "0002"],
      )

  def test_network_config_checks_actor_dimensions_and_normalization(self) -> None:
    with tempfile.TemporaryDirectory() as temporary_directory:
      checkpoint = Path(temporary_directory)
      config = {
          "action_size": 29,
          "normalize_observations": True,
          "observation_size": {"state": {"shape": [103]}},
      }
      (checkpoint / "ppo_network_config.json").write_text(
          json.dumps(config)
      )
      self.assertEqual(read_network_config(checkpoint), config)

      config["action_size"] = 28
      (checkpoint / "ppo_network_config.json").write_text(
          json.dumps(config)
      )
      with self.assertRaisesRegex(ValueError, "action size"):
        read_network_config(checkpoint)


if __name__ == "__main__":
  unittest.main()
