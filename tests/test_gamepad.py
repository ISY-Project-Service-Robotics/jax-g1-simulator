from __future__ import annotations

import unittest

from g1_simulator.gamepad import map_axes


class GamepadTests(unittest.TestCase):
  def test_neutral_sticks_produce_zero_command(self) -> None:
    self.assertEqual(map_axes(0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

  def test_stick_commands_use_training_ranges_and_robot_axes(self) -> None:
    self.assertEqual(
        map_axes(-1.0, -1.0, -1.0),
        (1.0, 0.5, 1.0),
    )
    self.assertEqual(
        map_axes(1.0, 1.0, 1.0),
        (-1.0, -0.5, -1.0),
    )

  def test_deadzone_suppresses_small_stick_drift(self) -> None:
    self.assertEqual(map_axes(0.04, -0.05, 0.02), (0.0, 0.0, 0.0))


if __name__ == "__main__":
  unittest.main()
