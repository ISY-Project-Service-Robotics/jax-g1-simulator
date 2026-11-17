"""SDL joystick input mapped to G1 velocity commands."""

from __future__ import annotations

import math
from typing import Any, ClassVar


def _apply_deadzone(value: float, deadzone: float) -> float:
  magnitude = abs(value)
  if magnitude <= deadzone:
    return 0.0
  return math.copysign((magnitude - deadzone) / (1.0 - deadzone), value)


def map_axes(
    left_x: float,
    left_y: float,
    right_x: float,
    *,
    deadzone: float = 0.08,
    forward_limit: float = 1.0,
    lateral_limit: float = 0.5,
    yaw_limit: float = 1.0,
) -> tuple[float, float, float]:
  """Map centered SDL axes to the trained policy's (forward, lateral, yaw)."""
  forward = -_apply_deadzone(left_y, deadzone) * forward_limit
  lateral = -_apply_deadzone(left_x, deadzone) * lateral_limit
  yaw = -_apply_deadzone(right_x, deadzone) * yaw_limit
  return forward, lateral, yaw


class Gamepad:
  """Poll one SDL joystick and report edge-triggered control buttons."""

  _BUTTON_ACTIONS: ClassVar[dict[int, str]] = {
      0: "reset",
      6: "stop",
      7: "pause",
  }

  def __init__(self) -> None:
    import pygame

    self._pygame = pygame
    pygame.init()
    pygame.joystick.init()
    self._joystick: Any | None = None
    self._previous_buttons: dict[int, bool] = {}
    self._connect()

  def _connect(self) -> None:
    if self._joystick is not None and self._joystick.get_init():
      return
    self._joystick = None
    if self._pygame.joystick.get_count() == 0:
      return
    self._joystick = self._pygame.joystick.Joystick(0)
    self._joystick.init()
    self._previous_buttons.clear()

  @property
  def name(self) -> str | None:
    return self._joystick.get_name() if self._joystick is not None else None

  def poll(
      self,
  ) -> tuple[tuple[float, float, float], set[str], str]:
    self._pygame.event.pump()
    self._connect()
    if self._joystick is None:
      return (0.0, 0.0, 0.0), set(), "Disconnected"

    joystick = self._joystick
    axis_count = joystick.get_numaxes()
    left_x = joystick.get_axis(0) if axis_count > 0 else 0.0
    left_y = joystick.get_axis(1) if axis_count > 1 else 0.0
    right_x_axis = 3 if axis_count > 3 else 2
    right_x = (
        joystick.get_axis(right_x_axis)
        if axis_count > right_x_axis
        else 0.0
    )

    pressed = set()
    for button, action in self._BUTTON_ACTIONS.items():
      is_down = button < joystick.get_numbuttons() and bool(
          joystick.get_button(button)
      )
      if is_down and not self._previous_buttons.get(button, False):
        pressed.add(action)
      self._previous_buttons[button] = is_down

    commands = map_axes(left_x, left_y, right_x)
    return commands, pressed, joystick.get_name()

  def close(self) -> None:
    if self._joystick is not None:
      self._joystick.quit()
    self._pygame.quit()
