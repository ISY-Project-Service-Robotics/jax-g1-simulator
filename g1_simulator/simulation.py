"""MJX policy execution and live MuJoCo viewer synchronization."""

from __future__ import annotations

import json
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import mujoco
import mujoco.viewer

from g1_simulator.checkpoints import Checkpoint, discover_checkpoints, load_policy
from g1_simulator.gamepad import Gamepad

LOGGER = logging.getLogger(__name__)
CONTROL_DT = 0.02


@dataclass(frozen=True)
class SimulationStatus:
  message: str = "Initializing"
  running: bool = False
  checkpoint: str = "None"
  policy: str = "Not loaded"
  gamepad: str = "Disconnected"
  frequency_hz: float = 0.0
  sim_time: float = 0.0
  command: tuple[float, float, float] = (0.0, 0.0, 0.0)
  joints: tuple[tuple[str, float], ...] = field(default_factory=tuple)


class G1Simulation:
  """Wrap the original registered environment and its trained actor."""

  def __init__(self, checkpoint_root: Path) -> None:
    import jax
    import jax.numpy as jnp
    from ml_collections import config_dict
    from mujoco_playground import registry

    self._jax = jax
    self._jnp = jnp
    self.checkpoint_root = checkpoint_root
    config_path = checkpoint_root / "config.json"
    if not config_path.is_file():
      raise FileNotFoundError(f"Training environment config not found: {config_path}")

    config = config_dict.ConfigDict(
        json.loads(config_path.read_text(encoding="utf-8"))
    )
    saved_impl = config.impl
    if saved_impl == "warp" and not any(
        device.platform in ("gpu", "cuda") for device in jax.devices()
    ):
      config.impl = "jax"
      self.backend_message = (
          "CPU MJX selected; the checkpoint was trained with Warp."
      )
    else:
      self.backend_message = f"MJX {config.impl} backend."

    self.env = registry.load("G1JoystickFlatTerrain", config=config)
    self._key = jax.random.PRNGKey(0)
    self._reset_fn = jax.jit(self.env.reset)
    self._step_fn = jax.jit(self.env.step)
    self.state = self._reset()
    self.visual_data = mujoco.MjData(self.env.mj_model)
    self.policy_name = "Not loaded"
    self._policy_fn: Any | None = None
    self._command = jnp.zeros(3, dtype=jnp.float32)

    warmup_action = jnp.zeros(self.env.action_size, dtype=jnp.float32)
    warmup_state = self._step_fn(self.state, warmup_action)
    warmup_state.data.time.block_until_ready()
    self.state = self._reset()
    self.update_visual_data()

  def _reset(self):
    self._key, reset_key = self._jax.random.split(self._key)
    state = self._reset_fn(reset_key)
    state.data.time.block_until_ready()
    self._command = self._jnp.zeros(3, dtype=self._jnp.float32)
    return state

  def reset(self) -> None:
    self.state = self._reset()
    self.update_visual_data()

  def load_checkpoint(self, checkpoint: Checkpoint) -> None:
    candidate = load_policy(checkpoint.path)
    policy_fn = self._jax.jit(candidate)
    self._key, policy_key = self._jax.random.split(self._key)
    actions, _ = policy_fn(self.state.obs, policy_key)
    actions.block_until_ready()
    if actions.shape != (self.env.action_size,):
      raise ValueError(
          f"Checkpoint produced actions with shape {actions.shape}; "
          f"expected ({self.env.action_size},)."
      )
    self._policy_fn = policy_fn
    self.policy_name = checkpoint.name

  def step(self, command: tuple[float, float, float]) -> None:
    if self._policy_fn is None:
      raise RuntimeError("Load a policy checkpoint before starting the simulation.")

    self._command = self._jnp.asarray(command, dtype=self._jnp.float32)
    info = dict(self.state.info)
    info["command"] = self._command
    observation = dict(self.state.obs)
    observation["state"] = observation["state"].at[9:12].set(self._command)
    self.state = self.state.replace(info=info, obs=observation)

    self._key, policy_key = self._jax.random.split(self._key)
    action, _ = self._policy_fn(self.state.obs, policy_key)
    self.state = self._step_fn(self.state, action)
    self.state.data.time.block_until_ready()
    self.update_visual_data()

  def update_visual_data(self) -> None:
    qpos = self._jax.device_get(self.state.data.qpos)
    qvel = self._jax.device_get(self.state.data.qvel)
    ctrl = self._jax.device_get(self.state.data.ctrl)
    self.visual_data.qpos[:] = qpos
    self.visual_data.qvel[:] = qvel
    self.visual_data.ctrl[:] = ctrl
    self.visual_data.time = float(self.state.data.time)
    mujoco.mj_forward(self.env.mj_model, self.visual_data)

  def joint_readings(self) -> tuple[tuple[str, float], ...]:
    model = self.env.mj_model
    positions = self.visual_data.qpos
    readings = []
    for joint_id in range(1, model.njnt):
      name = mujoco.mj_id2name(
          model, mujoco.mjtObj.mjOBJ_JOINT, joint_id
      ) or f"joint_{joint_id}"
      qpos_address = model.jnt_qposadr[joint_id]
      readings.append((name, float(positions[qpos_address])))
    return tuple(readings)

  @property
  def sim_time(self) -> float:
    return float(self.state.data.time)

  @property
  def command(self) -> tuple[float, float, float]:
    return tuple(float(value) for value in self._command)


class SimulationWorker:
  """Own the MJX loop, gamepad, viewer, and cross-thread status."""

  def __init__(self, checkpoint_root: Path) -> None:
    self.checkpoint_root = checkpoint_root
    self._commands: queue.Queue[tuple[str, str | None]] = queue.Queue()
    self._shutdown = threading.Event()
    self._status_lock = threading.Lock()
    self._status = SimulationStatus()
    self._thread = threading.Thread(
        target=self._run, name="g1-simulation", daemon=True
    )
    self._thread.start()

  def start(self) -> None:
    self._commands.put(("start", None))

  def pause(self) -> None:
    self._commands.put(("pause", None))

  def stop(self) -> None:
    self._commands.put(("stop", None))

  def reset(self) -> None:
    self._commands.put(("reset", None))

  def load_checkpoint(self, name: str) -> None:
    self._commands.put(("load", name))

  def close(self) -> None:
    self._shutdown.set()
    self._thread.join(timeout=5.0)

  def status(self) -> SimulationStatus:
    with self._status_lock:
      return self._status

  def _set_status(self, **changes: Any) -> None:
    with self._status_lock:
      values = self._status.__dict__ | changes
      self._status = SimulationStatus(**values)

  def _run(self) -> None:
    gamepad: Gamepad | None = None
    try:
      self._set_status(message="Loading original G1 environment")
      simulation = G1Simulation(self.checkpoint_root)
      gamepad = Gamepad()
      self._set_status(
          message=simulation.backend_message,
          gamepad=gamepad.name or "Disconnected",
      )
      checkpoints = discover_checkpoints(self.checkpoint_root)
      if not checkpoints:
        raise FileNotFoundError(
            f"No complete PPO checkpoints found in {self.checkpoint_root}"
        )

      running = False
      selected_checkpoint = checkpoints[-1]
      policy_loaded = False
      frame_count = 0
      fps_started = time.perf_counter()

      with mujoco.viewer.launch_passive(
          simulation.env.mj_model, simulation.visual_data
      ) as viewer:
        self._load_policy(simulation, selected_checkpoint)
        policy_loaded = True
        self._set_status(
            checkpoint=selected_checkpoint.name,
            policy=f"Ready ({selected_checkpoint.name})",
        )

        while viewer.is_running() and not self._shutdown.is_set():
          for action, value in self._drain_commands():
            if action == "start":
              if policy_loaded:
                running = True
                self._set_status(message="Running", running=True)
              else:
                self._set_status(message="Load a policy checkpoint first")
            elif action == "pause":
              running = False
              self._set_status(message="Paused", running=False)
            elif action == "stop":
              running = False
              self._set_status(message="Stopped", running=False)
            elif action == "reset":
              simulation.reset()
              self._set_status(message="Robot reset")
            elif action == "load" and value is not None:
              selected_checkpoint = self._find_checkpoint(checkpoints, value)
              self._set_status(
                  message=f"Loading checkpoint {selected_checkpoint.name}",
                  policy="Loading",
              )
              self._load_policy(simulation, selected_checkpoint)
              policy_loaded = True
              self._set_status(
                  message=f"Loaded checkpoint {selected_checkpoint.name}",
                  checkpoint=selected_checkpoint.name,
                  policy=f"Ready ({selected_checkpoint.name})",
              )

          command, buttons, gamepad_name = gamepad.poll()
          self._set_status(gamepad=gamepad_name)
          if "reset" in buttons:
            simulation.reset()
            self._set_status(message="Robot reset")
          if "pause" in buttons:
            running = not running
            self._set_status(
                message="Running" if running else "Paused",
                running=running,
            )
          if "stop" in buttons:
            running = False
            self._set_status(message="Stopped", running=False)

          tick_started = time.perf_counter()
          if running:
            simulation.step(command)
            frame_count += 1

          viewer.sync()
          now = time.perf_counter()
          if now - fps_started >= 1.0:
            self._set_status(
                frequency_hz=frame_count / (now - fps_started),
                sim_time=simulation.sim_time,
                command=simulation.command,
                joints=simulation.joint_readings(),
            )
            frame_count = 0
            fps_started = now

          delay = CONTROL_DT - (time.perf_counter() - tick_started)
          if delay > 0:
            time.sleep(delay if running else min(delay, 1.0 / 60.0))

      self._set_status(message="Viewer closed", running=False)
    except Exception as error:
      LOGGER.exception("G1 simulator stopped unexpectedly")
      self._set_status(message=f"Error: {error}", policy="Error", running=False)
    finally:
      if gamepad is not None:
        gamepad.close()

  def _load_policy(
      self, simulation: G1Simulation, checkpoint: Checkpoint
  ) -> None:
    simulation.load_checkpoint(checkpoint)

  @staticmethod
  def _find_checkpoint(checkpoints: list[Checkpoint], name: str) -> Checkpoint:
    for checkpoint in checkpoints:
      if checkpoint.name == name:
        return checkpoint
    raise ValueError(f"Checkpoint {name!r} is not available.")

  def _drain_commands(self) -> list[tuple[str, str | None]]:
    commands = []
    while True:
      try:
        commands.append(self._commands.get_nowait())
      except queue.Empty:
        return commands
