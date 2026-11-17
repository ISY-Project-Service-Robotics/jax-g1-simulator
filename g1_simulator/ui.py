"""Tk control panel for checkpoint selection and live simulator status."""

from __future__ import annotations

import logging
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from g1_simulator.checkpoints import discover_checkpoints
from g1_simulator.simulation import SimulationStatus, SimulationWorker

LOGGER = logging.getLogger(__name__)


class ControlPanel:
  def __init__(self, root: tk.Tk, checkpoint_root: Path) -> None:
    self.root = root
    self.worker = SimulationWorker(checkpoint_root)
    self.root.title("JAX G1 Simulator")
    self.root.minsize(480, 430)
    self.root.protocol("WM_DELETE_WINDOW", self.close)

    self._message = tk.StringVar(value="Starting simulator")
    self._policy = tk.StringVar(value="Not loaded")
    self._gamepad = tk.StringVar(value="Disconnected")
    self._frequency = tk.StringVar(value="0.0 Hz")
    self._sim_time = tk.StringVar(value="0.00 s")
    self._command = tk.StringVar(value="0.00, 0.00, 0.00")
    self._checkpoint = tk.StringVar()
    self._joint_labels: dict[str, tk.StringVar] = {}

    self._build(checkpoint_root)
    self.root.after(100, self._refresh_status)

  def _build(self, checkpoint_root: Path) -> None:
    container = ttk.Frame(self.root, padding=12)
    container.pack(fill=tk.BOTH, expand=True)
    container.columnconfigure(1, weight=1)

    ttk.Label(container, text="Checkpoint").grid(row=0, column=0, sticky="w")
    try:
      checkpoints = discover_checkpoints(checkpoint_root)
      names = [checkpoint.name for checkpoint in checkpoints]
    except (FileNotFoundError, OSError) as error:
      LOGGER.exception("Could not discover G1 checkpoints")
      names = []
      self._message.set(f"Checkpoint error: {error}")

    self._checkpoint.set(names[-1] if names else "")
    self._checkpoint_box = ttk.Combobox(
        container,
        textvariable=self._checkpoint,
        values=names,
        state="readonly" if names else "disabled",
    )
    self._checkpoint_box.grid(row=0, column=1, sticky="ew", padx=(8, 8))
    ttk.Button(
        container, text="Load checkpoint", command=self._load_checkpoint
    ).grid(row=0, column=2, sticky="ew")

    controls = ttk.Frame(container)
    controls.grid(row=1, column=0, columnspan=3, sticky="ew", pady=(12, 12))
    for column in range(4):
      controls.columnconfigure(column, weight=1)
    ttk.Button(controls, text="Start", command=self.worker.start).grid(
        row=0, column=0, sticky="ew", padx=(0, 4)
    )
    ttk.Button(controls, text="Pause", command=self.worker.pause).grid(
        row=0, column=1, sticky="ew", padx=4
    )
    ttk.Button(controls, text="Stop", command=self.worker.stop).grid(
        row=0, column=2, sticky="ew", padx=4
    )
    ttk.Button(controls, text="Reset", command=self.worker.reset).grid(
        row=0, column=3, sticky="ew", padx=(4, 0)
    )

    status = ttk.LabelFrame(container, text="Simulation", padding=8)
    status.grid(row=2, column=0, columnspan=3, sticky="ew")
    status.columnconfigure(1, weight=1)
    self._add_status(status, 0, "Status", self._message)
    self._add_status(status, 1, "Policy inference", self._policy)
    self._add_status(status, 2, "Gamepad", self._gamepad)
    self._add_status(status, 3, "Control frequency", self._frequency)
    self._add_status(status, 4, "Simulation time", self._sim_time)
    self._add_status(status, 5, "Command (x, y, yaw)", self._command)

    joints = ttk.LabelFrame(container, text="Joint positions (rad)", padding=8)
    joints.grid(row=3, column=0, columnspan=3, sticky="nsew", pady=(12, 0))
    joints.columnconfigure(1, weight=1)
    for row, name in enumerate(
        (
            "left_hip_pitch",
            "left_knee",
            "left_ankle_pitch",
            "right_hip_pitch",
            "right_knee",
            "right_ankle_pitch",
            "waist_yaw",
        )
    ):
      value = tk.StringVar(value="--")
      self._joint_labels[name] = value
      ttk.Label(joints, text=name.replace("_", " ")).grid(
          row=row, column=0, sticky="w", padx=(0, 16)
      )
      ttk.Label(joints, textvariable=value).grid(
          row=row, column=1, sticky="e"
      )

  @staticmethod
  def _add_status(
      parent: ttk.LabelFrame, row: int, title: str, value: tk.StringVar
  ) -> None:
    ttk.Label(parent, text=title).grid(row=row, column=0, sticky="w", padx=(0, 16))
    ttk.Label(parent, textvariable=value).grid(row=row, column=1, sticky="w")

  def _load_checkpoint(self) -> None:
    name = self._checkpoint.get()
    if name:
      self.worker.load_checkpoint(name)

  def _refresh_status(self) -> None:
    status = self.worker.status()
    self._apply_status(status)
    if self.root.winfo_exists():
      self.root.after(100, self._refresh_status)

  def _apply_status(self, status: SimulationStatus) -> None:
    self._message.set(status.message)
    self._policy.set(status.policy)
    self._gamepad.set(status.gamepad)
    self._frequency.set(f"{status.frequency_hz:.1f} Hz")
    self._sim_time.set(f"{status.sim_time:.2f} s")
    self._command.set(", ".join(f"{value:+.2f}" for value in status.command))
    readings = dict(status.joints)
    for name, value in self._joint_labels.items():
      if name in readings:
        value.set(f"{readings[name]:+.3f}")

  def close(self) -> None:
    self.worker.close()
    self.root.destroy()


def run() -> None:
  logging.basicConfig(
      level=logging.INFO,
      format="%(asctime)s %(levelname)s %(name)s: %(message)s",
  )
  project_root = Path(__file__).resolve().parent.parent
  checkpoint_root = project_root / "data" / "training_checkpoints"
  root = tk.Tk()
  ControlPanel(root, checkpoint_root)
  root.mainloop()
