# JAX G1 Simulator

A desktop controller and 3D viewer for the Unitree G1 joystick policy trained
with MuJoCo Playground. The robot's actions come from the Brax PPO checkpoint;
the simulator does not implement a separate walking controller.

## Architecture

- `g1_simulator/checkpoints.py` discovers complete Orbax checkpoints and
  restores their saved Brax network and observation normalizer.
- `g1_simulator/simulation.py` runs the original
  `G1JoystickFlatTerrain` MJX environment and mirrors its state into the
  MuJoCo viewer.
- `g1_simulator/gamepad.py` maps a connected joystick to the policy's forward,
  lateral, and yaw commands.
- `g1_simulator/ui.py` provides checkpoint selection, simulation controls,
  and live status and joint readings.

The policy takes the original 103-element `state` observation (with its
checkpoint-saved normalization), outputs 29 actions, and the original
environment applies its 0.5 action scale at a 50 Hz control rate / 500 Hz
physics rate. The saved 216-element privileged observation is used by the
critic architecture but is not an actor input.

The checkpoint network config stores optional kernel initializers as `null`.
Some Brax releases mishandle those entries in `checkpoint.load_policy`; the
loader here uses Brax's own Orbax deserializer and network factory while
preserving the saved defaults, without editing the checkpoint files.

## Launch

The default launcher uses the training environment at `~/mujoco_playground`
and its `.venv`. The repository's `data/training_checkpoints` link should point
to the training run's `checkpoints` directory.

```bash
./launch.sh
```

The launcher defaults JAX to CPU to avoid selecting an incompatible CUDA plugin
in the existing environment. Set `JAX_PLATFORMS` to a working accelerator
platform to override this. To use different paths:

```bash
MUJOCO_PLAYGROUND_ROOT=/path/to/mujoco_playground \
SIMULATOR_PYTHON=/path/to/python ./launch.sh
```

The application opens a control panel and a MuJoCo viewer. Choose a checkpoint
and click **Load checkpoint**, connect a gamepad, then click **Start**.
Controls: left stick forward/lateral, right-stick horizontal for yaw, **A**
reset, **Start** pause/resume, and **Back** stop. Panel buttons provide the same
simulation controls. The stick mapping assumes SDL joystick axes 0/1 for the
left stick and axis 3 (or axis 2 if only three axes are available) for
right-stick horizontal.

If running in a different Python environment, install the same dependencies as
the training checkout (MuJoCo Playground, JAX, Brax, Orbax, pygame, and
Tkinter), then set `MUJOCO_PLAYGROUND_ROOT` and `SIMULATOR_PYTHON` as above.
