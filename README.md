# JAX G1 Simulator

A desktop controller and 3D viewer for a Unitree G1 joystick policy trained with MuJoCo Playground. The robot is controlled by a Brax PPO policy restored from a training checkpoint; this project does not provide a separate walking controller.

## Requirements

- Linux desktop session with a working graphical display (Tkinter and the MuJoCo viewer need a GUI).
- A MuJoCo Playground checkout and the Python dependencies listed in `requirements.txt`.
- A complete G1 joystick PPO training run and its checkpoint directory.
- A connected SDL-compatible gamepad for joystick control. The control panel can also start, pause, stop, and reset the simulation without a gamepad.

Tkinter must also be available to the Python installation (on some Linux distributions it is provided by a system package).

## Set Up Your Checkpoints

The simulator looks for training files in `data/training_checkpoints` inside this repository. Point that path at the **training run's checkpoint root**. That root must contain the environment's `config.json` and one or more numbered checkpoint directories, for example:

```text
/path/to/training-run/checkpoints/
  config.json
  100000000/
    commit_success.txt
    ppo_network_config.json
    ... Orbax checkpoint data ...
  200000000/
    commit_success.txt
    ppo_network_config.json
    ... Orbax checkpoint data ...
```

Use the complete checkpoint output from training; do not copy only the JSON files. The simulator discovers numbered directories that contain both `commit_success.txt` and `ppo_network_config.json`, and orders them by training step. For example, if `data/training_checkpoints` does not already exist:

```bash
ln -s /path/to/training-run/checkpoints data/training_checkpoints
```

Replace the example path with the checkpoint root produced by your run. The same root-level `config.json` is used to recreate the training environment. The path is fixed in `g1_simulator/ui.py`; `MUJOCO_PLAYGROUND_ROOT` configures the source checkout, not the checkpoint directory.

### Checkpoint compatibility

This application currently loads the `G1JoystickFlatTerrain` environment and expects a compatible Brax PPO actor checkpoint with:

- A 103-element `state` actor observation.
- 29 policy actions.
- Observation normalization enabled and saved with the policy.
- The network configuration, normalizer, and Orbax state from the same training run.

Checkpoints for another task, a different observation or action layout, or a policy without the saved normalizer are not compatible. The 216-element privileged observation belongs to the critic; it is not the actor input.

## Launch

Create the local virtual environment and install its dependencies once:

```bash
uv venv --python 3.12 --prompt jax-g1-simulation .venv
MUJOCO_PLAYGROUND_ROOT="${MUJOCO_PLAYGROUND_ROOT:-$HOME/mujoco_playground}"
uv pip install --python .venv/bin/python -r requirements.txt -e "$MUJOCO_PLAYGROUND_ROOT"
```

Then, from the repository root, start the simulator:

```bash
./launch.sh
```

The launcher uses `.venv/bin/python` by default. To use a different MuJoCo Playground checkout or Python executable:

```bash
MUJOCO_PLAYGROUND_ROOT=/path/to/mujoco_playground \
SIMULATOR_PYTHON=/path/to/python ./launch.sh
```

The launcher defaults `JAX_PLATFORMS` to `cpu`, which is useful when the installed JAX environment has no compatible accelerator. If your JAX and MuJoCo Playground setup supports an accelerator, you can override it, for example:

```bash
JAX_PLATFORMS=your-platform ./launch.sh
```

Use the platform name supported by your installed JAX version. When `MUJOCO_PLAYGROUND_ROOT` is set, the launcher adds that checkout and this repository to `PYTHONPATH`. Otherwise, MuJoCo Playground is loaded from the installed dependencies.

## Use The Simulator

1. Start the application and wait for the control panel and MuJoCo viewer to initialize.
2. Select a checkpoint in the dropdown and click **Load checkpoint**. The latest discovered step is selected by default.
3. Click **Start** in the control panel, or use the gamepad's Start button to resume.
4. Use **Pause**, **Stop**, and **Reset** in the panel as needed. Close the MuJoCo viewer or control panel to exit.

Gamepad mapping (SDL axes/buttons):

| Input | Action |
| --- | --- |
| Left stick up/down | Forward/backward command |
| Left stick left/right | Lateral command |
| Right stick horizontal | Yaw command |
| A button | Reset the robot |
| Start button | Toggle pause/resume |
| Back button | Stop the simulation |

The default axis mapping uses SDL axes 0 and 1 for the left stick and axis 3 for right-stick horizontal (axis 2 is used if the controller has only three axes). A small deadzone filters stick drift. The panel reports policy, gamepad, simulation time, control frequency, command, and selected joint positions.

## Development

The main modules are:

- `g1_simulator/checkpoints.py`: checkpoint discovery, validation, and Brax policy restoration.
- `g1_simulator/simulation.py`: MJX stepping and MuJoCo viewer synchronization.
- `g1_simulator/gamepad.py`: SDL joystick input and command mapping.
- `g1_simulator/ui.py`: Tkinter control panel and checkpoint selection.

Run the unit tests in the local environment:

```bash
./.venv/bin/python -m unittest discover -s tests
```

## Troubleshooting

- **Python executable not found:** create `.venv` and install `requirements.txt`, or set `SIMULATOR_PYTHON` to another compatible Python executable.
- **MuJoCo Playground import error:** reinstall `requirements.txt`, or set `MUJOCO_PLAYGROUND_ROOT` to the checkout directory that contains the `mujoco_playground/` package.
- **No complete PPO checkpoints found:** confirm `data/training_checkpoints` points to the checkpoint root, not a numbered checkpoint directory, and that at least one numbered directory has both required marker/config files.
- **Training environment config not found:** ensure the checkpoint root contains the run's `config.json` next to the numbered checkpoint directories.
- **Incompatible observation/action size or normalization:** use a checkpoint trained for the G1 joystick flat-terrain task with the compatible actor layout described above.
- **JAX accelerator/plugin errors:** run with the default CPU setting, or configure a JAX/MuJoCo Playground accelerator stack that matches the installed drivers and packages.
- **No graphical viewer:** run from a desktop session with a display available; the simulator opens both a Tkinter window and a MuJoCo viewer.