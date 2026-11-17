"""Discovery and restoration of MuJoCo Playground Brax PPO checkpoints."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ACTOR_OBSERVATION_SIZE = 103
ACTION_SIZE = 29
INITIALIZER_KEYS = (
    "policy_network_kernel_init_fn",
    "value_network_kernel_init_fn",
    "mean_kernel_init_fn",
)


@dataclass(frozen=True)
class Checkpoint:
  name: str
  path: Path
  step: int


def discover_checkpoints(root: Path) -> list[Checkpoint]:
  """Return complete, numbered checkpoints in training-step order."""
  if not root.is_dir():
    raise FileNotFoundError(f"Checkpoint directory does not exist: {root}")

  checkpoints = []
  for path in root.iterdir():
    if not path.is_dir() or not path.name.isdecimal():
      continue
    if not (path / "commit_success.txt").is_file():
      continue
    if not (path / "ppo_network_config.json").is_file():
      continue
    checkpoints.append(Checkpoint(path.name, path, int(path.name)))

  return sorted(checkpoints, key=lambda checkpoint: checkpoint.step)


def read_network_config(path: Path) -> dict[str, Any]:
  config_path = path / "ppo_network_config.json"
  if not config_path.is_file():
    raise FileNotFoundError(f"PPO network config not found: {config_path}")

  config = json.loads(config_path.read_text(encoding="utf-8"))
  observation_size = config.get("observation_size", {})
  actor_observation = observation_size.get("state", {}).get("shape", [])
  if actor_observation != [ACTOR_OBSERVATION_SIZE]:
    raise ValueError(
        f"{path.name} has actor observation shape {actor_observation}; "
        f"expected [{ACTOR_OBSERVATION_SIZE}] for G1JoystickFlatTerrain."
    )
  if config.get("action_size") != ACTION_SIZE:
    raise ValueError(
        f"{path.name} has action size {config.get('action_size')}; "
        f"expected {ACTION_SIZE} for G1JoystickFlatTerrain."
    )
  if not config.get("normalize_observations"):
    raise ValueError(
        f"{path.name} does not enable the observation normalization used "
        "by this policy."
    )
  return config


def load_policy(path: Path):
  """Restore a deterministic Brax policy, including checkpoint normalization."""
  from brax.training import checkpoint as brax_checkpoint
  from brax.training import networks as brax_networks
  from brax.training.agents.ppo import checkpoint as ppo_checkpoint
  from brax.training.agents.ppo import networks as ppo_networks
  from ml_collections import config_dict

  config_data = read_network_config(path)
  factory_kwargs = config_data.get("network_factory_kwargs")
  if not isinstance(factory_kwargs, dict):
    raise TypeError(f"Invalid PPO network config in {path}")

  factory_kwargs = dict(factory_kwargs)
  activation = factory_kwargs.get("activation")
  if isinstance(activation, str):
    try:
      factory_kwargs["activation"] = brax_networks.ACTIVATION[activation]
    except KeyError as error:
      raise ValueError(
          f"Unsupported Brax activation {activation!r} in {path}"
      ) from error

  for key in INITIALIZER_KEYS:
    initializer = factory_kwargs.get(key)
    if initializer is None:
      continue
    try:
      factory_kwargs[key] = brax_networks.KERNEL_INITIALIZER[initializer]
    except KeyError as error:
      raise ValueError(
          f"Unsupported Brax kernel initializer {initializer!r} in {path}"
      ) from error

  config_data["network_factory_kwargs"] = factory_kwargs
  network_config = config_dict.ConfigDict(config_data)
  network = brax_checkpoint.get_network(
      network_config, ppo_networks.make_ppo_networks
  )
  params = ppo_checkpoint.load(path)
  make_inference_fn = ppo_networks.make_inference_fn(network)
  return make_inference_fn(params, deterministic=True)
