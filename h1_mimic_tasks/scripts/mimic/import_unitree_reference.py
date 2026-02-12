#!/usr/bin/env python3
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Import Unitree replay JSON episodes as reference-motion HDF5.

This script converts Unitree `data.json` episode files (from unitree_sim_isaaclab
replay/record flow) into an IsaacLab-style HDF5 dataset so they can be reused
from the `h1_mimic_tasks` workspace as reference motion.

Notes:
- Output actions are stored as concatenated joint command vectors:
  [left_arm, right_arm, left_ee, right_ee]
- This is a reference-motion bridge. It does not automatically solve action-space
  mismatch with a specific Mimic task.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import torch

from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler


def _find_data_json_files(path: Path) -> list[Path]:
    if path.is_file():
        if path.name != "data.json":
            raise ValueError(f"Expected a data.json file, got: {path}")
        return [path]
    if not path.is_dir():
        raise FileNotFoundError(f"Input path does not exist: {path}")

    files = list(path.glob("**/data.json"))
    if not files:
        raise FileNotFoundError(f"No data.json files found under: {path}")

    def _episode_num(p: Path) -> int:
        m = re.search(r"episode_(\d+)", str(p))
        return int(m.group(1)) if m else 10**12

    files.sort(key=_episode_num)
    return files


def _is_numeric_list(values: list[Any]) -> bool:
    return all(isinstance(v, (int, float)) for v in values)


def _to_tensors(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _to_tensors(v) for k, v in obj.items()}
    if isinstance(obj, list):
        if obj and _is_numeric_list(obj):
            return torch.tensor(obj, dtype=torch.float32)
        if obj and all(isinstance(x, list) and _is_numeric_list(x) for x in obj):
            return torch.tensor(obj, dtype=torch.float32)
        return [_to_tensors(x) for x in obj]
    return obj


def _parse_sim_state(sim_state_field: Any) -> tuple[dict[str, Any], str]:
    # Unitree stores this as either dict or JSON-encoded string.
    payload = sim_state_field
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, dict):
        raise ValueError(f"Unsupported sim_state type: {type(payload)}")

    task_name = payload.get("task_name", "")
    init_state = payload.get("init_state", {})
    if isinstance(init_state, str):
        init_state = json.loads(init_state)
    if not isinstance(init_state, dict):
        raise ValueError(f"Unsupported init_state type: {type(init_state)}")
    return init_state, task_name


def _read_action_vector(item: dict[str, Any]) -> torch.Tensor:
    action = item.get("actions", {})
    if not isinstance(action, dict) or not action:
        raise ValueError("Missing `actions` in episode item")

    def _qpos(key: str) -> torch.Tensor:
        vec = action.get(key, {}).get("qpos", [])
        if not isinstance(vec, list):
            raise ValueError(f"Expected list for actions.{key}.qpos")
        return torch.tensor(vec, dtype=torch.float32)

    left_arm = _qpos("left_arm")
    right_arm = _qpos("right_arm")
    left_ee = _qpos("left_ee")
    right_ee = _qpos("right_ee")
    return torch.cat([left_arm, right_arm, left_ee, right_ee], dim=0)


def _convert_one_episode(
    json_path: Path,
    write_states: bool,
    max_steps: int | None,
    stride: int,
) -> tuple[EpisodeData, str]:
    with json_path.open("r", encoding="utf-8") as f:
        content = json.load(f)

    items = content.get("data", [])
    if not isinstance(items, list) or len(items) == 0:
        raise ValueError(f"No episode items in: {json_path}")

    first_init_state, task_name = _parse_sim_state(items[0].get("sim_state", {}))
    episode = EpisodeData()
    episode.add("initial_state", _to_tensors(first_init_state))

    steps_written = 0
    for idx, item in enumerate(items):
        if idx % stride != 0:
            continue
        if max_steps is not None and steps_written >= max_steps:
            break

        act = _read_action_vector(item)
        episode.add("actions", act)

        if write_states:
            step_state, _ = _parse_sim_state(item.get("sim_state", {}))
            episode.add("states", _to_tensors(step_state))

        steps_written += 1

    episode.pre_export()
    episode.success = False
    return episode, task_name


def main():
    parser = argparse.ArgumentParser(description="Import Unitree data.json episodes to HDF5 reference motion.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to a data.json file or directory containing episode_*/data.json.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/mimic/unitree_reference_raw.hdf5",
        help="Output HDF5 path.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="unitree_reference_motion",
        help="Dataset env name metadata (can be overridden later when consuming).",
    )
    parser.add_argument(
        "--write_states",
        action="store_true",
        default=False,
        help="Also write per-step state dict from sim_state.init_state.",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Optional max steps per episode after stride.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Step stride (>=1) to downsample trajectories.",
    )
    args = parser.parse_args()

    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    input_path = Path(args.input_path).expanduser().resolve()
    json_files = _find_data_json_files(input_path)

    handler = HDF5DatasetFileHandler()
    handler.create(args.output, env_name=args.env_name)
    handler.add_env_args(
        {
            "source": "unitree_sim_isaaclab",
            "format": "unitree_data_json_bridge_v1",
            "action_layout": "[left_arm, right_arm, left_ee, right_ee]",
            "num_source_episodes": len(json_files),
        }
    )

    task_names = []
    for demo_id, json_file in enumerate(json_files):
        episode, task_name = _convert_one_episode(
            json_path=json_file,
            write_states=args.write_states,
            max_steps=args.max_steps,
            stride=args.stride,
        )
        handler.write_episode(episode, demo_id=demo_id)
        handler.flush()
        if task_name:
            task_names.append(task_name)
        print(f"[import] wrote demo_{demo_id} from {json_file}")

    if task_names:
        unique_task_names = sorted(set(task_names))
        handler.add_env_args({"unitree_task_names": unique_task_names})
        handler.flush()
        print(f"[import] task names: {unique_task_names}")

    handler.close()
    print(f"[import] done -> {args.output}")


if __name__ == "__main__":
    main()
