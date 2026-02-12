# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Record demonstration rollouts from Isaac Sim into an HDF5 dataset."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record H1 Mimic demos to HDF5.")
parser.add_argument("--task", type=str, default="H1-Pick-Block-Mimic-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--num_demos", type=int, default=5)
parser.add_argument("--demo_length", type=int, default=300)
parser.add_argument("--output", type=str, default="outputs/mimic/h1_pick_block_raw.hdf5")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.utils.datasets import EpisodeData, HDF5DatasetFileHandler
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import h1_mimic_tasks.tasks  # noqa: F401


def main():
    device = getattr(args_cli, "device", "cuda:0")
    env_cfg = parse_env_cfg(args_cli.task, device=device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    handler = HDF5DatasetFileHandler()
    handler.create(args_cli.output, env_name=args_cli.task)

    for demo_id in range(args_cli.num_demos):
        obs, _ = env.reset()
        episode = EpisodeData()
        for _ in range(args_cli.demo_length):
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            obs, _, terminated, truncated, _ = env.step(actions)
            episode.add("actions", actions.cpu())
            episode.add("states", env.unwrapped.scene.get_state(is_relative=True))
            if torch.any(terminated | truncated):
                break
        episode.pre_export()
        episode.success = False
        handler.write_episode(episode, demo_id=demo_id)
        handler.flush()

    handler.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
