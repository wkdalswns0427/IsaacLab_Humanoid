# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch

from isaaclab.envs.manager_based_rl_env import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def cube_pos_w(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    cube = env.scene[cube_cfg.name]
    return cube.data.root_pos_w - env.scene.env_origins


def cube_lin_vel_w(env: ManagerBasedRLEnv, cube_cfg: SceneEntityCfg = SceneEntityCfg("cube")) -> torch.Tensor:
    cube = env.scene[cube_cfg.name]
    return cube.data.root_lin_vel_w


def ee_to_cube_vec(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    robot = env.scene[robot_cfg.name]
    cube = env.scene[cube_cfg.name]
    ee_pos = robot.data.body_pos_w[:, robot_cfg.body_ids[0]]
    cube_pos = cube.data.root_pos_w
    return (cube_pos - env.scene.env_origins) - (ee_pos - env.scene.env_origins)


def ee_to_cube_distance(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    cube_cfg: SceneEntityCfg = SceneEntityCfg("cube"),
) -> torch.Tensor:
    vec = ee_to_cube_vec(env, robot_cfg=robot_cfg, cube_cfg=cube_cfg)
    return torch.norm(vec, dim=-1)
