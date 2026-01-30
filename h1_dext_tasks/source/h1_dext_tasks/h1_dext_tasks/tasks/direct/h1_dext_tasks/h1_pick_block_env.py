# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .h1_pick_block_env_cfg import H1PickBlockEnvCfg


class H1PickBlockEnv(DirectRLEnv):
    cfg: H1PickBlockEnvCfg

    def __init__(self, cfg: H1PickBlockEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._default_dof_pos = self.robot.data.default_joint_pos
        self._default_dof_vel = self.robot.data.default_joint_vel
        self._default_root_state = self.robot.data.default_root_state

        self._left_ee_idx = self._resolve_body_index(self.cfg.left_ee_body_names)
        self._right_ee_idx = self._resolve_body_index(self.cfg.right_ee_body_names)

        self._cube_pos = self.cube.data.root_pos_w
        self._cube_lin_vel = self.cube.data.root_lin_vel_w

    def _resolve_body_index(self, name_keys: list[str]) -> int:
        body_ids, _ = self.robot.find_bodies(name_keys, preserve_order=True)
        if len(body_ids) > 0:
            return body_ids[0]
        fallback_ids, _ = self.robot.find_bodies(self.cfg.ee_fallback_body_names, preserve_order=True)
        if len(fallback_ids) > 0:
            return fallback_ids[0]
        return 0

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cube = RigidObject(self.cfg.cube_cfg)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["cube"] = self.cube

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.actions = actions.clamp(-1.0, 1.0)

    def _apply_action(self) -> None:
        targets = self._default_dof_pos + self.actions * self.cfg.action_scale
        self.robot.set_joint_position_target(targets)

    def _compute_intermediate_values(self):
        body_pos_w = self.robot.data.body_pos_w
        left_pos = body_pos_w[:, self._left_ee_idx]
        right_pos = body_pos_w[:, self._right_ee_idx]
        ee_pos = 0.5 * (left_pos + right_pos)

        cube_pos = self.cube.data.root_pos_w
        cube_lin_vel = self.cube.data.root_lin_vel_w
        cube_pos_local = cube_pos - self.scene.env_origins

        ee_pos_local = ee_pos - self.scene.env_origins
        ee_to_cube = cube_pos_local - ee_pos_local

        return cube_pos_local, cube_lin_vel, ee_to_cube

    def _get_observations(self) -> dict:
        cube_pos, cube_lin_vel, ee_to_cube = self._compute_intermediate_values()
        dof_pos = self.robot.data.joint_pos - self._default_dof_pos
        dof_vel = self.robot.data.joint_vel

        obs = torch.cat((dof_pos, dof_vel, cube_pos, cube_lin_vel, ee_to_cube), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        cube_pos, _, ee_to_cube = self._compute_intermediate_values()

        dist = torch.norm(ee_to_cube, dim=-1)
        lift = torch.clamp(cube_pos[:, 2] - self.cfg.lift_height, min=0.0)
        success = (cube_pos[:, 2] > self.cfg.success_height).float()

        rew_alive = self.cfg.rew_scale_alive
        rew_dist = -self.cfg.rew_scale_dist * dist
        rew_lift = self.cfg.rew_scale_lift * lift
        rew_success = self.cfg.rew_scale_success * success
        rew_action = self.cfg.rew_scale_action * torch.sum(self.actions * self.actions, dim=-1)

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["success_rate"] = success.mean()
        self.extras["log"]["rew_alive"] = torch.as_tensor(rew_alive, device=self.device).mean()
        self.extras["log"]["rew_dist"] = rew_dist.mean()
        self.extras["log"]["rew_lift"] = rew_lift.mean()
        self.extras["log"]["rew_success"] = rew_success.mean()
        self.extras["log"]["rew_action"] = rew_action.mean()

        return rew_alive + rew_dist + rew_lift + rew_success + rew_action

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        base_height = self.robot.data.root_pos_w[:, 2]
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        terminated = base_height < self.cfg.termination_height
        if self.cfg.terminate_on_success:
            cube_pos = self.cube.data.root_pos_w - self.scene.env_origins
            terminated = terminated | (cube_pos[:, 2] > self.cfg.success_height)
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        # reset robot
        joint_pos = self._default_dof_pos[env_ids]
        joint_vel = self._default_dof_vel[env_ids]

        default_root_state = self._default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        # reset cube
        cube_state = self.cube.data.default_root_state[env_ids].clone()
        pos_noise = sample_uniform(
            -self.cfg.cube_pos_noise_xy,
            self.cfg.cube_pos_noise_xy,
            (len(env_ids), 2),
            device=self.device,
        )
        cube_state[:, 0] = cube_state[:, 0] + pos_noise[:, 0] + self.scene.env_origins[env_ids, 0]
        cube_state[:, 1] = cube_state[:, 1] + pos_noise[:, 1] + self.scene.env_origins[env_ids, 1]
        cube_state[:, 2] = 0.1 + self.scene.env_origins[env_ids, 2]
        cube_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        cube_state[:, 7:] = 0.0
        self.cube.write_root_pose_to_sim(cube_state[:, :7], env_ids)
        self.cube.write_root_velocity_to_sim(cube_state[:, 7:], env_ids)
