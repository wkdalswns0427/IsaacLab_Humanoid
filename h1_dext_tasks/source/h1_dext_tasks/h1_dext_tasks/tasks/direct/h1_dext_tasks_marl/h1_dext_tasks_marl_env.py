# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectMARLEnv
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import sample_uniform

from .h1_dext_tasks_marl_env_cfg import H1DextTasksMarlEnvCfg


class H1DextTasksMarlEnv(DirectMARLEnv):
    cfg: H1DextTasksMarlEnvCfg

    def __init__(self, cfg: H1DextTasksMarlEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._robot0_default_dof_pos = self.robot_0.data.default_joint_pos
        self._robot1_default_dof_pos = self.robot_1.data.default_joint_pos
        self._robot0_default_dof_vel = self.robot_0.data.default_joint_vel
        self._robot1_default_dof_vel = self.robot_1.data.default_joint_vel
        self._robot0_default_root_state = self.robot_0.data.default_root_state
        self._robot1_default_root_state = self.robot_1.data.default_root_state

        self._left_ee_idx_0 = self._resolve_body_index(self.robot_0, self.cfg.left_ee_body_names)
        self._right_ee_idx_0 = self._resolve_body_index(self.robot_0, self.cfg.right_ee_body_names)
        self._left_ee_idx_1 = self._resolve_body_index(self.robot_1, self.cfg.left_ee_body_names)
        self._right_ee_idx_1 = self._resolve_body_index(self.robot_1, self.cfg.right_ee_body_names)

        self._hand_contact_ids_0 = self._resolve_hand_contact_ids(self.contact_sensor_0)
        self._hand_contact_ids_1 = self._resolve_hand_contact_ids(self.contact_sensor_1)

    def _resolve_body_index(self, robot: Articulation, name_keys: list[str]) -> int:
        body_ids, _ = robot.find_bodies(name_keys, preserve_order=True)
        if len(body_ids) > 0:
            return body_ids[0]
        fallback_ids, _ = robot.find_bodies(self.cfg.ee_fallback_body_names, preserve_order=True)
        if len(fallback_ids) > 0:
            return fallback_ids[0]
        return 0

    def _resolve_hand_contact_ids(self, sensor: ContactSensor) -> torch.Tensor:
        body_ids, _ = sensor.find_bodies(self.cfg.hand_contact_body_names, preserve_order=True)
        if len(body_ids) == 0:
            body_ids, _ = sensor.find_bodies(self.cfg.ee_fallback_body_names, preserve_order=True)
        if len(body_ids) == 0:
            body_ids = [0]
        return torch.tensor(body_ids, device=self.device, dtype=torch.long)

    def _setup_scene(self):
        self.robot_0 = Articulation(self.cfg.robot_cfg.replace(prim_path="/World/envs/env_.*/Robot0"))
        self.robot_1 = Articulation(self.cfg.robot_cfg.replace(prim_path="/World/envs/env_.*/Robot1"))
        self.cube_0 = RigidObject(self.cfg.cube_cfg_0)
        self.cube_1 = RigidObject(self.cfg.cube_cfg_1)
        self.contact_sensor_0 = ContactSensor(self.cfg.contact_sensor_0)
        self.contact_sensor_1 = ContactSensor(self.cfg.contact_sensor_1)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.articulations["robot_0"] = self.robot_0
        self.scene.articulations["robot_1"] = self.robot_1
        self.scene.rigid_objects["cube_0"] = self.cube_0
        self.scene.rigid_objects["cube_1"] = self.cube_1
        self.scene.sensors["contact_sensor_0"] = self.contact_sensor_0
        self.scene.sensors["contact_sensor_1"] = self.contact_sensor_1

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = {agent: act.clamp(-1.0, 1.0) for agent, act in actions.items()}

    def _apply_action(self) -> None:
        self.robot_0.set_joint_position_target(self._robot0_default_dof_pos + self.actions["agent_0"] * self.cfg.action_scale)
        self.robot_1.set_joint_position_target(self._robot1_default_dof_pos + self.actions["agent_1"] * self.cfg.action_scale)

    def _compute_obs(self, robot: Articulation, cube: RigidObject, left_idx: int, right_idx: int) -> torch.Tensor:
        body_pos_w = robot.data.body_pos_w
        left_pos = body_pos_w[:, left_idx]
        right_pos = body_pos_w[:, right_idx]
        ee_pos = 0.5 * (left_pos + right_pos)

        cube_pos = cube.data.root_pos_w
        cube_lin_vel = cube.data.root_lin_vel_w
        cube_pos_local = cube_pos - self.scene.env_origins

        ee_pos_local = ee_pos - self.scene.env_origins
        ee_to_cube = cube_pos_local - ee_pos_local

        dof_pos = robot.data.joint_pos - robot.data.default_joint_pos
        dof_vel = robot.data.joint_vel

        return torch.cat((dof_pos, dof_vel, cube_pos_local, cube_lin_vel, ee_to_cube), dim=-1)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        obs_0 = self._compute_obs(self.robot_0, self.cube_0, self._left_ee_idx_0, self._right_ee_idx_0)
        obs_1 = self._compute_obs(self.robot_1, self.cube_1, self._left_ee_idx_1, self._right_ee_idx_1)
        return {"agent_0": obs_0, "agent_1": obs_1}

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        rew_0 = self._compute_reward(
            self.robot_0, self.cube_0, self._left_ee_idx_0, self._right_ee_idx_0, self.contact_sensor_0, self._hand_contact_ids_0, self.actions["agent_0"]
        )
        rew_1 = self._compute_reward(
            self.robot_1, self.cube_1, self._left_ee_idx_1, self._right_ee_idx_1, self.contact_sensor_1, self._hand_contact_ids_1, self.actions["agent_1"]
        )
        return {"agent_0": rew_0, "agent_1": rew_1}

    def _compute_reward(
        self,
        robot: Articulation,
        cube: RigidObject,
        left_idx: int,
        right_idx: int,
        sensor: ContactSensor,
        hand_contact_ids: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        body_pos_w = robot.data.body_pos_w
        left_pos = body_pos_w[:, left_idx]
        right_pos = body_pos_w[:, right_idx]
        ee_pos = 0.5 * (left_pos + right_pos)

        cube_pos = cube.data.root_pos_w - self.scene.env_origins
        ee_pos_local = ee_pos - self.scene.env_origins
        ee_to_cube = cube_pos - ee_pos_local

        dist = torch.norm(ee_to_cube, dim=-1)
        lift = torch.clamp(cube_pos[:, 2] - self.cfg.lift_height, min=0.0)
        success = (cube_pos[:, 2] > self.cfg.success_height).float()

        net_contact_forces = sensor.data.net_forces_w_history
        hand_contact_force = torch.max(torch.norm(net_contact_forces[:, :, hand_contact_ids], dim=-1), dim=1)[0]
        contact_bonus = torch.nn.functional.relu(hand_contact_force - self.cfg.contact_force_threshold)

        rew_dist = -self.cfg.rew_scale_dist * dist
        rew_lift = self.cfg.rew_scale_lift * lift
        rew_success = self.cfg.rew_scale_success * success
        rew_action = self.cfg.rew_scale_action * torch.sum(actions * actions, dim=-1)
        rew_contact = self.cfg.rew_scale_contact * contact_bonus

        return rew_dist + rew_lift + rew_success + rew_action + rew_contact

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        base_height_0 = self.robot_0.data.root_pos_w[:, 2]
        base_height_1 = self.robot_1.data.root_pos_w[:, 2]
        terminated_0 = base_height_0 < self.cfg.termination_height
        terminated_1 = base_height_1 < self.cfg.termination_height

        if self.cfg.terminate_on_success:
            cube_0_pos = self.cube_0.data.root_pos_w - self.scene.env_origins
            cube_1_pos = self.cube_1.data.root_pos_w - self.scene.env_origins
            terminated_0 = terminated_0 | (cube_0_pos[:, 2] > self.cfg.success_height)
            terminated_1 = terminated_1 | (cube_1_pos[:, 2] > self.cfg.success_height)

        terminated = {"agent_0": terminated_0, "agent_1": terminated_1}
        time_outs = {"agent_0": time_out, "agent_1": time_out}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot_0._ALL_INDICES
        super()._reset_idx(env_ids)

        self._reset_robot(self.robot_0, self._robot0_default_dof_pos, self._robot0_default_dof_vel, self._robot0_default_root_state, env_ids, self.cfg.robot_0_y_offset)
        self._reset_robot(self.robot_1, self._robot1_default_dof_pos, self._robot1_default_dof_vel, self._robot1_default_root_state, env_ids, self.cfg.robot_1_y_offset)

        self._reset_cube(self.cube_0, env_ids, base_y_offset=-0.1)
        self._reset_cube(self.cube_1, env_ids, base_y_offset=0.1)

    def _reset_robot(
        self,
        robot: Articulation,
        default_dof_pos: torch.Tensor,
        default_dof_vel: torch.Tensor,
        default_root_state: torch.Tensor,
        env_ids: Sequence[int],
        y_offset: float,
    ) -> None:
        joint_pos = default_dof_pos[env_ids]
        joint_vel = default_dof_vel[env_ids]

        root_state = default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 1] += y_offset

        robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _reset_cube(self, cube: RigidObject, env_ids: Sequence[int], base_y_offset: float) -> None:
        cube_state = cube.data.default_root_state[env_ids].clone()
        pos_noise = sample_uniform(
            -self.cfg.cube_pos_noise_xy,
            self.cfg.cube_pos_noise_xy,
            (len(env_ids), 2),
            device=self.device,
        )
        cube_state[:, 0] = 0.6 + pos_noise[:, 0] + self.scene.env_origins[env_ids, 0]
        cube_state[:, 1] = base_y_offset + pos_noise[:, 1] + self.scene.env_origins[env_ids, 1]
        cube_state[:, 2] = 0.1 + self.scene.env_origins[env_ids, 2]
        cube_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device)
        cube_state[:, 7:] = 0.0
        cube.write_root_pose_to_sim(cube_state[:, :7], env_ids)
        cube.write_root_velocity_to_sim(cube_state[:, 7:], env_ids)
