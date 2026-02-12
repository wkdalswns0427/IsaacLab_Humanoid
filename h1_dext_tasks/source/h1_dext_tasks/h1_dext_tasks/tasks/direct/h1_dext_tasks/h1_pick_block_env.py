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
from isaaclab.sensors import ContactSensor
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils.math import euler_xyz_from_quat
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
        self._hand_contact_ids = self._resolve_hand_contact_ids()
        self._left_hand_contact_ids = self._resolve_hand_contact_ids(self.cfg.left_hand_contact_body_names)
        self._right_hand_contact_ids = self._resolve_hand_contact_ids(self.cfg.right_hand_contact_body_names)
        self._knee_joint_ids = self._resolve_knee_joint_ids()
        self._left_knee_joint_ids = self._resolve_knee_joint_ids(self.cfg.left_knee_joint_names)
        self._right_knee_joint_ids = self._resolve_knee_joint_ids(self.cfg.right_knee_joint_names)
        self._hip_abduction_joint_ids = self._resolve_hip_abduction_joint_ids()

        self._cube_pos = self.cube.data.root_pos_w
        self._cube_lin_vel = self.cube.data.root_lin_vel_w
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)
        self._prev_actions = torch.zeros_like(self.actions)

    def _resolve_body_index(self, name_keys: list[str]) -> int:
        body_ids, _ = self.robot.find_bodies(name_keys, preserve_order=True)
        if len(body_ids) > 0:
            return body_ids[0]
        fallback_ids, _ = self.robot.find_bodies(self.cfg.ee_fallback_body_names, preserve_order=True)
        if len(fallback_ids) > 0:
            return fallback_ids[0]
        return 0

    def _resolve_hand_contact_ids(self, name_keys: list[str] | None = None) -> torch.Tensor:
        if name_keys is None:
            name_keys = self.cfg.hand_contact_body_names
        body_ids, _ = self._contact_sensor.find_bodies(name_keys, preserve_order=True)
        if len(body_ids) == 0:
            body_ids, _ = self._contact_sensor.find_bodies(self.cfg.ee_fallback_body_names, preserve_order=True)
        if len(body_ids) == 0:
            body_ids = [0]
        return torch.tensor(body_ids, device=self.device, dtype=torch.long)

    def _resolve_knee_joint_ids(self, name_keys: list[str] | None = None) -> torch.Tensor:
        if name_keys is None:
            name_keys = self.cfg.knee_joint_names
        joint_ids, _ = self.robot.find_joints(name_keys, preserve_order=True)
        if len(joint_ids) == 0:
            joint_ids = list(range(self.robot.num_joints))
        return torch.tensor(joint_ids, device=self.device, dtype=torch.long)

    def _resolve_hip_abduction_joint_ids(self) -> torch.Tensor:
        joint_ids, _ = self.robot.find_joints(self.cfg.hip_abduction_joint_names, preserve_order=True)
        if len(joint_ids) == 0:
            joint_ids = list(range(self.robot.num_joints))
        return torch.tensor(joint_ids, device=self.device, dtype=torch.long)

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        self.cube = RigidObject(self.cfg.cube_cfg)
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)

        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())

        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])

        self.scene.articulations["robot"] = self.robot
        self.scene.rigid_objects["cube"] = self.cube
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self._prev_actions = self.actions.clone()
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
        cube_lifted = cube_pos[:, 2] > self.cfg.lift_height
        time_s = self.episode_length_buf * (self.cfg.sim.dt * self.cfg.decimation)
        base_pos_local = self.robot.data.root_pos_w - self.scene.env_origins

        rew_alive = self.cfg.rew_scale_alive
        rew_action = self.cfg.rew_scale_action * torch.sum(self.actions * self.actions, dim=-1)
        rew_action_rate = self.cfg.rew_scale_action_rate * torch.sum(
            (self.actions - self._prev_actions) * (self.actions - self._prev_actions), dim=-1
        )

        # Stage 1: keep upright and bend knees before going for the box.
        roll, pitch, _ = euler_xyz_from_quat(self.robot.data.root_quat_w)
        tilt_sq = roll * roll + pitch * pitch
        posture_upright = torch.exp(-tilt_sq / (self.cfg.upright_tilt_sigma * self.cfg.upright_tilt_sigma))
        hip_pos = self.robot.data.joint_pos[:, self._hip_abduction_joint_ids]
        hip_mean = torch.mean(hip_pos, dim=-1)
        hip_centered = torch.exp(
            -((hip_mean - self.cfg.target_hip_abduction) ** 2)
            / (self.cfg.hip_abduction_sigma * self.cfg.hip_abduction_sigma)
        )
        rew_posture = (
            self.cfg.rew_scale_posture_upright * posture_upright
            + self.cfg.rew_scale_hip_abduction * hip_centered
        )
        left_knee = torch.mean(self.robot.data.joint_pos[:, self._left_knee_joint_ids], dim=-1)
        right_knee = torch.mean(self.robot.data.joint_pos[:, self._right_knee_joint_ids], dim=-1)
        kneel_knee = torch.exp(
            -((left_knee - self.cfg.target_kneel_knee) ** 2)
            / (self.cfg.kneel_knee_sigma * self.cfg.kneel_knee_sigma)
        )
        kneel_activate = (time_s >= self.cfg.bend_start_time_s).float()
        rew_kneel = self.cfg.rew_scale_kneel * kneel_knee * kneel_activate

        # Stage 2: reach and make contact with the box.
        reach_score = torch.exp(-dist / self.cfg.stage_reach_distance)
        rew_reach = self.cfg.rew_scale_reach * reach_score
        # Reward for hand end-effectors making contact with the cube (contact-based shaping).
        net_contact_forces = self._contact_sensor.data.net_forces_w_history
        contact_mag = torch.norm(net_contact_forces[:, :, self._hand_contact_ids], dim=-1)
        hand_contact_force = contact_mag.amax(dim=(1, 2))
        left_contact_force = torch.norm(net_contact_forces[:, :, self._left_hand_contact_ids], dim=-1).amax(dim=(1, 2))
        right_contact_force = torch.norm(net_contact_forces[:, :, self._right_hand_contact_ids], dim=-1).amax(
            dim=(1, 2)
        )
        contact_bonus = torch.nn.functional.relu(hand_contact_force - self.cfg.contact_force_threshold)
        rew_contact = self.cfg.rew_scale_contact * contact_bonus
        both_contact = (left_contact_force > self.cfg.contact_force_threshold) & (
            right_contact_force > self.cfg.contact_force_threshold
        )
        grasp_bonus = both_contact.float()
        rew_grasp = self.cfg.rew_scale_grasp * grasp_bonus
        # Penalize excessively large contact impulses to discourage kicking.
        contact_excess = torch.nn.functional.relu(hand_contact_force - self.cfg.contact_force_penalty_threshold)
        rew_contact_force_penalty = self.cfg.rew_scale_contact_force_penalty * contact_excess
        # Stage 3: lift reward activates once both end-effectors have contact.
        rew_lift = self.cfg.rew_scale_lift * lift * both_contact.float()
        rew_success = self.cfg.rew_scale_success * success

        # Staged schedule: stand briefly, then bend/reach if still alive.
        stage0_mask = (time_s < self.cfg.stand_duration_s).float()
        stage1_mask = ((time_s >= self.cfg.stand_duration_s) & (time_s < self.cfg.bend_start_time_s)).float()
        stage2_mask = (time_s >= self.cfg.bend_start_time_s).float()
        stage3_mask = cube_lifted.float()

        # Penalize standing over the cube to avoid stepping on it.
        base_xy = base_pos_local[:, :2]
        cube_xy = cube_pos[:, :2]
        base_cube_dist = torch.norm(base_xy - cube_xy, dim=-1)
        base_over_cube = torch.nn.functional.relu(self.cfg.base_over_cube_radius - base_cube_dist)
        rew_base_over_cube = self.cfg.rew_scale_base_over_cube_penalty * base_over_cube

        total_reward = (
            rew_alive
            + rew_action
            + rew_action_rate
            + stage0_mask * (0.5 * rew_posture)
            + stage1_mask * (0.5 * rew_posture)
            + stage2_mask * (0.25 * rew_posture + rew_kneel + rew_reach + rew_contact + rew_grasp)
            + stage3_mask
            * (0.25 * rew_posture + rew_kneel + rew_reach + rew_contact + rew_grasp + rew_lift + rew_success)
            + rew_contact_force_penalty
            + rew_base_over_cube
        )

        if "log" not in self.extras:
            self.extras["log"] = {}
        self.extras["log"]["success_rate"] = success.mean()
        self.extras["log"]["stage0_frac"] = stage0_mask.mean()
        self.extras["log"]["stage1_frac"] = stage1_mask.mean()
        self.extras["log"]["stage2_frac"] = stage2_mask.mean()
        self.extras["log"]["stage3_frac"] = stage3_mask.mean()
        self.extras["log"]["rew_alive"] = torch.as_tensor(rew_alive, device=self.device).mean()
        self.extras["log"]["rew_posture"] = rew_posture.mean()
        self.extras["log"]["rew_reach"] = rew_reach.mean()
        self.extras["log"]["rew_hip_abduction"] = hip_centered.mean()
        self.extras["log"]["rew_lift"] = rew_lift.mean()
        self.extras["log"]["rew_success"] = rew_success.mean()
        self.extras["log"]["rew_action"] = rew_action.mean()
        self.extras["log"]["rew_action_rate"] = rew_action_rate.mean()
        self.extras["log"]["rew_grasp"] = rew_grasp.mean()
        self.extras["log"]["both_contact"] = both_contact.float().mean()
        self.extras["log"]["rew_contact_force_penalty"] = rew_contact_force_penalty.mean()
        self.extras["log"]["rew_kneel"] = rew_kneel.mean()
        self.extras["log"]["rew_base_over_cube"] = rew_base_over_cube.mean()

        return total_reward

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
        self.actions[env_ids] = 0.0
        self._prev_actions[env_ids] = 0.0
