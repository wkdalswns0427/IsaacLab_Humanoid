# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from collections.abc import Sequence

import isaaclab.utils.math as PoseUtils
from isaaclab.envs import ManagerBasedRLMimicEnv


class H1PickBlockMimicEnv(ManagerBasedRLMimicEnv):
    """Minimal Mimic-compatible env wrapper for H1 pick-block.

    This provides basic end-effector pose access and subtask signals for dataset annotation.
    Implements a simple differential IK-compatible action conversion (delta pose action).
    """

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._robot = self.scene["robot"]
        self._cube = self.scene["cube"]
        self._left_eef_body_ids, _ = self._robot.find_bodies([".*left_shoulder.*"], preserve_order=True)
        self._right_eef_body_ids, _ = self._robot.find_bodies([".*right_shoulder.*"], preserve_order=True)
        if len(self._left_eef_body_ids) == 0:
            self._left_eef_body_ids = [0]
        if len(self._right_eef_body_ids) == 0:
            self._right_eef_body_ids = [0]

    def get_robot_eef_pose(self, eef_name: str, env_ids: Sequence[int] | None = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = slice(None)
        if eef_name == "left":
            body_id = self._left_eef_body_ids[0]
        else:
            body_id = self._right_eef_body_ids[0]
        pos = self._robot.data.body_pos_w[env_ids, body_id]
        quat = self._robot.data.body_quat_w[env_ids, body_id]
        return PoseUtils.make_pose(pos, PoseUtils.matrix_from_quat(quat))

    def target_eef_pose_to_action(
        self,
        target_eef_pose_dict: dict,
        gripper_action_dict: dict,
        action_noise_dict: dict | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        eef_name = "left"
        (target_eef_pose,) = target_eef_pose_dict.values()
        target_pos, target_rot = PoseUtils.unmake_pose(target_eef_pose)

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=[env_id])[0]
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        delta_position = target_pos - curr_pos
        delta_rot_mat = target_rot.matmul(curr_rot.transpose(-1, -2))
        delta_quat = PoseUtils.quat_from_matrix(delta_rot_mat)
        delta_rotation = PoseUtils.axis_angle_from_quat(delta_quat)

        pose_action = torch.cat([delta_position, delta_rotation], dim=0)
        if action_noise_dict is not None and eef_name in action_noise_dict:
            noise = action_noise_dict[eef_name] * torch.randn_like(pose_action)
            pose_action = torch.clamp(pose_action + noise, -1.0, 1.0)
        return pose_action

    def action_to_target_eef_pose(self, action: torch.Tensor) -> dict[str, torch.Tensor]:
        eef_name = "left"
        delta_position = action[:, :3]
        delta_rotation = action[:, 3:6]

        curr_pose = self.get_robot_eef_pose(eef_name, env_ids=None)
        curr_pos, curr_rot = PoseUtils.unmake_pose(curr_pose)

        target_pos = curr_pos + delta_position
        delta_rotation_angle = torch.linalg.norm(delta_rotation, dim=-1, keepdim=True)
        delta_rotation_axis = delta_rotation / delta_rotation_angle

        is_close_to_zero_angle = torch.isclose(delta_rotation_angle, torch.zeros_like(delta_rotation_angle)).squeeze(1)
        delta_rotation_axis[is_close_to_zero_angle] = torch.zeros_like(delta_rotation_axis)[is_close_to_zero_angle]

        delta_quat = PoseUtils.quat_from_angle_axis(delta_rotation_angle.squeeze(1), delta_rotation_axis).squeeze(0)
        delta_rot_mat = PoseUtils.matrix_from_quat(delta_quat)
        target_rot = torch.matmul(delta_rot_mat, curr_rot)

        target_poses = PoseUtils.make_pose(target_pos, target_rot).clone()
        return {eef_name: target_poses}

    def actions_to_gripper_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        # No explicit gripper in this task.
        return {}

    def get_subtask_start_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        cube_pos = self._cube.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        ee_pos = self._robot.data.body_pos_w[env_ids, self._left_eef_body_ids[0]] - self.scene.env_origins[env_ids]
        dist = torch.norm(cube_pos - ee_pos, dim=-1)
        return {"reach": dist < 0.4}

    def get_subtask_term_signals(self, env_ids: Sequence[int] | None = None) -> dict[str, torch.Tensor]:
        if env_ids is None:
            env_ids = slice(None)
        cube_pos = self._cube.data.root_pos_w[env_ids] - self.scene.env_origins[env_ids]
        return {"lift": cube_pos[:, 2] > 0.12}
