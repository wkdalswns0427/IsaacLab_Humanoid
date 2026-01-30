# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab_assets import H1_MINIMAL_CFG

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass


@configclass
class H1PickBlockDextrousEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 2
    episode_length_s = 10.0
    action_scale = 0.5  # [rad]
    action_space = 19
    observation_space = 47
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = H1_MINIMAL_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    contact_sensor: ContactSensorCfg = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=0.005,
        track_air_time=False,
    )

    # object
    cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.4, 0.9)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.1)),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=512, env_spacing=4.0, replicate_physics=True)

    # reset randomization
    cube_pos_noise_xy = 0.1

    # end-effector candidates (regex)
    left_ee_body_names = [
        ".*left_elbow.*",
        ".*left_shoulder.*",
    ]
    right_ee_body_names = [
        ".*right_elbow.*",
        ".*right_shoulder.*",
    ]
    ee_fallback_body_names = [
        ".*torso_link.*",
        ".*torso.*",
        ".*pelvis.*",
        ".*base.*",
    ]

    # hand contact bodies (regex)
    hand_contact_body_names = [
        ".*left_elbow.*",
        ".*right_elbow.*",
        ".*left_shoulder.*",
        ".*right_shoulder.*",
    ]

    # rewards
    rew_scale_dist = 1.0
    rew_scale_lift = 5.0
    rew_scale_success = 10.0
    rew_scale_action = -0.01
    rew_scale_contact = 0.5
    contact_force_threshold = 2.0
    lift_height = 0.12
    success_height = 0.25

    # terminations
    termination_height = 0.6
    terminate_on_success = False
