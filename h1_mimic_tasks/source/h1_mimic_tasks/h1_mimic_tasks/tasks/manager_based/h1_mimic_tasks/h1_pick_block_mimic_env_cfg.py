# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from isaaclab.utils import configclass

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.envs.mdp import observations as obs
from isaaclab.envs.mdp import terminations as terms

from . import mdp

try:
    from isaaclab_assets import H1_2_CFG as _ROBOT_CFG
except Exception:
    from isaaclab_assets import H1_MINIMAL_CFG as _ROBOT_CFG


@configclass
class H1PickBlockSceneCfg(InteractiveSceneCfg):
    """Scene configuration for H1 pick-block mimic."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    robot: ArticulationCfg = _ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cube: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=False),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.5, 0.0, 0.1)),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )


@configclass
class ActionsCfg:
    arm_action = DifferentialInverseKinematicsActionCfg(
        asset_name="robot",
        joint_names=[".*left_.*(shoulder|elbow).*"],
        body_name="left_elbow_link",
        controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
        scale=0.5,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(func=obs.joint_pos_rel)
        joint_vel = ObsTerm(func=obs.joint_vel_rel)
        cube_pos = ObsTerm(func=mdp.cube_pos_w, params={"cube_cfg": SceneEntityCfg("cube")})
        cube_lin_vel = ObsTerm(func=mdp.cube_lin_vel_w, params={"cube_cfg": SceneEntityCfg("cube")})
        ee_to_cube = ObsTerm(
            func=mdp.ee_to_cube_vec,
            params={"robot_cfg": SceneEntityCfg("robot", body_names=[".*left_elbow.*"]), "cube_cfg": SceneEntityCfg("cube")},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # Mild shaping; dataset generation doesn't rely on rewards.
    reach = RewTerm(func=mdp.ee_to_cube_distance, weight=-0.1, params={"robot_cfg": SceneEntityCfg("robot", body_names=[".*left_shoulder.*"]), "cube_cfg": SceneEntityCfg("cube")})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=terms.time_out, time_out=True)


@configclass
class H1PickBlockEnvCfg(ManagerBasedRLEnvCfg):
    # simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120)
    decimation = 2
    episode_length_s = 10.0
    # scene
    scene: H1PickBlockSceneCfg = H1PickBlockSceneCfg(num_envs=32, env_spacing=4.0, replicate_physics=True)
    # actions
    actions: ActionsCfg = ActionsCfg()
    # observations
    observations: ObservationsCfg = ObservationsCfg()
    # rewards
    rewards: RewardsCfg = RewardsCfg()
    # terminations
    terminations: TerminationsCfg = TerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # This task only controls one arm; keep base fixed to avoid immediate collapse.
        self.scene.robot.spawn.articulation_props.fix_root_link = True


@configclass
class H1PickBlockMimicEnvCfg(H1PickBlockEnvCfg, MimicEnvCfg):
    """Mimic environment config for H1 pick-block."""

    def __post_init__(self):
        super().__post_init__()
        self.datagen_config.name = "demo_src_h1_pick_block"
        self.datagen_config.generation_guarantee = False
        self.datagen_config.generation_keep_failed = True
        self.datagen_config.generation_num_trials = 5
        self.datagen_config.generation_select_src_per_subtask = True
        self.datagen_config.seed = 1

        subtask_configs = []
        subtask_configs.append(
            SubTaskConfig(
                object_ref="cube",
                subtask_term_signal="lift",
                subtask_term_offset_range=(5, 10),
                selection_strategy="nearest_neighbor_object",
                selection_strategy_kwargs={"nn_k": 3},
                action_noise=0.02,
                num_interpolation_steps=5,
                description="Grasp and lift cube",
            )
        )
        self.subtask_configs["h1"] = subtask_configs
