import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="H1 sitting on a chair in front of a table.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import H1_MINIMAL_CFG

import math
import torch

def apply_seated_pose(robot: Articulation, sim: SimulationContext):

    robot.update(sim.get_physics_dt())

    jp = robot.data.default_joint_pos.clone()
    jv = robot.data.default_joint_vel.clone()

    names = robot.data.joint_names
    print("Joint names:", names)

    def set_joint(sub, angle_deg):
        ang = math.radians(angle_deg)
        for i, name in enumerate(names):
            if sub in name:
                jp[:, i] = ang

    # Basic sitting posture
    set_joint("left_hip_pitch", 60)
    set_joint("right_hip_pitch", 60)
    set_joint("left_knee", -80)
    set_joint("right_knee", -80)
    set_joint("left_ankle_pitch", 20)
    set_joint("right_ankle_pitch", 20)
    set_joint("left_shoulder_pitch", 20)
    set_joint("right_shoulder_pitch", 20)
    set_joint("left_elbow", 60)
    set_joint("right_elbow", 60)

    robot.write_joint_state_to_sim(jp, jv)
    robot.write_root_velocity_to_sim(torch.zeros_like(robot.data.root_vel_w))


def design_scene():

    # Ground
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/Ground", ground_cfg)

    # Light
    light_cfg = sim_utils.DomeLightCfg(intensity=3000.0)
    light_cfg.func("/World/Light", light_cfg)

    # Create an origin group
    prim_utils.create_prim("/World/Origin", "Xform")

    # Set pelvis slightly above chair so physics lets it settle
    h1_cfg = H1_MINIMAL_CFG.replace(
        prim_path="/World/Origin/H1",
        init_state=H1_MINIMAL_CFG.init_state.replace(
            pos=(0.0, 0.0, 0.55),  # pelvis above seat height
            rot=(1, 0, 0, 0),
        )
    )
    robot = Articulation(h1_cfg)

    # table_usd = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    # table_cfg = sim_utils.UsdFileCfg(usd_path=table_usd)
    # table_cfg.func(
    #     "/World/Origin/Table",
    #     table_cfg,
    #     translation=(1.4, 0.0, 1.05),
    # )

    chair_usd = f"{ISAAC_NUCLEUS_DIR}/Environments/Office/Props/SM_Armchair.usd"
    chair_cfg = sim_utils.UsdFileCfg(usd_path=chair_usd)
    chair_cfg.func(
        "/World/Origin/Chair",
        chair_cfg,
        translation=(0.0, 0.0, 0.0),   # adjust if needed
        orientation=(1, 0, 0, 0),
    )
    

    return robot

def main():

    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device)
    sim = SimulationContext(sim_cfg)

    # Camera view
    sim.set_camera_view(
        eye=[2.5, 0.0, 2.0],
        target=[0.3, 0.0, 1.0],
    )

    robot = design_scene()

    sim.reset()
    apply_seated_pose(robot, sim)

    print("[INFO] H1 should now be sitting on the chair.")

    while simulation_app.is_running():
        sim.step()
        robot.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()
