# Copyright (c) 2022-2025, The Isaac Lab Project Developers
# SPDX-License-Identifier: BSD-3-Clause

"""
Simple scene: H1 humanoid standing in front of a table.

"""

import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="H1 standing in front of a table.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaacsim.core.utils.prims as prim_utils

import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext

from isaaclab.assets import Articulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# H1 robot config from IsaacLab assets
from isaaclab_assets import H1_MINIMAL_CFG  # or H1_CFG if you prefer full config


def design_scene():
    """Design the scene: ground, light, H1, table."""
    # Ground
    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)

    # Light
    light_cfg = sim_utils.DomeLightCfg(
        intensity=3000.0,
        color=(0.75, 0.75, 0.75),
    )
    light_cfg.func("/World/Light", light_cfg)

    # Create an origin transform so we can move everything easily if needed
    prim_utils.create_prim("/World/Origin", "Xform", translation=(0.0, 0.0, 0.0))

    # ---------- Spawn H1 ----------
    # Use the minimal H1 config and just change its prim path
    h1_cfg = H1_MINIMAL_CFG.replace(prim_path="/World/Origin/H1")
    robot = Articulation(h1_cfg)

    # ---------- Spawn Table ----------
    # This is the same table used in the spawn_prims tutorial
    table_usd = f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"
    table_cfg = sim_utils.UsdFileCfg(usd_path=table_usd)

    # Position logic:
    #   - H1 is roughly at x = 0
    #   - Table is placed at x = 0.9 m in front of H1
    #   - z = 1.05 is from the IsaacLab spawn_prims example so the legs sit on the ground
    table_cfg.func("/World/Origin/Table", table_cfg, translation=(2.0, 0.0, 1.05),)

    return robot


def main():
    # Simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=1.0 / 120.0, device=args_cli.device,)
    sim = SimulationContext(sim_cfg)

    # Camera looking at H1 and the table
    sim.set_camera_view(
        eye=[3.0, 0.0, 2.0],   # camera position
        target=[0.4, 0.0, 1.0] # look-at point (between H1 and table)
    )

    # Build scene
    robot = design_scene()

    # Reset simulation so physics & handles are created
    sim.reset()
    print("[INFO] Scene ready: H1 should be standing in front of the table.")

    # Main loop
    while simulation_app.is_running():
        sim.step()
        # Update robot buffers (not strictly necessary if we're not reading state,
        # but good practice if you later add control)
        robot.update(sim.get_physics_dt())


if __name__ == "__main__":
    main()
    simulation_app.close()
