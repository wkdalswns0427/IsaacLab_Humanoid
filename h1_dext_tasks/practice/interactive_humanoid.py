import argparse
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="H1 standing in front of a table.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---------- Rest of imports (after app launch) ----------
import isaacsim.core.utils.prims as prim_utils
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab_assets import H1_MINIMAL_CFG
