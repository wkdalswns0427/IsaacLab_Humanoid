# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Wrapper for IsaacLab Mimic dataset generation script."""

import os
import subprocess
import sys

ISAACLAB_ROOT = os.environ.get("ISAACLAB_ROOT", "/home/mchang344/RICAL_IsaacLab")
SCRIPT_PATH = os.path.join(ISAACLAB_ROOT, "scripts", "imitation_learning", "isaaclab_mimic", "generate_dataset.py")

if not os.path.exists(SCRIPT_PATH):
    raise FileNotFoundError(f"Mimic generate script not found: {SCRIPT_PATH}")

cmd = [sys.executable, SCRIPT_PATH] + sys.argv[1:]
raise SystemExit(subprocess.call(cmd))
