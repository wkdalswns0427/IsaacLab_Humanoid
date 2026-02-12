# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Local Franka mimic practice task registration and modules."""

import gymnasium as gym


gym.register(
    id="Franka-Stack-Cube-Mimic-Practice-v0",
    entry_point=(f"{__name__}.franka_stack_ik_rel_mimic_env:FrankaCubeStackIKRelMimicEnv"),
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.franka_stack_ik_rel_mimic_env_cfg:FrankaCubeStackIKRelMimicEnvCfg"
        ),
    },
)
