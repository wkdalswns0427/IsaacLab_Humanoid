# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

# Register Gym environments.

gym.register(
    id="H1-Pick-Block-Mimic-v0",
    entry_point=f"{__name__}.h1_pick_block_mimic_env:H1PickBlockMimicEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_pick_block_mimic_env_cfg:H1PickBlockMimicEnvCfg",
    },
)
