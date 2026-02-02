# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##


gym.register(
    id="H1-Pick-Block-Direct-v0",
    entry_point=f"{__name__}.h1_pick_block_env:H1PickBlockEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_pick_block_env_cfg:H1PickBlockEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1PickBlockPPORunnerCfg",
    },
)

gym.register(
    id="H1-Pick-Block-Dextrous-Direct-v0",
    entry_point=f"{__name__}.h1_pick_block_dextrous_env:H1PickBlockDextrousEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.h1_pick_block_dextrous_env_cfg:H1PickBlockDextrousEnvCfg",
        "rl_games_cfg_entry_point": f"{agents.__name__}:rl_games_ppo_cfg.yaml",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:H1PickBlockDextrousPPORunnerCfg",
    },
)
