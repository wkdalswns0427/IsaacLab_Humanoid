# H1 Dext Tasks (Isaac Lab Extension)

This repository is an Isaac Lab extension project. It contains the direct RL H1 pick-and-lift tasks
(standard and dextrous contact variants).

## Project layout (high level)

```
../h1_dext_tasks
  logs/                      # experiment logs (runtime output)
  outputs/                   # saved checkpoints, videos, configs
  practice/                  # scratch files (not part of the package)
  scripts/                   # launch helpers (train, list_envs, zero_agent, random_agent)
  source/
    h1_dext_tasks/           # Python package root
      h1_dext_tasks/         # extension module
        tasks/
          direct/
            h1_dext_tasks/   # direct RL tasks (this is where H1 pick tasks live)
          manager_based/
            h1_dext_tasks/   # manager-based tasks (locomotion templates)
        ui_extension_example.py
```

## Where tasks are registered

Gym environments are registered in:

```
source/h1_dext_tasks/h1_dext_tasks/tasks/direct/h1_dext_tasks/__init__.py
source/h1_dext_tasks/h1_dext_tasks/tasks/direct/h1_dext_tasks_marl/__init__.py
source/h1_dext_tasks/h1_dext_tasks/tasks/manager_based/h1_dext_tasks/__init__.py
```

Those registrations point to:
- an environment class (runtime logic)
- a config class (scene, robot, rewards, observation sizes)
- an RL config entry point (RSL-RL or rl_games)

## H1 pick tasks (direct RL)

### 1) H1-Pick-Block-Direct-v0 (non-dextrous)

Files:
- Env: `source/h1_dext_tasks/h1_dext_tasks/tasks/direct/h1_dext_tasks/h1_pick_block_env.py`
- Config: `source/h1_dext_tasks/h1_dext_tasks/tasks/direct/h1_dext_tasks/h1_pick_block_env_cfg.py`
- Gym registration: `source/h1_dext_tasks/h1_dext_tasks/tasks/direct/h1_dext_tasks/__init__.py`

Key behavior:
- Spawns Unitree H1 (minimal) + 20 cm cube on the ground.
- Actions: joint position targets (19 DoF, scaled).
- Observations: joint pos/vel + cube pos/vel + end-effector to cube vector.
- Rewards: distance to cube, lift height, success height, action penalty.

### 2) H1-Pick-Block-Dextrous-Direct-v0 (dextrous contact)

Files:
- Env: `source/h1_dext_tasks/h1_dext_tasks/tasks/direct/h1_dext_tasks/h1_pick_block_dextrous_env.py`
- Config: `source/h1_dext_tasks/h1_dext_tasks/tasks/direct/h1_dext_tasks/h1_pick_block_dextrous_env_cfg.py`
- Gym registration: `source/h1_dext_tasks/h1_dext_tasks/tasks/direct/h1_dext_tasks/__init__.py`
- RSL-RL runner: `source/h1_dext_tasks/h1_dext_tasks/tasks/direct/h1_dext_tasks/agents/rsl_rl_ppo_cfg.py`

Key behavior:
- Same base task as H1-Pick-Block-Direct-v0.
- Adds a ContactSensor on the robot.
- Adds a hand-contact reward term (based on contact force on hand/wrist bodies).

## Removed tasks

Manager-based templates and MARL tasks are not registered in this project.

## How to run

Install editable package:

```
python -m pip install -e source/h1_dext_tasks
```

List envs:

```
python scripts/list_envs.py
```

Run a task:

```
python scripts/<RL_LIBRARY>/train.py --task=<TASK_NAME>
```

Zero-action sanity check:

```
python scripts/zero_agent.py --task=<TASK_NAME>
```

## How to play a trained result

1) Find a checkpoint in:
```
logs/rsl_rl/<experiment_name>/<run_timestamp>/checkpoints/
```

2) Run the RSL-RL play script:
```
python scripts/rsl_rl/play.py --task=<TASK_NAME> --checkpoint=<checkpoint_file> --load_run=<run_timestamp> 
```

## Notes for customization

- End-effector and hand-contact body names are regex-based in the H1 pick configs.
  Update these if your H1 asset uses different link names:
  - `left_ee_body_names`, `right_ee_body_names`
  - `hand_contact_body_names`
- Cube size is 0.2 m (20 cm) and spawn position is set in the config.
- Reward scales, lift height, and success height are all in the config files.
