# H1 Mimic Tasks (Isaac Lab Extension)

This repository is an Isaac Lab extension project that provides a Mimic-compatible environment
for H1 pick-and-lift demonstrations.

## Project layout (high level)

```
../h1_mimic_tasks
  logs/                      # experiment logs (runtime output)
  outputs/                   # saved checkpoints, videos, configs
  practice/                  # scratch files (not part of the package)
  scripts/
    mimic/                   # dataset generation helpers (Isaac Sim)
  source/
    h1_mimic_tasks/          # Python package root
      h1_mimic_tasks/        # extension module
        tasks/
          manager_based/
            h1_mimic_tasks/  # Mimic pick-block task
```

## Environments

Gym registration:
- `source/h1_mimic_tasks/h1_mimic_tasks/tasks/manager_based/h1_mimic_tasks/__init__.py`

Task IDs:
- `H1-Pick-Block-Mimic-v0`
- `Franka-Stack-Cube-Mimic-Practice-v0` (local copied Franka mimic example files)

## Dataset generation (Isaac Sim)

These scripts live in `scripts/mimic/`:
- `record_demos.py`: run the environment and record HDF5 demos
- `annotate_demos.py`: wrap IsaacLab Mimic annotation script
- `generate_dataset.py`: wrap IsaacLab Mimic dataset generation script
- `import_unitree_reference.py`: import Unitree `data.json` episodes into HDF5 reference-motion format

Use `record_demos.py` first to create a raw HDF5, then annotate and generate a Mimic dataset.

Example:

```
python -m pip install -e source/h1_mimic_tasks
python scripts/mimic/record_demos.py --task=H1-Pick-Block-Mimic-v0 --num_demos=5 --demo_length=300
export ISAACLAB_ROOT=/home/mchang344/RICAL_IsaacLab
python scripts/mimic/annotate_demos.py --input_file outputs/mimic/h1_pick_block_raw.hdf5 --output_file outputs/mimic/h1_pick_block_annotated.hdf5
python scripts/mimic/generate_dataset.py --input_file outputs/mimic/h1_pick_block_annotated.hdf5 --output_file outputs/mimic/h1_pick_block_generated.hdf5
```

## Import Unitree Reference Motion

To bridge trajectories from `unitree_sim_isaaclab` into this project:

```bash
python scripts/mimic/import_unitree_reference.py \
  --input_path /path/to/unitree/episode_root_or_data_json \
  --output outputs/mimic/unitree_reference_raw.hdf5 \
  --write_states
```

The importer writes:
- `initial_state` from Unitree `sim_state.init_state`
- `actions` as concatenated `[left_arm, right_arm, left_ee, right_ee]`
- optional per-step `states` when `--write_states` is enabled

Important:
- This creates a reusable reference-motion dataset in IsaacLab HDF5 format.
- It does **not** automatically resolve action-space mismatch with a specific Mimic task (e.g., 6-DoF IK action tasks).

## Franka Mimic Practice (same project workflow)

If you want a known-working mimic baseline before tuning H1, use the local Franka
practice task id registered by this extension.

Example workflow:

```bash
python -m pip install -e source/h1_mimic_tasks
export ISAACLAB_ROOT=/home/mchang344/RICAL_IsaacLab

# 1) Record source demos (teleoperation)
$ISAACLAB_ROOT/isaaclab.sh -p $ISAACLAB_ROOT/scripts/tools/record_demos.py \
  --task Franka-Stack-Cube-Mimic-Practice-v0 \
  --teleop_device keyboard \
  --dataset_file outputs/mimic/franka_stack_raw.hdf5 \
  --num_demos 5 \
  --device cpu

# 2) Annotate mimic signals
python scripts/mimic/annotate_demos.py \
  --task Franka-Stack-Cube-Mimic-Practice-v0 \
  --input_file outputs/mimic/franka_stack_raw.hdf5 \
  --output_file outputs/mimic/franka_stack_annotated.hdf5 \
  --auto \
  --device cpu

# 3) Generate more mimic demos
python scripts/mimic/generate_dataset.py \
  --task Franka-Stack-Cube-Mimic-Practice-v0 \
  --input_file outputs/mimic/franka_stack_annotated.hdf5 \
  --output_file outputs/mimic/franka_stack_generated.hdf5 \
  --generation_num_trials 20 \
  --num_envs 1 \
  --device cpu
```

## Set Robot Movement In `record_demos.py`

`scripts/mimic/record_demos.py` currently records by stepping the env with scripted actions.
Set movement by editing the per-step action tensor inside the demo loop.

Action format for this task:
- `[dx, dy, dz, dRx, dRy, dRz]` (left-arm differential IK delta pose)

Example phased motion (approach -> move down -> lift):

```python
for step_idx in range(args_cli.demo_length):
    actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
    if step_idx < 80:
        actions[:, 0] = 0.01   # move +x
    elif step_idx < 140:
        actions[:, 2] = -0.01  # move down
    elif step_idx < 220:
        actions[:, 2] = 0.01   # move up

    # optional: clamp to safe bounds
    actions = torch.clamp(actions, -1.0, 1.0)
```

Tips:
- Keep deltas small (`0.005` to `0.02`) for stable behavior.
- Start with translation only (`dx, dy, dz`), then add rotation (`dRx, dRy, dRz`).
- With fixed base, this controls arm motion only (not walking/balance).

## Simple IK/controller notes

The Mimic env uses a **differential IK action term** (6‑DoF delta pose) and converts target end‑effector poses
to actions in:

`source/h1_mimic_tasks/h1_mimic_tasks/tasks/manager_based/h1_mimic_tasks/h1_pick_block_mimic_env.py`

Key points:
- Actions are **delta pose** `[dx, dy, dz, dRx, dRy, dRz]` in axis‑angle form.
- `target_eef_pose_to_action()` computes the delta between the target pose and current left‑arm pose.
- `action_to_target_eef_pose()` inverts this to recover target pose from a recorded action.
- The action term is configured in:
  `source/h1_mimic_tasks/h1_mimic_tasks/tasks/manager_based/h1_mimic_tasks/h1_pick_block_mimic_env_cfg.py`

If you want a more accurate IK, replace the placeholder with a task‑space controller (e.g., OSC) or
arm‑specific IK with known link names.

## Notes

- H1-2 is not present in the default IsaacLab assets; this project will fall back to H1 if H1-2 is not found.
- The Mimic environment implements basic end-effector pose access and subtask signals.
