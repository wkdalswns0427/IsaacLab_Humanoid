# H1-2 Dexterous Task Controller

This project intends to solve dexterous manipulation and locomotion tasks with Unitree H1-2 robot in hazardous field environment.

## H1_DEXT_TASK (trials)
Model Free trials on modified PPO induced policy

## H1_mimic_tasks (trials)
unitree_sim_isaaclab(https://github.com/unitreerobotics/unitree_sim_isaaclab) locally installed environment.
use official unitree sim env for reference motion for mimic package in isaac
**must use unitree based conda env upon 3.11.14(current)** although the teleimager requires 3.10 walk around by locally installing 
```
cat .gitmodules 2>/dev/null
git submodule status
git submodule update --init --recursive
find teleimager -maxdepth 3 -type f
pip3 install -e .
```
in teleimager and redirecting python directory
```
export PYTHONPATH=$PWD/teleimager/src:$PWD:$PYTHONPATH
python sim_main.py --device cpu --enable_cameras \
  --task Isaac-PickPlace-Cylinder-G129-Dex1-Joint \
  --enable_dex1_dds --robot_type g129
```
the package has modifications regarding dependencies