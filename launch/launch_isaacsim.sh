#!/bin/bash
# =============================================================================
# launch_isaacsim.sh — Sets up the ROS2 + Isaac Sim environment and runs test.py
# Run from anywhere: bash launch/launch_isaacsim.sh
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISAAC_WS="/home/affix/IsaacSim-ros_workspaces"
PYENV_PYTHON="/home/affix/.pyenv/versions/3.11.14/envs/python3.11/bin/activate"
ISAAC_BRIDGE="/home/affix/.pyenv/versions/3.11.14/envs/python3.11/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy"

# ROS2 workspaces
source "$ISAAC_WS/build_ws/jazzy/jazzy_ws/install/local_setup.bash"
source "$ISAAC_WS/build_ws/jazzy/isaac_sim_ros_ws/install/local_setup.bash"

# Isaac Sim internal ROS2 bridge libraries (Python 3.11 compatible)
# Prepend so these take priority over the Python 3.12 workspace paths set by local_setup.bash
export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$ISAAC_BRIDGE/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$ISAAC_BRIDGE/rclpy:$PYTHONPATH

# Python environment
source "$PYENV_PYTHON"

python3 "$PROJECT_ROOT/src/simulation.py"
