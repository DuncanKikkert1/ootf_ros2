#!/bin/bash
# =============================================================================
# launch_publisher.sh — Sets up the ROS2 environment and runs ros_publisher.py
# Run from anywhere: bash launch/launch_publisher.sh
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ISAAC_WS="/home/affix/IsaacSim-ros_workspaces"
PYENV_PYTHON="/home/affix/.pyenv/versions/3.11.14/envs/python3.11/bin/activate"
ISAAC_BRIDGE="/home/affix/.pyenv/versions/3.11.14/envs/python3.11/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/jazzy"

source "$ISAAC_WS/build_ws/jazzy/jazzy_ws/install/local_setup.bash"
source "$ISAAC_WS/build_ws/jazzy/isaac_sim_ros_ws/install/local_setup.bash"

export ROS_DISTRO=jazzy
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$ISAAC_BRIDGE/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$ISAAC_BRIDGE/rclpy:$PYTHONPATH

source "$PYENV_PYTHON"

python3 "$PROJECT_ROOT/src/ros_publisher.py"
