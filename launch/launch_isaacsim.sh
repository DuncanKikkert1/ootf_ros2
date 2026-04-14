#!/bin/bash
# =============================================================================
# launch_isaacsim.sh — Universal launch script for simulation.py.
# Auto-detects ROS distro, Python, and Isaac Sim paths.
# Run from anywhere: bash launch/launch_isaacsim.sh
#
# If the Isaac Sim ROS workspace is in a non-standard location, set ISAAC_WS:
#   ISAAC_WS=/path/to/workspace bash launch/launch_isaacsim.sh
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- Auto-detect ROS distro ---
if [ -z "$ROS_DISTRO" ]; then
    ROS_DISTRO=$(ls /opt/ros/ 2>/dev/null | head -1)
    [ -z "$ROS_DISTRO" ] && { echo "ERROR: No ROS distro found in /opt/ros/"; exit 1; }
fi

# --- Auto-detect Python with isaacsim ---
PYTHON=""
for py in $(find ~/.pyenv/versions -name "python3" 2>/dev/null); do
    if $py -c "import isaacsim" 2>/dev/null; then
        PYTHON=$py; break
    fi
done
[ -z "$PYTHON" ] && PYTHON=$(which python3)

# --- Auto-detect Isaac Sim bridge path ---
ISAAC_BRIDGE=$($PYTHON -c "
import os, isaacsim
print(os.path.join(os.path.dirname(isaacsim.__file__), 'exts', 'isaacsim.ros2.bridge', '$ROS_DISTRO'))
")

# --- Auto-detect Isaac Sim ROS workspace ---
if [ -z "$ISAAC_WS" ]; then
    for candidate in \
        "$HOME/IsaacSim-ros_workspaces" \
        "$HOME/isaac_ros_ws" \
        "/opt/IsaacSim-ros_workspaces"
    do
        if [ -d "$candidate/build_ws/$ROS_DISTRO" ]; then
            ISAAC_WS=$candidate; break
        fi
    done
    [ -z "$ISAAC_WS" ] && { echo "ERROR: Isaac Sim ROS workspace not found. Set ISAAC_WS=/path/to/workspace"; exit 1; }
fi

echo "Using ROS_DISTRO : $ROS_DISTRO"
echo "Using Python     : $PYTHON"
echo "Using ISAAC_WS   : $ISAAC_WS"
echo "Using bridge     : $ISAAC_BRIDGE"

source "$ISAAC_WS/build_ws/$ROS_DISTRO/${ROS_DISTRO}_ws/install/local_setup.bash"
source "$ISAAC_WS/build_ws/$ROS_DISTRO/isaac_sim_ros_ws/install/local_setup.bash"

export ROS_DISTRO
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export LD_LIBRARY_PATH=$ISAAC_BRIDGE/lib:$LD_LIBRARY_PATH
export PYTHONPATH=$ISAAC_BRIDGE/rclpy:$PYTHONPATH

$PYTHON "$PROJECT_ROOT/src/simulation.py"
