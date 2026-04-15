#!/bin/bash
# =============================================================================
# launch_joint_client.sh — Launch script for joint_service_client.py.
# Auto-detects ROS distro and sources the standard ROS2 setup.
# Run from anywhere: bash launch/launch_joint_client.sh
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# --- Auto-detect ROS distro ---
if [ -z "$ROS_DISTRO" ]; then
    ROS_DISTRO=$(ls /opt/ros/ 2>/dev/null | head -1)
    [ -z "$ROS_DISTRO" ] && { echo "ERROR: No ROS distro found in /opt/ros/"; exit 1; }
fi

echo "Using ROS_DISTRO : $ROS_DISTRO"

source "/opt/ros/$ROS_DISTRO/setup.bash"

export ROS_DISTRO
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

python3 "$PROJECT_ROOT/src/joint_service_client.py"
