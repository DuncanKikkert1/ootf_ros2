#!/bin/bash
# =============================================================================
# debug.sh — Debug and diagnostic tools.
#
# Usage:
#   bash launch/debug.sh sim      # Isaac Sim ROS2 node only
#   bash launch/debug.sh bridge   # TCP → ROS2 bridge only
#   bash launch/debug.sh joint    # Doosan joint service client only
#   bash launch/debug.sh policy   # Policy diagnosis (GT vs predicted actions)
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COMPONENT=$1
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="$PROJECT_ROOT/debug/logs"
mkdir -p "$LOG_DIR/sim" "$LOG_DIR/collect" "$LOG_DIR/gripper"

if [ -z "$COMPONENT" ]; then
    echo "Usage: bash launch/debug.sh [sim|bridge|joint|policy]"
    exit 1
fi

# ── Detect ROS distro ────────────────────────────────────────────────────────
if [ -z "$ROS_DISTRO" ]; then
    ROS_DISTRO=$(ls /opt/ros/ 2>/dev/null | head -1)
    [ -z "$ROS_DISTRO" ] && { echo "ERROR: No ROS distro found in /opt/ros/"; exit 1; }
fi

case "$COMPONENT" in

    sim)
        # ── Isaac Sim node (requires isaacsim Python + ROS2 bridge) ──────────
        ISAAC_PY=""
        for py in $(find ~/.pyenv/versions -name "python3" 2>/dev/null | sort -r) "$(which python3 2>/dev/null)"; do
            "$py" -c "import isaacsim" 2>/dev/null && ISAAC_PY="$py" && break
        done
        [ -z "$ISAAC_PY" ] && { echo "ERROR: No Python with isaacsim found."; exit 1; }

        ISAAC_BRIDGE=$("$ISAAC_PY" -c "
import os, isaacsim
print(os.path.join(os.path.dirname(isaacsim.__file__), 'exts', 'isaacsim.ros2.bridge', '$ROS_DISTRO'))
" 2>/dev/null)

        if [ -z "$ISAAC_WS" ]; then
            for candidate in "$HOME/IsaacSim-ros_workspaces" "$HOME/isaac_ros_ws" "/opt/IsaacSim-ros_workspaces"; do
                [ -d "$candidate/build_ws/$ROS_DISTRO" ] && ISAAC_WS="$candidate" && break
            done
            [ -z "$ISAAC_WS" ] && { echo "ERROR: Isaac Sim ROS workspace not found. Set ISAAC_WS=/path."; exit 1; }
        fi

        source "$ISAAC_WS/build_ws/$ROS_DISTRO/${ROS_DISTRO}_ws/install/local_setup.bash"
        source "$ISAAC_WS/build_ws/$ROS_DISTRO/isaac_sim_ros_ws/install/local_setup.bash"

        export ROS_DISTRO
        export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
        export LD_LIBRARY_PATH="$ISAAC_BRIDGE/lib:$LD_LIBRARY_PATH"
        export PYTHONPATH="$ISAAC_BRIDGE/rclpy:$PYTHONPATH"

        LOG="$LOG_DIR/sim/sim_${TIMESTAMP}.log"
        echo "Starting sim_node.py  (ROS_DISTRO=$ROS_DISTRO)"
        echo "Logging to $LOG"
        "$ISAAC_PY" -u "$PROJECT_ROOT/src/isaac/sim_node.py" 2>&1 | tee "$LOG"
        ;;

    bridge)
        # ── TCP → ROS2 bridge ────────────────────────────────────────────────
        source "/opt/ros/$ROS_DISTRO/setup.bash"
        export ROS_DISTRO
        export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

        LOG="$LOG_DIR/bridge_${TIMESTAMP}.log"
        echo "Starting tcp_ros_bridge.py  (ROS_DISTRO=$ROS_DISTRO)"
        echo "Logging to $LOG"
        python3 "$PROJECT_ROOT/src/vla/tcp_ros_bridge.py" 2>&1 | tee "$LOG"
        ;;

    joint)
        # ── Doosan joint service client ──────────────────────────────────────
        source "/opt/ros/$ROS_DISTRO/setup.bash"
        export ROS_DISTRO
        export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

        LOG="$LOG_DIR/joint_${TIMESTAMP}.log"
        echo "Starting joint_service_client.py  (ROS_DISTRO=$ROS_DISTRO)"
        echo "Logging to $LOG"
        python3 "$PROJECT_ROOT/src/vla/joint_service_client.py" 2>&1 | tee "$LOG"
        ;;

    gripper-test)
        # ── Surface gripper test (requires sim to be running) ────────────────
        source "/opt/ros/$ROS_DISTRO/setup.bash"
        export ROS_DISTRO
        export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

        LOG="$LOG_DIR/gripper/gripper_${TIMESTAMP}.log"
        echo "Running gripper test  (ROS_DISTRO=$ROS_DISTRO)"
        echo "Logging to $LOG"
        python3 "$PROJECT_ROOT/debug/debug_gripper.py" 2>&1 | tee "$LOG"
        ;;

    replay)
        # ── GT action replay (requires sim to be running) ─────────────────────
        source "/opt/ros/$ROS_DISTRO/setup.bash"
        export ROS_DISTRO
        export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

        mkdir -p "$LOG_DIR/replay"
        LOG="$LOG_DIR/replay/replay_${TIMESTAMP}.log"
        echo "Running GT action replay  (ROS_DISTRO=$ROS_DISTRO)"
        echo "Logging to $LOG"
        shift   # drop "replay" so remaining args (--npz, --step-delay, …) pass through
        python3 "$PROJECT_ROOT/debug/debug_replay.py" "$@" 2>&1 | tee "$LOG"
        ;;

    collect)
        # ── Dry-run episode collector (no files written) ─────────────────────
        ISAAC_PY=""
        for py in $(find ~/.pyenv/versions -name "python3" 2>/dev/null | sort -r) "$(which python3 2>/dev/null)"; do
            "$py" -c "import isaacsim" 2>/dev/null && ISAAC_PY="$py" && break
        done
        [ -z "$ISAAC_PY" ] && { echo "ERROR: No Python with isaacsim found."; exit 1; }

        echo "Using Python : $ISAAC_PY"
        echo ""
        LOG="$LOG_DIR/collect/collect_debug_${TIMESTAMP}.log"
        echo "Logging to $LOG"
        shift   # drop "collect" so remaining args (--seq, --seed, …) pass through
        exec "$ISAAC_PY" -u "$PROJECT_ROOT/debug/debug_episode_collect.py" "$@" 2>&1 | tee "$LOG"
        ;;

    policy)
        # ── Policy diagnosis (GT vs predicted actions) ───────────────────────
        OCTO_PY=""
        for py in $(find ~/.pyenv/versions -name "python3" 2>/dev/null | sort -r) "$(which python3 2>/dev/null)"; do
            "$py" -c "import jax; import octo" 2>/dev/null && OCTO_PY="$py" && break
        done
        [ -z "$OCTO_PY" ] && { echo "ERROR: No Python with jax + octo found."; exit 1; }

        echo "Using Python : $OCTO_PY"
        echo ""
        shift   # drop "policy" so remaining args pass through to the script
        exec "$OCTO_PY" "$PROJECT_ROOT/debug/debug_policy.py" "$@"
        ;;

    *)
        echo "ERROR: Unknown component '$COMPONENT'."
        echo "Usage: bash launch/debug.sh [sim|bridge|joint|collect|policy|replay|gripper-test]"
        exit 1
        ;;
esac
