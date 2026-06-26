#!/bin/bash
# =============================================================================
# sim_inference.sh — Isaac Sim scene + Octo VLA inference loop.
#
# Starts sim_node.py (Isaac Sim + ROS2) and run_octo_live.py (Octo VLA) as
# parallel processes. Ctrl+C shuts both down cleanly.
#
# Usage:
#   bash launch/sim_inference.sh --instruction "pick up the object"
#
# Optional env override:
#   ISAAC_WS=/path/to/workspace bash launch/sim_inference.sh ...
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR="$PROJECT_ROOT/debug/logs/run/sim"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/$(date +%Y%m%d_%H%M%S).log"
export PYTHONUNBUFFERED=1
exec > >(stdbuf -oL tee -a "$LOG_FILE") 2>&1
echo "Logging to $LOG_FILE"

# ── Detect ROS distro ────────────────────────────────────────────────────────
if [ -z "$ROS_DISTRO" ]; then
    ROS_DISTRO=$(ls /opt/ros/ 2>/dev/null | head -1)
    [ -z "$ROS_DISTRO" ] && { echo "ERROR: No ROS distro found in /opt/ros/"; exit 1; }
fi

# ── Detect Python with isaacsim ──────────────────────────────────────────────
ISAAC_PY=""
for py in $(find ~/.pyenv/versions -name "python3" 2>/dev/null | sort -r) "$(which python3 2>/dev/null)"; do
    "$py" -c "import isaacsim" 2>/dev/null && ISAAC_PY="$py" && break
done
[ -z "$ISAAC_PY" ] && { echo "ERROR: No Python with isaacsim found."; exit 1; }

# ── Detect Python with jax + octo ────────────────────────────────────────────
OCTO_PY=""
for py in $(find ~/.pyenv/versions -name "python3" 2>/dev/null | sort -r) "$(which python3 2>/dev/null)"; do
    "$py" -c "import jax; import octo" 2>/dev/null && OCTO_PY="$py" && break
done
[ -z "$OCTO_PY" ] && { echo "ERROR: No Python with jax + octo found."; exit 1; }

# ── Isaac Sim bridge + workspace ─────────────────────────────────────────────
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
export PYTHONPATH="$ISAAC_BRIDGE/rclpy:$PROJECT_ROOT/src/comm:$PYTHONPATH"

echo "Using ROS_DISTRO : $ROS_DISTRO"
echo "Using Isaac Sim  : $ISAAC_PY"
echo "Using Octo       : $OCTO_PY"
echo ""

# ── Launch ───────────────────────────────────────────────────────────────────
"$ISAAC_PY" "$PROJECT_ROOT/src/isaac/sim_node.py" &
PID_SIM=$!

# sim_node.py touches /tmp/sim_node_ready just before entering its main loop,
# after USD load, render product creation, and physics settling are all done.
# This is more reliable than `ros2 node list` which depends on DDS discovery
# timing across different Python environments.
rm -f /tmp/sim_node_ready
echo "Waiting for Isaac Sim to finish initialising..."
TIMEOUT=300
ELAPSED=0
while [ ! -f /tmp/sim_node_ready ]; do
    if ! kill -0 "$PID_SIM" 2>/dev/null; then
        echo "ERROR: Isaac Sim process died during startup."
        exit 1
    fi
    if [ "$ELAPSED" -ge "$TIMEOUT" ]; then
        echo "ERROR: Timed out waiting for Isaac Sim after ${TIMEOUT}s."
        kill "$PID_SIM" 2>/dev/null
        exit 1
    fi
    sleep 2
    ELAPSED=$((ELAPSED + 2))
done
echo "Isaac Sim ready. Starting Octo VLA..."

# XLA_PYTHON_CLIENT_PREALLOCATE=false: JAX allocates on demand instead of
# grabbing a fixed pool (~75% of VRAM) at startup, leaving room for the RTX renderer.
#
# Only the sim-specific routing flags are set here; everything else (window-size 2,
# step-delay 0.2, action-horizon 4, grip-hold-steps 20, temporal-ensemble OFF) is now
# the run_octo_live.py default. Pass overrides after "$@", e.g.:
#   --temporal-ensemble                 enable action smoothing
#   --step-delay 0.0167                 60 Hz (collection --action-stride 1)
XLA_PYTHON_CLIENT_PREALLOCATE=false "$OCTO_PY" "$PROJECT_ROOT/src/vla/run_octo_live.py" \
    --ros2-camera \
    --ros2-output \
    "$@" &
PID_OCTO=$!

trap "echo ''; echo 'Shutting down...'; kill $PID_SIM $PID_OCTO 2>/dev/null; wait $PID_SIM $PID_OCTO 2>/dev/null" INT TERM
wait $PID_SIM $PID_OCTO
