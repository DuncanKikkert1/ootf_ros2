#!/bin/bash
# =============================================================================
# real_inference.sh — Real Doosan robot + MechEye camera + Octo VLA.
#
# Starts tcp_ros_bridge.py, joint_service_client.py, and run_octo_live.py
# as parallel processes. Ctrl+C shuts all three down cleanly.
#
# Usage:
#   bash launch/real_inference.sh --instruction "pick up the object"
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Logging ──────────────────────────────────────────────────────────────────
LOG_DIR="$PROJECT_ROOT/debug/logs/run/real"
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

# ── Detect Python with jax + octo ────────────────────────────────────────────
OCTO_PY=""
for py in $(find ~/.pyenv/versions -name "python3" 2>/dev/null | sort -r) "$(which python3 2>/dev/null)"; do
    "$py" -c "import jax; import octo" 2>/dev/null && OCTO_PY="$py" && break
done
[ -z "$OCTO_PY" ] && { echo "ERROR: No Python with jax + octo found."; exit 1; }

# ── ROS2 setup ───────────────────────────────────────────────────────────────
source "/opt/ros/$ROS_DISTRO/setup.bash"
export ROS_DISTRO
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONPATH="$PROJECT_ROOT/src/comm:$PYTHONPATH"

echo "Using ROS_DISTRO : $ROS_DISTRO"
echo "Using Octo       : $OCTO_PY"
echo ""

# ── Launch ───────────────────────────────────────────────────────────────────
python3 "$PROJECT_ROOT/src/vla/tcp_ros_bridge.py" &
PID_BRIDGE=$!

python3 "$PROJECT_ROOT/src/vla/joint_service_client.py" &
PID_JOINT=$!

"$OCTO_PY" "$PROJECT_ROOT/src/vla/run_octo_live.py" "$@" &
PID_OCTO=$!

trap "echo ''; echo 'Shutting down...'; kill $PID_BRIDGE $PID_JOINT $PID_OCTO 2>/dev/null; wait $PID_BRIDGE $PID_JOINT $PID_OCTO 2>/dev/null" INT TERM
wait $PID_BRIDGE $PID_JOINT $PID_OCTO
