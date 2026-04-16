#!/bin/bash
# =============================================================================
# launch.sh — Top-level launcher for the ootf_ros2 pipeline.
#
# Usage:
#   bash launch/launch.sh virtual                              # TCP bridge + Isaac Sim
#   bash launch/launch.sh real                                 # TCP bridge + Doosan joint service client
#   bash launch/launch.sh e2e [launch_octo.sh args]            # TCP bridge + joint client + Octo VLA
# =============================================================================

LAUNCH_DIR="$(cd "$(dirname "$0")" && pwd)"

MODE=$1

if [ -z "$MODE" ]; then
    echo "ERROR: No mode specified."
    echo "Usage: bash launch/launch.sh [virtual|real]"
    exit 1
fi

case "$MODE" in
    virtual)
        echo "=== Starting in VIRTUAL mode (Isaac Sim) ==="
        bash "$LAUNCH_DIR/launch_bridge.sh" &
        PID_BRIDGE=$!
        bash "$LAUNCH_DIR/launch_isaacsim.sh" &
        PID_SECOND=$!
        trap "echo ''; echo 'Shutting down...'; kill $PID_BRIDGE $PID_SECOND 2>/dev/null; wait $PID_BRIDGE $PID_SECOND 2>/dev/null" INT TERM
        wait $PID_BRIDGE $PID_SECOND
        ;;
    real)
        echo "=== Starting in REAL mode (Doosan robot) ==="
        bash "$LAUNCH_DIR/launch_bridge.sh" &
        PID_BRIDGE=$!
        bash "$LAUNCH_DIR/launch_joint_client.sh" &
        PID_SECOND=$!
        trap "echo ''; echo 'Shutting down...'; kill $PID_BRIDGE $PID_SECOND 2>/dev/null; wait $PID_BRIDGE $PID_SECOND 2>/dev/null" INT TERM
        wait $PID_BRIDGE $PID_SECOND
        ;;
    e2e)
        echo "=== Starting in E2E mode (Octo VLA + TCP bridge + Doosan robot) ==="
        bash "$LAUNCH_DIR/launch_bridge.sh" &
        PID_BRIDGE=$!
        bash "$LAUNCH_DIR/launch_joint_client.sh" &
        PID_JOINT=$!
        bash "$LAUNCH_DIR/launch_octo.sh" "${@:2}" &
        PID_OCTO=$!
        trap "echo ''; echo 'Shutting down...'; kill $PID_BRIDGE $PID_JOINT $PID_OCTO 2>/dev/null; wait $PID_BRIDGE $PID_JOINT $PID_OCTO 2>/dev/null" INT TERM
        wait $PID_BRIDGE $PID_JOINT $PID_OCTO
        ;;
    *)
        echo "ERROR: Unknown mode '$MODE'. Use 'virtual', 'real', or 'e2e'."
        exit 1
        ;;
esac
