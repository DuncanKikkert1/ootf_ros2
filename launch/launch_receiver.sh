#!/bin/bash
# =============================================================================
# launch_receiver.sh — Runs receiver.py (no ROS2 sourcing required)
# Run from anywhere: bash launch/launch_receiver.sh
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python3 "$PROJECT_ROOT/src/scripts/receiver.py"
