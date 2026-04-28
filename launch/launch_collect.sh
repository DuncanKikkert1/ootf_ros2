#!/bin/bash
# =============================================================================
# launch_collect.sh — Launch synthetic data collection + Octo finetuning.
#
# Auto-detects Python with isaacsim installed.  All arguments are forwarded
# to collect_and_train.py.
#
# Examples:
#   # Full pipeline (collect → convert → finetune):
#   bash launch/launch_collect.sh --output-dir data/exp_01 --n-episodes 500
#
#   # Collect only (verify camera view / waypoints before committing):
#   bash launch/launch_collect.sh --output-dir data/exp_01 --collect-only
#
#   # Finetune on data collected earlier:
#   bash launch/launch_collect.sh --output-dir data/exp_01 --finetune-only
#
#   # Custom workspace geometry:
#   bash launch/launch_collect.sh --output-dir data/exp_02 \
#       --n-episodes 1000 --table-z 0.78 --pick-x 0.40 0.60
#
#   # Different finetune mode:
#   bash launch/launch_collect.sh --output-dir data/exp_03 \
#       --finetune-mode full --n-finetune-steps 50000
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
COLLECT_SCRIPT="$PROJECT_ROOT/src/collect_and_train.py"

# --- Auto-detect Python with isaacsim ---
PYTHON=""
for py in \
    $(find ~/.pyenv/versions -name "python3" 2>/dev/null | sort -r) \
    $(which python3 2>/dev/null)
do
    if "$py" -c "import isaacsim" 2>/dev/null; then
        PYTHON="$py"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Could not find a Python installation with isaacsim."
    echo "       Install with: pip install isaacsim==4.5.0 --extra-index-url https://pypi.nvidia.com"
    exit 1
fi

echo "Using Python     : $PYTHON"
echo "Output script    : $COLLECT_SCRIPT"
echo ""

"$PYTHON" "$COLLECT_SCRIPT" "$@"
