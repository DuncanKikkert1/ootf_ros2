#!/bin/bash
# =============================================================================
# launch_octo.sh — Launch the Octo VLA inference loop (run_octo_live.py).
#
# Auto-detects a Python environment that has jax and octo installed.
# All extra arguments are forwarded directly to run_octo_live.py.
#
# Examples:
#   bash launch/launch_octo.sh --instruction "pick up the circle"
#   bash launch/launch_octo.sh --usb-camera 0 --show-camera \
#                               --instruction "pick up the circle"
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VLA_SCRIPT="$PROJECT_ROOT/octo_vla/run_octo_live.py"

# --- Auto-detect Python with jax + octo installed ---
PYTHON=""
for py in \
    $(find ~/.pyenv/versions -name "python3" 2>/dev/null | sort -r) \
    $(which python3 2>/dev/null)
do
    if "$py" -c "import jax; import octo" 2>/dev/null; then
        PYTHON="$py"
        break
    fi
done

if [ -z "$PYTHON" ]; then
    echo "ERROR: Could not find a Python installation with jax and octo."
    echo "       Run:  pip install -e ~/Documents/octo"
    echo "       And:  pip install 'jax[cuda12_pip]==0.4.20' -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html"
    exit 1
fi

echo "Using Python : $PYTHON"
echo "VLA script   : $VLA_SCRIPT"
echo ""

"$PYTHON" "$VLA_SCRIPT" "$@"
