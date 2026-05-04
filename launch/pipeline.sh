#!/bin/bash
# =============================================================================
# pipeline.sh — Collect synthetic episodes → convert to TFDS → finetune Octo.
#
# Usage:
#   bash launch/pipeline.sh --output-dir data/exp_01 --n-episodes 200
#   bash launch/pipeline.sh --output-dir data/exp_01 --collect-only
#   bash launch/pipeline.sh --output-dir data/exp_01 --convert-only
#   bash launch/pipeline.sh --output-dir data/exp_01 --finetune-only
#
# All unrecognised flags (--n-episodes, --pick-x, --seed, etc.) are forwarded
# to collect.py.  Training flags (--pretrained-path, --finetune-mode,
# --n-finetune-steps, --batch-size) are forwarded to pipeline.py.
# =============================================================================

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ── Parse arguments ───────────────────────────────────────────────────────────
OUTPUT_DIR=""
PHASE="all"
COLLECT_ARGS=()
TRAIN_ARGS=()

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)                       OUTPUT_DIR="$2";               shift 2 ;;
        --collect-only)                     PHASE="collect";               shift   ;;
        --convert-only)                     PHASE="convert";               shift   ;;
        --finetune-only)                    PHASE="finetune";              shift   ;;
        --pretrained-path|--finetune-mode)  TRAIN_ARGS+=("$1" "$2");      shift 2 ;;
        --n-finetune-steps|--batch-size)    TRAIN_ARGS+=("$1" "$2");      shift 2 ;;
        *)                                  COLLECT_ARGS+=("$1");          shift   ;;
    esac
done

if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: --output-dir is required."
    echo "Usage: bash launch/pipeline.sh --output-dir data/exp_01 [options]"
    exit 1
fi

RAW_DIR="$OUTPUT_DIR/raw"

# ── Auto-detect Python with isaacsim ─────────────────────────────────────────
detect_isaac_python() {
    for py in $(find ~/.pyenv/versions -name "python3" 2>/dev/null | sort -r) "$(which python3 2>/dev/null)"; do
        "$py" -c "import isaacsim" 2>/dev/null && echo "$py" && return
    done
}

# ── Auto-detect Python with jax + octo (for finetune) ────────────────────────
detect_train_python() {
    for py in $(find ~/.pyenv/versions -name "python3" 2>/dev/null | sort -r) "$(which python3 2>/dev/null)"; do
        "$py" -c "import jax; import octo" 2>/dev/null && echo "$py" && return
    done
    which python3 2>/dev/null
}

# ── Collect ───────────────────────────────────────────────────────────────────
if [[ "$PHASE" == "all" || "$PHASE" == "collect" ]]; then
    ISAAC_PY=$(detect_isaac_python)
    if [ -z "$ISAAC_PY" ]; then
        echo "ERROR: No Python with isaacsim found."
        echo "       Install: pip install isaacsim==4.5.0 --extra-index-url https://pypi.nvidia.com"
        exit 1
    fi
    echo "=== Collect ==="
    echo "  Python : $ISAAC_PY"
    echo "  Output : $RAW_DIR"
    echo ""
    "$ISAAC_PY" "$PROJECT_ROOT/src/isaac/collect.py" \
        --output-dir "$RAW_DIR" \
        "${COLLECT_ARGS[@]}"

    # Isaac Sim's simulation_app.close() calls sys.exit(0) internally, so the
    # exit code is unreliable. Check for actual output instead.
    EPISODE_COUNT=$(find "$RAW_DIR" -name "*.npz" -maxdepth 1 2>/dev/null | wc -l)
    if [ "$EPISODE_COUNT" -eq 0 ]; then
        echo "ERROR: Collect phase produced no episodes in $RAW_DIR — check output above."
        exit 1
    fi
    echo "  Episodes collected: $EPISODE_COUNT"
fi

# ── Convert + finetune ────────────────────────────────────────────────────────
if [[ "$PHASE" == "all" || "$PHASE" == "convert" || "$PHASE" == "finetune" ]]; then
    TRAIN_PY=$(detect_train_python)
    if [ -z "$TRAIN_PY" ]; then
        echo "ERROR: No Python found for training."
        exit 1
    fi

    PHASE_FLAG=""
    [[ "$PHASE" == "convert"  ]] && PHASE_FLAG="--convert-only"
    [[ "$PHASE" == "finetune" ]] && PHASE_FLAG="--finetune-only"

    echo "=== Train ==="
    echo "  Python  : $TRAIN_PY"
    echo "  Raw dir : $RAW_DIR"
    echo ""
    "$TRAIN_PY" "$PROJECT_ROOT/src/training/pipeline.py" \
        --raw-dir    "$RAW_DIR" \
        --output-dir "$OUTPUT_DIR" \
        $PHASE_FLAG \
        "${TRAIN_ARGS[@]}" || exit 1
fi
