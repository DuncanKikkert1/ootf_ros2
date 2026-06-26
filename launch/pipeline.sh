#!/bin/bash
# =============================================================================
# pipeline.sh — Collect synthetic episodes → convert to TFDS → finetune Octo.
#
# Usage:
#   bash launch/pipeline.sh --output-dir data/exp_01 --preset phased-cube
#   bash launch/pipeline.sh --output-dir data/exp_01 --n-episodes 200
#   bash launch/pipeline.sh --output-dir data/exp_01 --collect-only
#   bash launch/pipeline.sh --output-dir data/exp_01 --convert-only
#   bash launch/pipeline.sh --output-dir data/exp_01 --finetune-only
#   bash launch/pipeline.sh --output-dir data/exp_dagger --dagger-rollouts <dir>
#
# --preset phased-cube  expands to the canonical collection flags
#                       (--phased --forced-obj-sequence cube). Training defaults
#                       (--finetune-mode full, --task-modality text_conditioned)
#                       already match the canonical run, so nothing else is needed.
#
# --dagger-rollouts <dir>  DAgger mode: instead of collecting, relabel the
#                       on-policy rollout traces in <dir> (recorded by
#                       success_rate.py --record-dagger) with the sim oracle,
#                       aggregate them with the expert set, then convert +
#                       finetune.  Related flags:
#                         --expert-raw <dir>   expert episodes merged with the
#                                              corrective set
#                                              (default: data/exp_truecolor2/raw)
#                         --dagger-phase NAME  which rollout phase to relabel
#                                              (default: pick)
#                       NOTE: pass --pretrained-path and OOTF_PROPRIO=1 explicitly.
#                       The DAgger finetune inherits NEITHER the base checkpoint
#                       nor the proprio setting from the expert set, so a wrong
#                       base silently drops proprio → an image-only model that
#                       flails/rams in closed loop.
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
HEAD_ARGS=()
DAGGER_ROLLOUTS=""                              # set → DAgger mode: relabel+aggregate instead of collect
EXPERT_RAW="$PROJECT_ROOT/data/exp_truecolor2/raw"   # default expert set merged with the corrective set; override with --expert-raw
DAGGER_PHASE="pick"                             # which phase of the rollouts to relabel

while [[ $# -gt 0 ]]; do
    case $1 in
        --output-dir)                       OUTPUT_DIR="$2";               shift 2 ;;
        --collect-only)                     PHASE="collect";               shift   ;;
        --convert-only)                     PHASE="convert";               shift   ;;
        --finetune-only)                    PHASE="finetune";              shift   ;;
        --train-head-only)                  PHASE="train-head";            shift   ;;
        --collect-and-train-head)           PHASE="collect-train-head";    shift   ;;
        --dagger-rollouts)                  DAGGER_ROLLOUTS="$2";          shift 2 ;;
        --expert-raw)                       EXPERT_RAW="$2";               shift 2 ;;
        --dagger-phase)                     DAGGER_PHASE="$2";             shift 2 ;;
        --preset)
            # Shorthand for a named collection/training configuration so simple
            # runs don't have to spell out the full flag set.
            case "$2" in
                phased-cube)  COLLECT_ARGS+=(--phased --forced-obj-sequence cube) ;;
                *) echo "ERROR: unknown --preset '$2' (known: phased-cube)"; exit 1 ;;
            esac
            shift 2 ;;
        --pretrained-path|--finetune-mode)  TRAIN_ARGS+=("$1" "$2");      shift 2 ;;
        --task-modality)                    TRAIN_ARGS+=("$1" "$2");      shift 2 ;;
        --n-finetune-steps|--batch-size)    TRAIN_ARGS+=("$1" "$2");      shift 2 ;;
        # --overfit drives the whole chain: deterministic episodes in collect.py
        # AND no-augmentation/zero-wd/keep-dwell-frames in pipeline.py.
        --overfit)                          TRAIN_ARGS+=("$1"); COLLECT_ARGS+=("$1"); shift ;;
        --head-raw-dir)                     HEAD_RAW_DIR="$2";             shift 2 ;;
        --head-model-path)                  HEAD_ARGS+=("--model-path" "$2"); shift 2 ;;
        --head-batch-size)                  HEAD_ARGS+=("--batch-size" "$2"); shift 2 ;;
        --lam)                              HEAD_ARGS+=("--lam" "$2");     shift 2 ;;
        *)                                  COLLECT_ARGS+=("$1");          shift   ;;
    esac
done

if [ -z "$OUTPUT_DIR" ]; then
    echo "ERROR: --output-dir is required."
    echo "Usage: bash launch/pipeline.sh --output-dir data/exp_01 [options]"
    exit 1
fi

RAW_DIR="$OUTPUT_DIR/raw"

# ── Logging ───────────────────────────────────────────────────────────────────
case "$PHASE" in
    all)                 _LOG_SUBDIR="full"              ;;
    collect)             _LOG_SUBDIR="collect"           ;;
    convert)             _LOG_SUBDIR="collect"           ;;
    finetune)            _LOG_SUBDIR="finetune"          ;;
    train-head)          _LOG_SUBDIR="train_head"        ;;
    collect-train-head)  _LOG_SUBDIR="collect_train_head" ;;
    *)                   _LOG_SUBDIR="full"              ;;
esac
_LOG_DIR="$PROJECT_ROOT/debug/logs/octo_infer/$_LOG_SUBDIR"
mkdir -p "$_LOG_DIR"
_LOG_FILE="$_LOG_DIR/$(date +"%Y%m%d_%H%M%S").log"
exec > >(tee -a "$_LOG_FILE") 2>&1
echo "[LOG] Writing to $_LOG_FILE"

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

# ── DAgger prep (replaces Collect when --dagger-rollouts is given) ────────────
# Relabel on-policy rollouts with the sim oracle, then aggregate the corrective
# episodes with the expert set into RAW_DIR — so the normal Convert+Finetune
# below trains on D_expert ∪ D_dagger.  No Isaac/collect needed.
if [[ -n "$DAGGER_ROLLOUTS" ]]; then
    TRAIN_PY=$(detect_train_python)
    [ -z "$TRAIN_PY" ] && { echo "ERROR: No Python found for DAgger relabel."; exit 1; }
    [ -d "$DAGGER_ROLLOUTS" ] || { echo "ERROR: --dagger-rollouts dir not found: $DAGGER_ROLLOUTS"; exit 1; }
    [ -d "$EXPERT_RAW" ]      || { echo "ERROR: --expert-raw dir not found: $EXPERT_RAW"; exit 1; }

    echo "=== DAgger: relabel + aggregate ==="
    _CORR="$OUTPUT_DIR/corrective"
    "$TRAIN_PY" "$PROJECT_ROOT/debug/dagger_relabel.py" \
        --rollout-dir "$DAGGER_ROLLOUTS" --out-dir "$_CORR" --phase "$DAGGER_PHASE" || exit 1

    _NCORR=$(ls "$_CORR"/*.npz 2>/dev/null | wc -l)
    [ "$_NCORR" -eq 0 ] && { echo "ERROR: relabeler produced no corrective episodes."; exit 1; }

    mkdir -p "$RAW_DIR"
    rm -f "$RAW_DIR"/*.npz
    cp "$EXPERT_RAW"/*.npz "$RAW_DIR/"
    cp "$_CORR"/*.npz       "$RAW_DIR/"
    echo "  expert=$(ls "$EXPERT_RAW"/*.npz | wc -l)  corrective=$_NCORR  → aggregated $(ls "$RAW_DIR"/*.npz | wc -l) in $RAW_DIR"
fi

# ── Collect ───────────────────────────────────────────────────────────────────
if [[ ( "$PHASE" == "all" || "$PHASE" == "collect" ) && -z "$DAGGER_ROLLOUTS" ]]; then
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
        --action-stride 12 \
        "${COLLECT_ARGS[@]}"

    # Isaac Sim's simulation_app.close() calls sys.exit(0) internally, so the
    # exit code is unreliable. Check for actual output instead.
    EPISODE_COUNT=$(find "$RAW_DIR" -name "*.npz" -maxdepth 1 2>/dev/null | wc -l)
    if [ "$EPISODE_COUNT" -eq 0 ]; then
        echo "ERROR: Collect phase produced no episodes in $RAW_DIR — check output above."
        exit 1
    fi
    echo "  Episodes collected: $EPISODE_COUNT"

    # Isaac Sim's Kit process and background threads don't always exit immediately
    # when simulation_app.close() is called. Wait for them to fully release the
    # GPU before starting JAX, otherwise JAX initialisation can fail.
    echo "  Waiting for Isaac Sim processes to release GPU..."
    sleep 10
    pkill -f "kit_app" 2>/dev/null || true
    pkill -f "omni.kit" 2>/dev/null || true
    sleep 3
fi

# ── Convert ───────────────────────────────────────────────────────────────────
if [[ "$PHASE" == "all" || "$PHASE" == "convert" ]]; then
    TRAIN_PY=$(detect_train_python)
    if [ -z "$TRAIN_PY" ]; then echo "ERROR: No Python found for training."; exit 1; fi

    echo "=== Convert ==="
    echo "  Python  : $TRAIN_PY"
    echo "  Raw dir : $RAW_DIR"
    echo ""
    "$TRAIN_PY" "$PROJECT_ROOT/src/training/pipeline.py" \
        --raw-dir    "$RAW_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --convert-only \
        "${TRAIN_ARGS[@]}" || exit 1
fi

# ── Finetune ──────────────────────────────────────────────────────────────────
# Run as a separate process from convert so TensorFlow/TFDS state initialised
# during download_and_prepare() does not interfere with JAX GPU allocation.
if [[ "$PHASE" == "all" || "$PHASE" == "finetune" ]]; then
    TRAIN_PY=$(detect_train_python)
    if [ -z "$TRAIN_PY" ]; then echo "ERROR: No Python found for training."; exit 1; fi

    # Expose cuDNN (installed via pip as nvidia-cudnn-cu12) so JAX can use the GPU.
    _CUDNN_LIB=$("$TRAIN_PY" -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))" 2>/dev/null)/lib
    if [ -d "$_CUDNN_LIB" ]; then
        export LD_LIBRARY_PATH="$_CUDNN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    echo "=== Finetune ==="
    echo "  Python  : $TRAIN_PY"
    echo "  Raw dir : $RAW_DIR"
    echo ""
    "$TRAIN_PY" "$PROJECT_ROOT/src/training/pipeline.py" \
        --raw-dir    "$RAW_DIR" \
        --output-dir "$OUTPUT_DIR" \
        --finetune-only \
        "${TRAIN_ARGS[@]}" || exit 1
fi

# ── Train linear head ─────────────────────────────────────────────────────────
if [[ "$PHASE" == "train-head" ]]; then
    TRAIN_PY=$(detect_train_python)
    if [ -z "$TRAIN_PY" ]; then echo "ERROR: No Python found for training."; exit 1; fi

    _CUDNN_LIB=$("$TRAIN_PY" -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))" 2>/dev/null)/lib
    if [ -d "$_CUDNN_LIB" ]; then
        export LD_LIBRARY_PATH="$_CUDNN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    HEAD_NPZ="$OUTPUT_DIR/linear_head.npz"
    _HEAD_RAW="${HEAD_RAW_DIR:-$RAW_DIR}"

    echo "=== Train linear head ==="
    echo "  Python  : $TRAIN_PY"
    echo "  Raw dir : $_HEAD_RAW"
    echo "  Output  : $HEAD_NPZ"
    echo ""
    "$TRAIN_PY" "$PROJECT_ROOT/src/training/train_linear_head.py" \
        --raw-dir "$_HEAD_RAW" \
        --output  "$HEAD_NPZ" \
        "${HEAD_ARGS[@]}" || exit 1

    echo ""
    echo "Linear head saved to $HEAD_NPZ"
    echo "Run inference with:"
    echo "  ./run.sh sim --head-path $HEAD_NPZ --instruction \"...\""
fi

# ── Collect + train linear head ───────────────────────────────────────────────
if [[ "$PHASE" == "collect-train-head" ]]; then
    ISAAC_PY=$(detect_isaac_python)
    if [ -z "$ISAAC_PY" ]; then
        echo "ERROR: No Python with isaacsim found."; exit 1
    fi
    echo "=== Collect ==="
    echo "  Python : $ISAAC_PY"
    echo "  Output : $RAW_DIR"
    echo ""
    "$ISAAC_PY" "$PROJECT_ROOT/src/isaac/collect.py" \
        --output-dir "$RAW_DIR" \
        --action-stride 12 \
        "${COLLECT_ARGS[@]}"

    EPISODE_COUNT=$(find "$RAW_DIR" -name "*.npz" -maxdepth 1 2>/dev/null | wc -l)
    if [ "$EPISODE_COUNT" -eq 0 ]; then
        echo "ERROR: Collect produced no episodes in $RAW_DIR — check output above."
        exit 1
    fi
    echo "  Episodes collected: $EPISODE_COUNT"

    echo "  Waiting for Isaac Sim to release GPU..."
    sleep 10
    pkill -f "kit_app" 2>/dev/null || true
    pkill -f "omni.kit" 2>/dev/null || true
    sleep 3

    TRAIN_PY=$(detect_train_python)
    if [ -z "$TRAIN_PY" ]; then echo "ERROR: No Python found for training."; exit 1; fi

    _CUDNN_LIB=$("$TRAIN_PY" -c "import nvidia.cudnn, os; print(os.path.dirname(nvidia.cudnn.__file__))" 2>/dev/null)/lib
    if [ -d "$_CUDNN_LIB" ]; then
        export LD_LIBRARY_PATH="$_CUDNN_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    fi

    HEAD_NPZ="$OUTPUT_DIR/linear_head.npz"

    echo ""
    echo "=== Train linear head ==="
    echo "  Python  : $TRAIN_PY"
    echo "  Raw dir : $RAW_DIR"
    echo "  Output  : $HEAD_NPZ"
    echo ""
    "$TRAIN_PY" "$PROJECT_ROOT/src/training/train_linear_head.py" \
        --raw-dir "$RAW_DIR" \
        --output  "$HEAD_NPZ" \
        "${HEAD_ARGS[@]}" || exit 1

    echo ""
    echo "Done. Linear head saved to $HEAD_NPZ"
    echo "Run inference with:"
    echo "  ./run.sh sim --head-path $HEAD_NPZ --instruction \"...\""
fi
