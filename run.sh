#!/bin/bash
# =============================================================================
# run.sh — single entry point for ootf_ros2
#
# Usage:
#   ./run.sh pipeline --output-dir data/exp_01 --n-episodes 200
#   ./run.sh pipeline --output-dir data/exp_01 --collect-only
#   ./run.sh pipeline --output-dir data/exp_01 --convert-only
#   ./run.sh pipeline --output-dir data/exp_01 --finetune-only
#   ./run.sh sim      --instruction "pick up the object"
#   ./run.sh real     --instruction "pick up the object"
#   ./run.sh debug    sim|bridge|joint|policy
# =============================================================================

LAUNCH="$(cd "$(dirname "$0")/launch" && pwd)"
MODE="${1:-}"
shift 2>/dev/null || true

case "$MODE" in
    pipeline) bash "$LAUNCH/pipeline.sh"       "$@" ;;
    sim)      bash "$LAUNCH/sim_inference.sh"  "$@" ;;
    real)     bash "$LAUNCH/real_inference.sh" "$@" ;;
    debug)    bash "$LAUNCH/debug.sh"          "$@" ;;
    ""|-h|--help|help)
        echo "Usage: ./run.sh <mode> [args]"
        echo ""
        echo "  pipeline --output-dir <dir> [--n-episodes N]   Collect + convert + finetune"
        echo "  pipeline --output-dir <dir> --collect-only     Collect episodes only"
        echo "  pipeline --output-dir <dir> --convert-only     Convert .npz to TFDS only"
        echo "  pipeline --output-dir <dir> --finetune-only    Finetune Octo only"
        echo "  sim      [--instruction \"...\"]                  Isaac Sim + Octo VLA"
        echo "  real     [--instruction \"...\"]                  Real robot + Octo VLA"
        echo "  debug    sim|bridge|joint                       Single component"
        echo "  debug    policy [--n-episodes N] [--pretrained] Diagnose finetuned policy"
        exit 0
        ;;
    *)
        echo "ERROR: Unknown mode '$MODE'. Run './run.sh help' for usage."
        exit 1
        ;;
esac
