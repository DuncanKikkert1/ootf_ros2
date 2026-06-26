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
    pipeline)  bash "$LAUNCH/pipeline.sh"       "$@" ;;
    sim)       bash "$LAUNCH/sim_inference.sh"  "$@" ;;
    real)      bash "$LAUNCH/real_inference.sh" "$@" ;;
    debug)     bash "$LAUNCH/debug.sh"          "$@" ;;
    overnight) bash "$LAUNCH/overnight.sh"      "$@" ;;
    ""|-h|--help|help)
        echo "Usage: ./run.sh <mode> [args]     (each underlying script also supports --help)"
        echo ""
        echo "═══ PIPELINE — collect → convert → finetune ═══"
        echo "  pipeline --output-dir <dir> [phase] [flags]"
        echo "    Phase    (default: full collect+convert+finetune)"
        echo "      --collect-only | --convert-only | --finetune-only"
        echo "      --train-head-only | --collect-and-train-head   Linear head instead of finetune"
        echo "    Shorthand"
        echo "      --preset phased-cube     = --phased --forced-obj-sequence cube"
        echo "    Collect  --n-episodes N (200) · --seed N · --phased · --overfit · --gui"
        echo "             --task-type random|sorting · --forced-obj-sequence OBJ..."
        echo "             --domain-rand · --fixed-yaw RAD · --instruction \"...\""
        echo "    Train    --finetune-mode full|head_only|head_mlp_only   (default: full)"
        echo "             --task-modality text_conditioned|image_conditioned|multimodal"
        echo "             --n-finetune-steps N (50000) · --batch-size N (32) · --overfit"
        echo "    Head     --head-model-path hf://... · --head-batch-size N · --lam F (1e-4)"
        echo "    DAgger   --dagger-rollouts <dir>   Relabel on-policy rollouts (from"
        echo "             'debug success --record-dagger') + aggregate with expert set,"
        echo "             then convert+finetune (no collect)."
        echo "             --expert-raw <dir> (default data/exp_truecolor2/raw)"
        echo "             --dagger-phase NAME (pick)"
        echo "             Pass --pretrained-path and OOTF_PROPRIO=1 explicitly (base +"
        echo "             proprio are NOT inherited from the expert set)."
        echo ""
        echo "  overnight   Unattended (no args): 400 phased cube eps → train exp_20 (text)"
        echo "              + exp_21 (multimodal), each holdout-eval'd."
        echo "              nohup ./run.sh overnight > debug/logs/overnight.log 2>&1 &"
        echo ""
        echo "═══ INFERENCE ═══"
        echo "  sim  [flags]     Isaac Sim + Octo VLA"
        echo "  real [flags]     Real robot (MechEye + Doosan) + Octo VLA"
        echo "    --instruction \"...\" | --goal-image PNG        Task spec"
        echo "    --head-path <dir>/linear_head.npz             Linear head (recommended)"
        echo "    --phased · --phase-object cube · --zero-rotation"
        echo "    --temporal-ensemble                           Smooth actions (default off; on for real)"
        echo "    --grip-threshold/-open-threshold/-open-steps/-hold-steps   Gripper trigger"
        echo ""
        echo "═══ DEBUG ═══"
        echo "  debug sim | bridge | joint | gripper-test       Run one component in isolation"
        echo "  debug collect [--seq seq1-seq4] [--seed N]      Dry-run collector (writes nothing)"
        echo "  debug policy  [--n-episodes N] [--pretrained]   Open-loop policy vs GT (image → action)"
        echo "  debug replay  [--npz <episode.npz>]             GT action replay (action → motion)"
        echo "  debug success [--attempts N] [--phased] [...]   Closed-loop success rate over N resets"
        echo ""
        echo "Env toggles: OOTF_PROPRIO=0|1 · OOTF_SORT_PLATFORMS=1 · OOTF_RECORD_DIR=<dir>"
        exit 0
        ;;
    *)
        echo "ERROR: Unknown mode '$MODE'. Run './run.sh help' for usage."
        exit 1
        ;;
esac
