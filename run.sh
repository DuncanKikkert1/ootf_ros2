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
        echo "Usage: ./run.sh <mode> [args]"
        echo ""
        echo "Data collection and training:"
        echo "  pipeline --output-dir <dir> [--n-episodes N]        Collect + convert + finetune"
        echo "  pipeline --output-dir <dir> --collect-only          Collect episodes only"
        echo "  pipeline --output-dir <dir> --convert-only          Convert .npz to TFDS only"
        echo "  pipeline --output-dir <dir> --finetune-only         Finetune Octo (diffusion head)"
        echo "  pipeline --output-dir <dir> --train-head-only       Train linear regression head"
        echo "  pipeline --output-dir <dir> --collect-and-train-head  Collect episodes + train linear head"
        echo "  overnight                                            Unattended chain: train exp_11 + exp_12, each with holdout eval"
        echo "           Launch: nohup ./run.sh overnight > debug/logs/overnight.log 2>&1 &"
        echo "           [--head-model-path hf://...]                 Backbone for linear head"
        echo "           [--head-batch-size N]                        Forward-pass batch size (default 32)"
        echo "           [--lam F]                                     Ridge regularisation (default 1e-4)"
        echo ""
        echo "Inference:"
        echo "  sim  [--instruction \"...\"]                           Isaac Sim + Octo VLA"
        echo "  sim  --head-path <dir>/linear_head.npz [--instruction \"...\"]"
        echo "                                                        Isaac Sim + linear head (recommended)"
        echo "  real [--instruction \"...\"]                           Real robot + Octo VLA"
        echo "  real --head-path <dir>/linear_head.npz [--instruction \"...\"]"
        echo "                                                        Real robot + linear head"
        echo ""
        echo "Debug:"
        echo "  debug sim                                            Isaac Sim node only"
        echo "  debug bridge                                         TCP → ROS2 bridge only"
        echo "  debug joint                                          Joint service client only"
        echo "  debug gripper-test                                   Test surface gripper (sim must be running)"
        echo "  debug collect [--seq seq1-seq4] [--seed N]          Dry-run episode collector"
        echo "  debug policy [--n-episodes N] [--pretrained]        Diagnose policy vs ground truth (image → action)"
        echo "  debug replay [--npz <episode.npz>]                  Replay GT actions through the sim (action → motion)"
        echo "  debug success [--attempts N] [--phased]             Closed-loop success rate over N resets"
        exit 0
        ;;
    *)
        echo "ERROR: Unknown mode '$MODE'. Run './run.sh help' for usage."
        exit 1
        ;;
esac
