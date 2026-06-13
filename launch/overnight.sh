#!/bin/bash
# =============================================================================
# overnight.sh — phase-conditioned, maximally-varied dataset + dual training.
#
# Collects ONE varied dataset and trains TWO conditioning modalities on it:
#
#   data:    exp_15  — phased (pick/place/home split into separate trajectories),
#                      all three objects, full pick/place ranges, domain
#                      randomization on, random yaw (symmetry-folded target).
#   exp_15   text_conditioned   — per-phase language (proven path, runs first)
#   exp_16   multimodal         — per-phase language + goal image, shares exp_15 TFDS
#
# Phase = trajectory means Octo's goal relabeling stays within a phase, giving
# viewpoint-coherent subgoals from the wrist camera.  At inference, run with
# --phased: the gripper transitions switch pick → place → home.
#
# Launch:  nohup ./run.sh overnight > debug/logs/overnight.log 2>&1 &
# =============================================================================

cd "$(dirname "$0")/.." || exit 1

DATA=data/exp_15
HOLDOUT_DIR=data/exp_15_holdout
HOLDOUT=$HOLDOUT_DIR/raw
STEPS=50000

# Max-variety collection: all objects, default (full) ranges, domain rand on,
# random yaw (folded per object symmetry in task.py), phased segmentation.
COLLECT_ARGS=(--phased --forced-obj-sequence cube cylinder pyramid)

echo "=== overnight.sh started: $(date) ==="

# ── 1. Collect + convert the varied phased training set ──────────────────────
if [ "$(ls "$DATA"/raw/*.npz 2>/dev/null | wc -l)" -lt 100 ]; then
    echo "--- collect exp_15 (phased, varied)  ($(date)) ---"
    ./run.sh pipeline --output-dir "$DATA" --collect-only --n-episodes 200 "${COLLECT_ARGS[@]}"
fi
./run.sh pipeline --output-dir "$DATA" --convert-only

# ── 2. Phased holdout for open-loop eval ─────────────────────────────────────
if [ "$(ls "$HOLDOUT"/*.npz 2>/dev/null | wc -l)" -lt 50 ]; then
    echo "--- collect exp_15 holdout (phased, varied)  ($(date)) ---"
    ./run.sh pipeline --output-dir "$HOLDOUT_DIR" --collect-only \
        --n-episodes 50 --seed 777 "${COLLECT_ARGS[@]}"
fi

eval_ckpt() {
    local name=$1
    local ckpt
    ckpt=$(ls -d "data/$name"/checkpoint/octo_finetune/experiment_* 2>/dev/null | tail -1)
    if [ -n "$ckpt" ]; then
        ./run.sh debug policy --checkpoint "$ckpt" --raw-dir "$HOLDOUT" --n-episodes 50
    else
        echo "WARN: no $name checkpoint — skipping eval"
    fi
}

# ── 3. Train text_conditioned (per-phase language) — runs first ──────────────
echo "--- train exp_15 : text_conditioned  ($(date)) ---"
./run.sh pipeline --output-dir "$DATA" --finetune-only \
    --task-modality text_conditioned --finetune-mode full --n-finetune-steps "$STEPS"
eval_ckpt exp_15

# ── 4. Train multimodal (per-phase language + goal image), shares exp_15 TFDS ─
echo "--- train exp_16 : multimodal  ($(date)) ---"
mkdir -p data/exp_16
ln -sfn ../exp_15/tfds data/exp_16/tfds
./run.sh pipeline --output-dir data/exp_16 --finetune-only \
    --task-modality multimodal --finetune-mode full --n-finetune-steps "$STEPS"
eval_ckpt exp_16

echo "=== overnight.sh done: $(date) ==="
