#!/bin/bash
# =============================================================================
# overnight.sh — denser phased-cube dataset to improve grasp/approach precision.
#
# Closed-loop analysis showed the bottleneck is grasp reliability: the arm
# approaches but the suction just misses, and success needs the EEF within
# ~30 mm in XY.  Two levers attack this: (1) a grasp-dwell at inference (added
# to debug/success_rate.py — no retraining), and (2) MORE DATA so the approach
# lands closer more often.  This run is lever (2): 400 phased cube episodes
# (double the previous 200), trained as text and multimodal.
#
#   exp_20  phased, text_conditioned   (400 eps)
#   exp_21  phased, multimodal         (shares exp_20 TFDS)
#
# Launch:  nohup ./run.sh overnight > debug/logs/overnight.log 2>&1 &
# =============================================================================

cd "$(dirname "$0")/.." || exit 1

STEPS=50000
COLLECT_ARGS=(--phased --forced-obj-sequence cube)

echo "=== overnight.sh started: $(date) ==="

# ── 1. Collect 400 phased cube episodes + holdout ────────────────────────────
if [ "$(ls data/exp_20/raw/*.npz 2>/dev/null | wc -l)" -lt 200 ]; then
    echo "--- collect exp_20 (phased cube, 400 eps)  ($(date)) ---"
    ./run.sh pipeline --output-dir data/exp_20 --collect-only --n-episodes 400 "${COLLECT_ARGS[@]}"
fi
if [ "$(ls data/exp_20_holdout/raw/*.npz 2>/dev/null | wc -l)" -lt 30 ]; then
    echo "--- collect exp_20 holdout  ($(date)) ---"
    ./run.sh pipeline --output-dir data/exp_20_holdout --collect-only \
        --n-episodes 50 --seed 777 "${COLLECT_ARGS[@]}"
fi

# ── 2. Build TFDS ────────────────────────────────────────────────────────────
./run.sh pipeline --output-dir data/exp_20 --convert-only

eval_ckpt() {  # $1=name
    local ckpt
    ckpt=$(ls -d "data/$1"/checkpoint/octo_finetune/experiment_* 2>/dev/null | tail -1)
    if [ -n "$ckpt" ]; then
        ./run.sh debug policy --checkpoint "$ckpt" --raw-dir data/exp_20_holdout/raw --n-episodes 50
    else
        echo "WARN: no $1 checkpoint — skipping eval"
    fi
}

# ── 3. Train text (primary) then multimodal (shares exp_20 TFDS) ─────────────
echo "--- train exp_20 : text_conditioned  ($(date)) ---"
./run.sh pipeline --output-dir data/exp_20 --finetune-only \
    --task-modality text_conditioned --finetune-mode full --n-finetune-steps "$STEPS"
eval_ckpt exp_20

echo "--- train exp_21 : multimodal  ($(date)) ---"
mkdir -p data/exp_21
ln -sfn ../exp_20/tfds data/exp_21/tfds
./run.sh pipeline --output-dir data/exp_21 --finetune-only \
    --task-modality multimodal --finetune-mode full --n-finetune-steps "$STEPS"
eval_ckpt exp_21

echo "=== overnight.sh done: $(date) ==="

