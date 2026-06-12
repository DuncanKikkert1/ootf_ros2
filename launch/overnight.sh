#!/bin/bash
# =============================================================================
# overnight.sh — unattended train + holdout-eval chain.
#
# Two sequential experiments, each: 200-episode collection → TFDS → 50k-step
# full finetune → fresh-seed 10-episode holdout collection → open-loop
# debug_policy report (checkpoint vs holdout).
#
#   Run 1  exp_11  fixed yaw 0.0          — safe baseline
#   Run 2  exp_12  random yaw, folded EEF — tests symmetry-aware yaw folding
#
# Holdout reports land in debug/policy_diagnosis/ named after data+checkpoint.
# Launch:  nohup bash launch/overnight.sh > debug/logs/overnight.log 2>&1 &
# =============================================================================

cd "$(dirname "$0")/.." || exit 1

echo "=== overnight.sh started: $(date) ==="

# ── Run 1: exp_11 — fixed yaw (safe baseline) ────────────────────────────────
./run.sh pipeline --output-dir data/exp_11 --n-episodes 200 \
    --forced-obj-sequence cube --fixed-yaw 0.0 \
    --finetune-mode full --n-finetune-steps 50000

./run.sh pipeline --output-dir data/exp_11_holdout --collect-only \
    --n-episodes 10 --forced-obj-sequence cube --fixed-yaw 0.0 --seed 999

CKPT_11=$(ls -d data/exp_11/checkpoint/octo_finetune/experiment_* 2>/dev/null | tail -1)
if [ -n "$CKPT_11" ]; then
    ./run.sh debug policy --checkpoint "$CKPT_11" \
        --raw-dir data/exp_11_holdout/raw --n-episodes 10
else
    echo "WARN: no exp_11 checkpoint found — skipping holdout eval"
fi

# ── Run 2: exp_12 — random yaw with symmetry-folded EEF target ───────────────
./run.sh pipeline --output-dir data/exp_12 --n-episodes 200 \
    --forced-obj-sequence cube \
    --finetune-mode full --n-finetune-steps 50000

./run.sh pipeline --output-dir data/exp_12_holdout --collect-only \
    --n-episodes 10 --forced-obj-sequence cube --seed 998

CKPT_12=$(ls -d data/exp_12/checkpoint/octo_finetune/experiment_* 2>/dev/null | tail -1)
if [ -n "$CKPT_12" ]; then
    ./run.sh debug policy --checkpoint "$CKPT_12" \
        --raw-dir data/exp_12_holdout/raw --n-episodes 10
else
    echo "WARN: no exp_12 checkpoint found — skipping holdout eval"
fi

echo "=== overnight.sh done: $(date) ==="
