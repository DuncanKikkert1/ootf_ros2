#!/bin/bash
# =============================================================================
# overnight.sh — one varied phased collection, four models, two ablations.
#
# Collection is the expensive step, so ONE maximally-varied phased dataset is
# collected (all objects, full ranges, domain rand, folded yaw) and reused:
#
#   exp_15  phased,     text_conditioned   ← phased arm of the ablation
#   exp_17  NON-phased, text_conditioned   ← same frames, segments re-joined
#                                            (clean phased-vs-non-phased control)
#   exp_16  phased,     multimodal         ← language + goal image
#   exp_18  phased,     image_conditioned  ← goal image only
#
# Ablations enabled:
#   exp_15 vs exp_17  → does phase/horizon-shortening help closed loop?
#   exp_15/16/18      → which conditioning modality works on a wrist camera?
#
# exp_16/18 share exp_15's TFDS via symlink; exp_17 has its own (derived) TFDS.
# Text models run first (guaranteed useful); multimodal/image follow.
#
# Launch:  nohup ./run.sh overnight > debug/logs/overnight.log 2>&1 &
# =============================================================================

cd "$(dirname "$0")/.." || exit 1

STEPS=50000
COLLECT_ARGS=(--phased --forced-obj-sequence cube)

# Python with numpy for the (non-Isaac) derivation step.
PYNP=""
for py in $(find ~/.pyenv/versions -name python3 2>/dev/null | sort -r) "$(which python3)"; do
    "$py" -c "import numpy" 2>/dev/null && PYNP="$py" && break
done

echo "=== overnight.sh started: $(date) ==="

# ── 1. Collect the varied phased training set + holdout ──────────────────────
if [ "$(ls data/exp_15/raw/*.npz 2>/dev/null | wc -l)" -lt 100 ]; then
    echo "--- collect exp_15 (phased, varied)  ($(date)) ---"
    ./run.sh pipeline --output-dir data/exp_15 --collect-only --n-episodes 200 "${COLLECT_ARGS[@]}"
fi
if [ "$(ls data/exp_15_holdout/raw/*.npz 2>/dev/null | wc -l)" -lt 30 ]; then
    echo "--- collect exp_15 holdout (phased)  ($(date)) ---"
    ./run.sh pipeline --output-dir data/exp_15_holdout --collect-only \
        --n-episodes 50 --seed 777 "${COLLECT_ARGS[@]}"
fi

# ── 2. Derive the non-phased dataset (same frames, segments re-joined) ────────
echo "--- derive non-phased exp_17 from exp_15  ($(date)) ---"
"$PYNP" debug/derive_nonphased.py --phased-dir data/exp_15/raw         --out-dir data/exp_17/raw
"$PYNP" debug/derive_nonphased.py --phased-dir data/exp_15_holdout/raw --out-dir data/exp_17_holdout/raw

# ── 3. Build TFDS: exp_15 (phased) and exp_17 (non-phased) ────────────────────
./run.sh pipeline --output-dir data/exp_15 --convert-only
./run.sh pipeline --output-dir data/exp_17 --convert-only

eval_ckpt() {  # $1=experiment name  $2=holdout raw dir
    local ckpt
    ckpt=$(ls -d "data/$1"/checkpoint/octo_finetune/experiment_* 2>/dev/null | tail -1)
    if [ -n "$ckpt" ]; then
        ./run.sh debug policy --checkpoint "$ckpt" --raw-dir "$2" --n-episodes 50
    else
        echo "WARN: no $1 checkpoint — skipping eval"
    fi
}

train() {  # $1=name  $2=tfds-source(own|exp_15)  $3=modality
    echo "--- train $1 : $3  ($(date)) ---"
    if [ "$2" = "exp_15" ]; then
        mkdir -p "data/$1"; ln -sfn ../exp_15/tfds "data/$1/tfds"
    fi
    ./run.sh pipeline --output-dir "data/$1" --finetune-only \
        --task-modality "$3" --finetune-mode full --n-finetune-steps "$STEPS"
}

# ── 4. Text models first (guaranteed useful), then multimodal, then image ────
train exp_15 own    text_conditioned   ; eval_ckpt exp_15 data/exp_15_holdout/raw
train exp_17 own    text_conditioned   ; eval_ckpt exp_17 data/exp_17_holdout/raw
train exp_16 exp_15 multimodal         ; eval_ckpt exp_16 data/exp_15_holdout/raw
train exp_18 exp_15 image_conditioned  # text-only debug_policy can't eval goal-image

echo "=== overnight.sh done: $(date) ==="
