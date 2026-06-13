#!/usr/bin/env python3
# =============================================================================
# derive_nonphased.py — reconstruct whole-episode trajectories from a phased
#                       raw dir, so a non-phased model can be trained on the
#                       SAME underlying frames as a phased one (clean ablation).
#
# Phased collection writes episode_<idx>_<seg>_<phase>.npz.  This concatenates
# the segments of each episode back into one episode_<idx>.npz with a single
# full-task instruction.  No Isaac needed — pure numpy.
#
# Usage:
#   python debug/derive_nonphased.py --phased-dir data/exp_15/raw \
#                                    --out-dir   data/exp_17/raw
# =============================================================================

import argparse
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

OBJECTS      = ("cube", "cylinder", "pyramid")
FULL_TASK    = "pick up the {obj} from the pallet and place it on the conveyor"


def _episode_index(p: Path) -> str:
    """episode_000007_01_place.npz -> '000007'."""
    return p.stem.split("_")[1]


def _detect_object(instructions) -> str:
    """Return the object name mentioned in any segment instruction (default cube)."""
    blob = " ".join(str(i) for i in instructions).lower()
    for obj in OBJECTS:
        if obj in blob:
            return obj
    return "cube"


def main():
    ap = argparse.ArgumentParser(description="Concatenate phased segments into whole episodes.")
    ap.add_argument("--phased-dir", required=True, help="Phased raw dir (episode_*_*_*.npz)")
    ap.add_argument("--out-dir",    required=True, help="Output raw dir for whole-episode npz")
    args = ap.parse_args()

    src = Path(args.phased_dir)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    groups = defaultdict(list)
    for p in sorted(src.glob("episode_*_*.npz")):
        if len(p.stem.split("_")) >= 4:        # episode_<idx>_<seg>_<phase>
            groups[_episode_index(p)].append(p)

    if not groups:
        raise SystemExit(f"[ERR] No phased segment files in {src}")

    written = 0
    for idx, files in sorted(groups.items()):
        files = sorted(files)                  # segment order: 00,01,02 = pick,place,home
        imgs, acts, props, instrs = [], [], [], []
        for f in files:
            d = np.load(f, allow_pickle=True)
            imgs.append(d["images"]); acts.append(d["actions"]); props.append(d["proprios"])
            instrs.append(str(d["instruction"]))
        obj   = _detect_object(instrs)
        np.savez_compressed(
            out / f"episode_{idx}.npz",
            images      = np.concatenate(imgs,  axis=0),
            actions     = np.concatenate(acts,  axis=0),
            proprios    = np.concatenate(props, axis=0),
            instruction = FULL_TASK.replace("{obj}", obj),
            rng_state   = np.array([]),
        )
        written += 1

    print(f"[DERIVE] {written} whole-episode trajectories → {out}")


if __name__ == "__main__":
    main()
