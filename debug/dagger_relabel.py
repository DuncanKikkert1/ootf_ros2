#!/usr/bin/env python3
# =============================================================================
# dagger_relabel.py — DAgger oracle relabeler (component 2).
#
# Turns on-policy rollout traces (recorded by success_rate.py --record-dagger)
# into corrective training episodes.  For each policy-visited state it replaces
# the policy's action with the action the SIM ORACLE would take toward the known
# cube — a "go above, then descend straight down" controller in the robot/base
# frame (the same frame as proprio and the action deltas).
#
# Why this fixes the grasp: the policy is accurate on expert states but drifts
# ~100 mm off in closed loop (covariate shift).  These traces are exactly the
# drifted states; labeling them with the recovery action and re-finetuning
# teaches the model to servo back toward the cube — shrinking the XY scatter
# that caps grasp.
#
# Frames validated 2026-06-18: proprio XY converges to cube_rf during a good
# approach, and at grasp link_6_z ≈ cube_rf_z + 0.265 (cup tip 0.215 + cube
# half-height 0.05).  cube_rf is the cube CENTRE in the base frame.
#
# Usage:
#   python debug/dagger_relabel.py \
#       --rollout-dir data/exp_dagger/rollouts --out-dir data/exp_dagger/raw
# =============================================================================

import argparse
import sys
from pathlib import Path

import numpy as np


def parse_args():
    ap = argparse.ArgumentParser(description="DAgger oracle relabeler.")
    ap.add_argument("--rollout-dir", default="data/exp_dagger/rollouts",
                    help="Dir of recorded rollout traces (rollout_*.npz).")
    ap.add_argument("--out-dir",     default="data/exp_dagger/raw",
                    help="Output dir for relabeled training episodes.")
    ap.add_argument("--phase",       default="pick",
                    help="Only relabel steps of this phase (grasp DAgger = pick).")
    ap.add_argument("--instruction", default="pick up the cube from the pallet")
    # oracle geometry (base frame, metres) — see header for the validation
    ap.add_argument("--grasp-offset", type=float, default=0.265,
                    help="link_6 z above the cube CENTRE at grasp (cup 0.215 + half-cube 0.05).")
    ap.add_argument("--hover",        type=float, default=0.10,
                    help="Extra height above the grasp point for the lateral-align phase.")
    ap.add_argument("--stride-cap",   type=float, default=0.12,
                    help="Max per-step Δpos norm (matches the expert stride magnitude).")
    ap.add_argument("--xy-tol",       type=float, default=0.02,
                    help="XY error below which the oracle switches from align→descend and may close.")
    ap.add_argument("--z-tol",        type=float, default=0.03,
                    help="Z gap above grasp height below which the gripper closes.")
    return ap.parse_args()


def oracle_action(proprio: np.ndarray, cube_rf: np.ndarray, a) -> np.ndarray:
    """Corrective expert action (base-frame [Δpos, Δeuler, gripper]) from the
    current EEF pose toward the cube: align above at hover height, then descend
    straight down; close the gripper once seated at the grasp point."""
    pos     = proprio[:3].astype(np.float64)
    cube    = cube_rf.astype(np.float64)
    grasp_z = cube[2] + a.grasp_offset
    hover_z = grasp_z + a.hover
    xy_err  = float(np.linalg.norm(pos[:2] - cube[:2]))

    # Align over the cube at hover height first; only descend once centred.
    # When off-centre this lifts a prematurely-descended cup back to clearance.
    target_z = grasp_z if xy_err < a.xy_tol else hover_z
    target   = np.array([cube[0], cube[1], target_z])

    dpos = target - pos
    n    = float(np.linalg.norm(dpos))
    if n > a.stride_cap:
        dpos *= a.stride_cap / n

    # Straight-down cube pick: maintain orientation (collection d_euler ≈ 0 on
    # the approach).  Rotation correction is a no-op here.
    deuler = np.zeros(3)

    arrived = (xy_err < a.xy_tol) and ((pos[2] - grasp_z) < a.z_tol)
    grip    = 1.0 if arrived else 0.0
    return np.concatenate([dpos, deuler, [grip]]).astype(np.float32)


def main():
    args = parse_args()
    roll_dir = Path(args.rollout_dir)
    out_dir  = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    traces = sorted(roll_dir.glob("rollout_*.npz"))
    if not traces:
        sys.exit(f"[ERR] No rollout traces in {roll_dir}")

    n_ep = n_steps = n_grip = n_skip_noframe = 0
    dpos_norms = []
    for tr in traces:
        d = np.load(tr, allow_pickle=True)
        if "cube_rf" not in d:
            n_skip_noframe += 1
            continue   # pre-frame-fix trace (no base-frame cube) — skip
        phases  = d["phases"]
        keep    = (phases == args.phase)
        if not keep.any():
            continue
        cube_rf  = d["cube_rf"][keep]
        # skip steps whose cube_rf was unavailable when recorded (NaN)
        valid    = ~np.isnan(cube_rf).any(axis=1)
        if not valid.any():
            continue
        images   = d["images"][keep][valid]
        proprios = d["proprios"][keep][valid]
        cube_rf  = cube_rf[valid]

        actions = np.stack([oracle_action(proprios[i], cube_rf[i], args)
                            for i in range(len(proprios))]).astype(np.float32)

        out = out_dir / f"dagger_{tr.stem}.npz"
        np.savez_compressed(
            out,
            images      = images.astype(np.uint8),
            actions     = actions,
            proprios    = proprios.astype(np.float32),
            instruction = args.instruction,
        )
        n_ep    += 1
        n_steps += len(actions)
        n_grip  += int((actions[:, 6] > 0.5).sum())
        dpos_norms.extend(np.linalg.norm(actions[:, :3], axis=1).tolist())

    dn = np.array(dpos_norms) if dpos_norms else np.zeros(1)
    print(f"\n[DAGGER-RELABEL] wrote {n_ep} episodes → {out_dir}")
    print(f"  steps={n_steps}  gripper-close steps={n_grip} ({100*n_grip/max(1,n_steps):.0f}%)")
    print(f"  Δpos norm: mean={dn.mean()*1000:.0f}mm  p50={np.median(dn)*1000:.0f}mm  "
          f"max={dn.max()*1000:.0f}mm  (cap={args.stride_cap*1000:.0f}mm)")
    if n_skip_noframe:
        print(f"  skipped {n_skip_noframe} pre-frame-fix traces (no cube_rf)")


if __name__ == "__main__":
    main()
