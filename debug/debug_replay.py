#!/usr/bin/env python3
# =============================================================================
# debug_replay.py — Replay ground-truth actions from a recorded .npz episode
#                   through the live sim, bypassing the policy entirely.
#
# debug_policy.py tests image → action (does the model predict the recorded
# actions?).  This tool tests action → motion: it publishes the recorded
# actions to /eef_delta one by one and compares the achieved EEF pose from
# /eef_state against the recorded proprio trajectory.  If GT replay already
# drifts, no policy can succeed — the problem is in the controller chain
# (IK, substep interpolation, euler conventions, clamps), not the model.
#
# Requires a running sim:  ./run.sh debug sim
#
# Usage:
#   ./run.sh debug replay                                  # latest raw episode
#   ./run.sh debug replay --npz data/exp/raw/episode_000000.npz
#   ./run.sh debug replay --step-delay 0.25 --no-reset
# =============================================================================

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT       = Path(__file__).parent.parent
_DATA_ROOT = ROOT / "data"

HOME_POSITION   = [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]   # must match sim_node.py
ACTION_SUBSTEPS = 12   # physics frames per action — must match sim_node.py


def _latest_episode() -> Path | None:
    """Return the first episode of the most recently modified data/*/raw dir."""
    raw_dirs = [d for d in _DATA_ROOT.glob("*/raw") if d.is_dir()]
    if not raw_dirs:
        return None
    raw = max(raw_dirs, key=lambda p: p.stat().st_mtime)
    eps = sorted(raw.glob("*.npz"))
    return eps[0] if eps else None


def _ang_diff(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Wrap-aware per-component angle difference a - b in [-pi, pi)."""
    return (a - b + np.pi) % (2 * np.pi) - np.pi


def parse_args():
    ap = argparse.ArgumentParser(
        description="Replay recorded GT actions through the sim and measure tracking error."
    )
    ap.add_argument("--npz", type=str, default=None,
                    help="Episode .npz to replay (default: first episode of latest data/*/raw)")
    ap.add_argument("--frames-per-step", type=int, default=ACTION_SUBSTEPS,
                    help="Sim frames to wait between actions, counted via /eef_state "
                         "messages (default: 12 = one action stride). Frame counting "
                         "keeps pacing correct even when the sim runs slower than "
                         "real time — wall-clock pacing silently truncates every "
                         "action when it does.")
    ap.add_argument("--step-delay", type=float, default=None,
                    help="Use wall-clock pacing with this many seconds between actions "
                         "instead of frame counting (legacy behaviour; not recommended)")
    ap.add_argument("--no-reset", action="store_true",
                    help="Skip homing the arm and resetting the scene before replay")
    return ap.parse_args()


def main():
    args = parse_args()

    npz_path = Path(args.npz) if args.npz else _latest_episode()
    if npz_path is None or not npz_path.exists():
        sys.exit(f"[ERR] No episode found ({npz_path}). Pass --npz.")

    data     = np.load(npz_path, allow_pickle=True)
    actions  = data["actions"]                      # (T, 7)
    if "proprios" not in data.files:
        sys.exit("[ERR] Episode has no proprios — re-collect with current collect.py.")
    proprios = data["proprios"]                     # (T, 7) [x y z r p y grip]
    print(f"[REPLAY] {npz_path}  ({len(actions)} steps)")

    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Empty, Float64MultiArray

    rclpy.init()
    node       = rclpy.create_node("gt_replay")
    delta_pub  = node.create_publisher(Float64MultiArray, "/eef_delta",    10)
    joint_pub  = node.create_publisher(JointState,        "/joint_command", 10)
    reset_pub  = node.create_publisher(Empty,             "/reset_scene",    1)
    # /eef_state is published once per sim render frame, so counting messages
    # counts sim frames — the only pacing that stays correct when the sim runs
    # slower than real time.
    eef = {"v": None, "frames": 0}

    def _eef_cb(m):
        eef["v"]       = np.array(m.data[:7])
        eef["frames"] += 1

    node.create_subscription(Float64MultiArray, "/eef_state", _eef_cb, 10)

    def spin(seconds: float):
        end = time.time() + seconds
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.02)

    def wait_frames(n: int, timeout: float = 5.0):
        """Block until n more /eef_state messages (= sim frames) arrive."""
        target = eef["frames"] + n
        end    = time.time() + timeout
        while eef["frames"] < target:
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.time() > end:
                print(f"[REPLAY] WARNING: only {eef['frames'] - target + n}/{n} "
                      f"sim frames within {timeout}s — is the sim paused?")
                break

    try:
        print("[REPLAY] Waiting for /eef_state from sim_node …")
        while eef["v"] is None:
            rclpy.spin_once(node, timeout_sec=0.1)

        if not args.no_reset:
            print("[REPLAY] Homing arm + resetting scene …")
            jmsg = JointState()
            jmsg.position = list(HOME_POSITION)
            joint_pub.publish(jmsg)
            spin(2.0)
            reset_pub.publish(Empty())
            spin(1.0)

        start_err = np.linalg.norm(eef["v"][:3] - proprios[0][:3])
        print(f"[REPLAY] Start pose error vs proprios[0]: {start_err*1000:.1f} mm")
        if start_err > 0.05:
            print("[REPLAY] WARNING: arm is >5 cm from the episode start pose — "
                  "tracking errors below will include this offset.")
        pacing = (f"wall-clock {args.step_delay}s" if args.step_delay
                  else f"{args.frames_per_step} sim frames")
        print(f"[REPLAY] Pacing: {pacing} per action")

        pos_errs, rot_errs = [], []
        first_bad = None
        for k in range(len(actions) - 1):
            msg = Float64MultiArray()
            msg.data = [float(v) for v in actions[k]]
            delta_pub.publish(msg)
            if args.step_delay:
                spin(args.step_delay)
            else:
                wait_frames(args.frames_per_step)

            achieved = eef["v"]
            expected = proprios[k + 1]      # action[k] moves proprios[k] → proprios[k+1]
            p_err = achieved[:3] - expected[:3]
            r_err = _ang_diff(achieved[3:6], expected[3:6])
            pos_errs.append(p_err)
            rot_errs.append(r_err)
            p_norm = np.linalg.norm(p_err)
            if first_bad is None and p_norm > 0.02:
                first_bad = k
            print(f"[STEP {k:03d}]  pos_err={p_norm*1000:6.1f} mm  "
                  f"(dx={p_err[0]*1000:+6.1f} dy={p_err[1]*1000:+6.1f} dz={p_err[2]*1000:+6.1f})  "
                  f"rot_err={np.abs(r_err).max():.3f} rad  grip={actions[k][6]:.0f}")

        pos_errs = np.array(pos_errs)
        rot_errs = np.array(rot_errs)
        norms    = np.linalg.norm(pos_errs, axis=1)
        print("\n── GT replay summary ─────────────────────────────────────────")
        print(f"Steps           : {len(pos_errs)}")
        print(f"Pos error mean  : {norms.mean()*1000:.1f} mm   max: {norms.max()*1000:.1f} mm")
        print(f"Per-axis |mean| : x={np.abs(pos_errs[:,0]).mean()*1000:.1f}  "
              f"y={np.abs(pos_errs[:,1]).mean()*1000:.1f}  "
              f"z={np.abs(pos_errs[:,2]).mean()*1000:.1f} mm")
        print(f"Rot error mean  : {np.abs(rot_errs).mean():.4f} rad   "
              f"max: {np.abs(rot_errs).max():.4f} rad")
        # Errors at the gripper transitions are what decide pick/place success.
        # Transient lag during fast transfer segments is expected (the arm
        # trails its command by ~0.1 s, same as during collection); error at
        # the dwells should be near zero.
        grip  = actions[: len(pos_errs) + 1, 6]
        trans = np.where(np.abs(np.diff(grip)) > 0.5)[0]
        for k in trans:
            if k < len(pos_errs):
                kind = "close (pick) " if grip[k + 1] > 0.5 else "open (place)"
                print(f"Gripper {kind} step {k:3d}: pos error "
                      f"{np.linalg.norm(pos_errs[k])*1000:.1f} mm")
        if first_bad is not None:
            print(f"First step with pos error > 2 cm: step {first_bad}")
            print("→ controller chain drifts on ground-truth actions; "
                  "fix this before blaming the model.")
        else:
            print("→ controller tracks GT actions within 2 cm; "
                  "rollout failures are model/data-side, not controller-side.")
    except KeyboardInterrupt:
        print("\n[REPLAY] Stopped by user.")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
