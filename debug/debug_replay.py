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
    ap.add_argument("--grasp-test", action="store_true",
                    help="GROUND-TRUTH grasp test: replay once to find the cup's world "
                         "position at the grasp pose, place the cube exactly there, then "
                         "replay again and report whether the suction actually latches. "
                         "Answers 'can the controller+sim grasp with perfect actions?'")
    ap.add_argument("--pallet-z", type=float, default=0.48,
                    help="Cube-centre rest height on the pallet (grasp-test).")
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
    from std_msgs.msg import Empty, Float64MultiArray, String

    rclpy.init()
    node       = rclpy.create_node("gt_replay")
    delta_pub  = node.create_publisher(Float64MultiArray, "/eef_delta",    10)
    joint_pub  = node.create_publisher(JointState,        "/joint_command", 10)
    reset_pub  = node.create_publisher(Empty,             "/reset_scene",    1)
    setcube_pub = node.create_publisher(Float64MultiArray, "/set_cube_pose", 1)
    # /eef_state is published once per sim render frame, so counting messages
    # counts sim frames — the only pacing that stays correct when the sim runs
    # slower than real time.
    eef = {"v": None, "frames": 0}
    align = {"cup": None}                 # cup centroid world XYZ from /grasp_align
    cube  = {"v": None}                   # cube world XYZ from /cube_pose
    latched = {"v": False}                # pick_cube in /gripper_status

    def _eef_cb(m):
        eef["v"]       = np.array(m.data[:7])
        eef["frames"] += 1

    node.create_subscription(Float64MultiArray, "/eef_state", _eef_cb, 10)
    node.create_subscription(Float64MultiArray, "/grasp_align",
                             lambda m: align.update(cup=(np.array(m.data[3:6]) if len(m.data) >= 6 else None)), 10)
    node.create_subscription(Float64MultiArray, "/cube_pose",
                             lambda m: cube.update(v=np.array(m.data[:3])), 10)
    node.create_subscription(String, "/gripper_status",
                             lambda m: latched.update(v=("pick_cube" in m.data)), 10)

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

        if args.grasp_test:
            def home(reset_cube):
                jm = JointState(); jm.position = list(HOME_POSITION)
                joint_pub.publish(jm); spin(2.5)
                if reset_cube:
                    reset_pub.publish(Empty()); spin(1.5)

            def replay():
                for k in range(len(actions)):
                    m = Float64MultiArray(); m.data = [float(v) for v in actions[k]]
                    delta_pub.publish(m); wait_frames(args.frames_per_step)

            # PASS 1: reach the grasp pose, read where the suction cup ends up.
            print("[GRASP-TEST] Pass 1: replaying to find the grasp-pose cup position …")
            home(reset_cube=True)
            replay(); spin(0.5)
            cup = align["cup"]
            if cup is None:
                sys.exit("[ERR] No /grasp_align received — restart the sim (it must "
                         "publish the cup XYZ).")
            target = [float(cup[0]), float(cup[1]), float(args.pallet_z)]
            print(f"[GRASP-TEST] Cup ended at world {np.round(cup,3)} → "
                  f"placing cube at {np.round(target,3)}")

            # Place the cube exactly under the grasp pose, then re-home WITHOUT a
            # scene reset (so the cube stays put) and replay again.
            setcube_pub.publish(Float64MultiArray(data=target)); spin(1.0)
            print("[GRASP-TEST] Pass 2: re-homing (cube stays) + replaying the same actions …")
            home(reset_cube=False)
            cube_before = cube["v"].copy() if cube["v"] is not None else np.array(target)
            replay()
            for _ in range(20):     # hold closed so the suction can settle + latch
                delta_pub.publish(Float64MultiArray(data=[0, 0, 0, 0, 0, 0, 1.0]))
                wait_frames(args.frames_per_step)
            spin(0.5)

            cube_after = cube["v"]
            lift = float(cube_after[2] - cube_before[2]) if cube_after is not None else 0.0
            cf   = align["cup"]
            dxy  = (float(np.linalg.norm(cf[:2] - cube_after[:2]))
                    if cf is not None and cube_after is not None else float("nan"))
            print("\n── GROUND-TRUTH GRASP TEST ─────────────────────────────────")
            print(f"cube placed at        : {np.round(target,3)}")
            print(f"cube after grasp+hold : "
                  f"{np.round(cube_after,3) if cube_after is not None else None}")
            print(f"cup↔cube XY at end    : {dxy*1000:.0f} mm")
            print(f"cube lift during hold : {lift*1000:+.0f} mm")
            print(f"suction latched (pick_cube held): {latched['v']}")
            if latched["v"]:
                print("→ GRASP SUCCEEDED with ground-truth actions: the controller + sim "
                      "CAN grasp. So rollout failures are MODEL-side (it outputs wrong "
                      "actions), not the controller/sim.")
            else:
                print("→ GRASP FAILED even with perfect actions + the cube placed exactly "
                      "under the grasp pose: the controller/sim/gripper itself can't "
                      "execute the grasp — that must be fixed before any model can work.")
            return

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
