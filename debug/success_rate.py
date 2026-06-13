#!/usr/bin/env python3
# =============================================================================
# success_rate.py — automated closed-loop success-rate harness.
#
# Runs N pick-and-place attempts against a running Isaac Sim, resetting the
# scene between attempts (which re-randomizes the cube position by
# PICKUP_XY_VARIATION in sim_node.py), and tallies grasp / place success from
# the /cube_pose topic — no manual log-scraping.
#
# Reuses OctoPolicy for inference; the per-step gripper/phase processing mirrors
# run_octo_live.py (kept in sync deliberately — see _commit_gripper).
#
# Requires a running sim with a non-zero PICKUP_XY_VARIATION:
#   1. set PICKUP_XY_VARIATION = 0.10 in src/isaac/sim_node.py
#   2. ./run.sh debug sim
#   3. ./run.sh debug success --model-path <ckpt> --attempts 10 --phased
#
# Usage:
#   ./run.sh debug success --attempts 10 --phased --phase-object cube
#   ./run.sh debug success --attempts 10 \
#       --instruction "pick up the cube from the pallet and place it on the conveyor"
# =============================================================================

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src" / "vla"))
sys.path.insert(0, str(ROOT / "src" / "comm"))

from octo_policy import OctoPolicy
from mech_eye_camera import ROS2Camera


def _latest_checkpoint() -> str | None:
    data = ROOT / "data"
    cands = [d for d in data.glob("*/checkpoint/octo_finetune/experiment_*")
             if d.is_dir() and any(c.name.isdigit() for c in d.iterdir() if c.is_dir())]
    return str(max(cands, key=lambda p: p.stat().st_mtime)) if cands else None


PHASE_INSTRUCTIONS = {
    "pick":  "pick up the {obj} from the pallet",
    "place": "place the {obj} on the conveyor belt",
    "home":  "return to the home position",
}


def parse_args():
    ap = argparse.ArgumentParser(description="Closed-loop pick-and-place success rate.")
    ap.add_argument("--model-path",  default=None, help="Checkpoint (default: latest local)")
    ap.add_argument("--dataset-name", default="ootf_synthetic")
    ap.add_argument("--step",        type=int, default=None)
    ap.add_argument("--window-size", type=int, default=2)
    ap.add_argument("--attempts",    type=int, default=10)
    ap.add_argument("--max-steps",   type=int, default=80, help="Step cap per attempt")
    ap.add_argument("--ros2-topic",  default="/mecheye/color")
    ap.add_argument("--sync-frames", type=int, default=12)
    ap.add_argument("--step-delay",  type=float, default=0.2)
    ap.add_argument("--reset-settle", type=float, default=2.0, help="Seconds to let the cube settle after reset")
    # task spec
    ap.add_argument("--phased",       action="store_true")
    ap.add_argument("--phase-object", default="cube")
    ap.add_argument("--instruction",  default=None)
    ap.add_argument("--zero-rotation", action="store_true")
    # gripper (mirror run_octo_live defaults)
    ap.add_argument("--grip-threshold",      type=float, default=0.9)
    ap.add_argument("--grip-open-threshold", type=float, default=0.5)
    ap.add_argument("--grip-open-steps",     type=int,   default=3)
    ap.add_argument("--grip-hold-steps",     type=int,   default=20)
    # success regions (defaults match sim_node pick centre + collect place-y belt)
    ap.add_argument("--lift-z-min",  type=float, default=0.05, help="Cube lift (m) for grasp success")
    ap.add_argument("--belt-y-max",  type=float, default=-0.80, help="Cube y below this = on belt side")
    ap.add_argument("--floor-z-min", type=float, default=0.30, help="Cube z above this = not dropped to floor")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.model_path is None:
        args.model_path = _latest_checkpoint()
        if args.model_path is None:
            sys.exit("[ERR] No local checkpoint found — pass --model-path.")
        print(f"[INFO] Using checkpoint: {args.model_path}")

    import rclpy
    from std_msgs.msg import Float64MultiArray, Empty

    rclpy.init()
    node      = rclpy.create_node("success_rate_harness")
    delta_pub = node.create_publisher(Float64MultiArray, "/eef_delta",   10)
    reset_pub = node.create_publisher(Empty,             "/reset_scene",  1)
    cube      = {"v": None}
    eefst     = {"v": None, "frames": 0}
    node.create_subscription(Float64MultiArray, "/cube_pose",
                             lambda m: cube.update(v=np.array(m.data[:3])), 10)
    def _eef_cb(m):
        eefst["v"] = np.array(m.data[:7]); eefst["frames"] += 1
    node.create_subscription(Float64MultiArray, "/eef_state", _eef_cb, 10)

    def wait_frames(n, timeout=5.0):
        target = eefst["frames"] + n
        end = time.time() + timeout
        while eefst["frames"] < target:
            rclpy.spin_once(node, timeout_sec=0.05)
            if time.time() > end:
                return
    def spin(sec):
        end = time.time() + sec
        while time.time() < end:
            rclpy.spin_once(node, timeout_sec=0.02)

    policy = OctoPolicy(model_path=args.model_path, dataset_name=args.dataset_name,
                        window_size=args.window_size, step=args.step)

    def apply_phase(phase):
        policy.set_task_text(PHASE_INSTRUCTIONS[phase].replace("{obj}", args.phase_object))

    camera = ROS2Camera(args.ros2_topic)
    camera.connect()
    print("[INFO] Waiting for /cube_pose and /eef_state …")
    while cube["v"] is None or eefst["v"] is None:
        rclpy.spin_once(node, timeout_sec=0.1)

    results = []
    try:
        for attempt in range(1, args.attempts + 1):
            print(f"\n========== ATTEMPT {attempt}/{args.attempts} ==========")
            reset_pub.publish(Empty())
            spin(args.reset_settle)
            cube_start = cube["v"].copy()
            print(f"[RESET] cube start: {cube_start.round(3)}")

            policy.reset()
            if args.phased:
                apply_phase("pick")
            elif args.instruction:
                policy.set_task_text(args.instruction)
            else:
                apply_phase("pick")   # fall back to phased pick text

            phase = "pick"; prev_grip = 0.0; grip_hold = 0; low_grip = 0
            cube_lift = 0.0

            for step in range(1, args.max_steps + 1):
                frame = camera.capture_rgb()
                proprio = eefst["v"]
                action = policy.step(frame, proprio=proprio)[0].copy()
                if args.zero_rotation:
                    action[3:6] = 0.0

                # --- gripper commit (mirrors run_octo_live._commit) ---
                if prev_grip < 0.5:
                    raw = 1.0 if action[6] > args.grip_threshold else 0.0
                    low_grip = 0
                else:
                    low_grip = low_grip + 1 if action[6] < args.grip_open_threshold else 0
                    raw = 0.0 if low_grip >= args.grip_open_steps else 1.0
                if raw > 0.5 and prev_grip < 0.5:
                    grip_hold = args.grip_hold_steps
                if grip_hold > 0:
                    action[6] = 1.0; grip_hold -= 1
                else:
                    action[6] = raw

                if args.phased:
                    if phase == "pick" and prev_grip < 0.5 and action[6] > 0.5:
                        phase = "place"; apply_phase(phase); policy.reset()
                    elif phase == "place" and prev_grip > 0.5 and action[6] < 0.5:
                        phase = "home"; apply_phase(phase); policy.reset()

                prev_grip = action[6]
                delta_pub.publish(Float64MultiArray(data=[float(v) for v in action]))

                if cube["v"] is not None:
                    cube_lift = max(cube_lift, cube["v"][2] - cube_start[2])

                if args.sync_frames > 0:
                    wait_frames(args.sync_frames)
                else:
                    spin(args.step_delay)

            cp = cube["v"]
            grasped = cube_lift > args.lift_z_min
            placed  = (cp is not None and cp[1] < args.belt_y_max and cp[2] > args.floor_z_min)
            results.append((grasped, placed))
            print(f"[RESULT] grasped={grasped}  placed={placed}  "
                  f"max_lift={cube_lift*1000:.0f}mm  cube_end={cp.round(3)}")

        g = sum(1 for gr, _ in results if gr)
        p = sum(1 for _, pl in results if pl)
        n = len(results)
        print("\n══════════ SUCCESS RATE ══════════")
        print(f"Attempts : {n}")
        print(f"Grasped  : {g}/{n}  ({100*g/n:.0f}%)")
        print(f"Placed   : {p}/{n}  ({100*p/n:.0f}%)")
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        camera.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
