#!/usr/bin/env python3
# debug_axis_check.py — Verify Octo EEF delta axis alignment against the robot.
#
# Sends fixed single-axis pulses (+/-) so you can visually confirm that each
# axis moves the EEF in the expected direction.
#
# Expected bridge_dataset (Octo) conventions:
#   dx > 0  → EEF moves forward   dy > 0 → EEF moves left
#   dz > 0  → EEF moves up        drx > 0 → rolls CCW from +x
#   dry > 0 → pitches up          drz > 0 → yaws left
#
# Usage:
#   bash launch/launch.sh virtual          # start sim (terminal 1)
#   python debug/debug_axis_check.py --ros2-output   # run check (terminal 2)
#   python debug/debug_axis_check.py --host 192.168.1.244 --port 9005

import argparse
import sys
import time


TRANSLATIONAL_DELTA = 0.005   # metres per step  (~0.5 cm)
ROTATIONAL_DELTA    = 0.03    # radians per step  (~1.7 deg)
STEPS_PER_PULSE     = 10      # steps in each +/- pulse
STEP_DELAY          = 0.15    # seconds between steps


AXES = [
    ("dx",  [TRANSLATIONAL_DELTA, 0, 0, 0, 0, 0, 0.5],
            [-TRANSLATIONAL_DELTA, 0, 0, 0, 0, 0, 0.5],
            "EEF should move forward (+) then backward (-)"),
    ("dy",  [0, TRANSLATIONAL_DELTA, 0, 0, 0, 0, 0.5],
            [0, -TRANSLATIONAL_DELTA, 0, 0, 0, 0, 0.5],
            "EEF should move left (+) then right (-)"),
    ("dz",  [0, 0, TRANSLATIONAL_DELTA, 0, 0, 0, 0.5],
            [0, 0, -TRANSLATIONAL_DELTA, 0, 0, 0, 0.5],
            "EEF should move up (+) then down (-)"),
    ("drx", [0, 0, 0, ROTATIONAL_DELTA, 0, 0, 0.5],
            [0, 0, 0, -ROTATIONAL_DELTA, 0, 0, 0.5],
            "EEF should roll CCW (+) then CW (-) viewed from +x"),
    ("dry", [0, 0, 0, 0, ROTATIONAL_DELTA, 0, 0.5],
            [0, 0, 0, 0, -ROTATIONAL_DELTA, 0, 0.5],
            "EEF should pitch up (+) then down (-)"),
    ("drz", [0, 0, 0, 0, 0, ROTATIONAL_DELTA, 0.5],
            [0, 0, 0, 0, 0, -ROTATIONAL_DELTA, 0.5],
            "EEF should yaw left (+) then right (-)"),
]


def parse_args():
    """Parse CLI arguments for sender mode, axis selection, and pulse parameters."""
    ap = argparse.ArgumentParser(
        description="Verify Octo EEF delta axis alignment against the robot."
    )
    ap.add_argument("--ros2-output", action="store_true",
                    help="Publish to /eef_delta ROS2 topic (use with virtual/Isaac Sim mode)")
    ap.add_argument("--host", type=str, default="127.0.0.1",
                    help="TCP receiver host (default: 127.0.0.1)")
    ap.add_argument("--port", type=int, default=9005,
                    help="TCP receiver port (default: 9005)")
    ap.add_argument("--axes", type=str, default=None,
                    help="Comma-separated list of axes to test, e.g. dx,dy,dz (default: all)")
    ap.add_argument("--steps", type=int, default=STEPS_PER_PULSE,
                    help=f"Steps per +/- pulse (default: {STEPS_PER_PULSE})")
    ap.add_argument("--delta-t", type=float, default=TRANSLATIONAL_DELTA,
                    help=f"Translational delta in metres (default: {TRANSLATIONAL_DELTA})")
    ap.add_argument("--delta-r", type=float, default=ROTATIONAL_DELTA,
                    help=f"Rotational delta in radians (default: {ROTATIONAL_DELTA})")
    ap.add_argument("--step-delay", type=float, default=STEP_DELAY,
                    help=f"Seconds between steps (default: {STEP_DELAY})")
    ap.add_argument("--no-pause", action="store_true",
                    help="Skip the Enter-to-continue prompt between axes")
    return ap.parse_args()


def send_pulse(sender, delta, steps, step_delay, label):
    """Send a fixed delta for the given number of steps."""
    vals = "  ".join(f"{v:+.4f}" for v in delta)
    print(f"    [{label}]  {vals}")
    for _ in range(steps):
        sender.send(delta)
        time.sleep(step_delay)


def run(sender, args):
    """Send +/- pulses for each selected axis and prompt for visual confirmation."""
    # Rebuild AXES with CLI-overridden magnitudes
    axes = [
        ("dx",  [args.delta_t, 0, 0, 0, 0, 0, 0.5],
                [-args.delta_t, 0, 0, 0, 0, 0, 0.5],
                "EEF should move forward (+) then backward (-)"),
        ("dy",  [0, args.delta_t, 0, 0, 0, 0, 0.5],
                [0, -args.delta_t, 0, 0, 0, 0, 0.5],
                "EEF should move left (+) then right (-)"),
        ("dz",  [0, 0, args.delta_t, 0, 0, 0, 0.5],
                [0, 0, -args.delta_t, 0, 0, 0, 0.5],
                "EEF should move up (+) then down (-)"),
        ("drx", [0, 0, 0, args.delta_r, 0, 0, 0.5],
                [0, 0, 0, -args.delta_r, 0, 0, 0.5],
                "EEF should roll CCW (+) then CW (-) viewed from +x"),
        ("dry", [0, 0, 0, 0, args.delta_r, 0, 0.5],
                [0, 0, 0, 0, -args.delta_r, 0, 0.5],
                "EEF should pitch up (+) then down (-)"),
        ("drz", [0, 0, 0, 0, 0, args.delta_r, 0.5],
                [0, 0, 0, 0, 0, -args.delta_r, 0.5],
                "EEF should yaw left (+) then right (-)"),
    ]

    filter_axes = set(args.axes.split(",")) if args.axes else None
    selected = [(n, p, n_, d) for n, p, n_, d in axes
                if filter_axes is None or n in filter_axes]

    if not selected:
        sys.exit(f"[ERR] No axes matched '{args.axes}'. Valid names: dx dy dz drx dry drz")

    duration = args.steps * args.step_delay
    print(f"\nAxis alignment check")
    print(f"  {args.steps} steps × {args.step_delay}s = {duration:.1f}s per pulse (+/-)")
    print(f"  translational delta: {args.delta_t*100:.1f} cm/step")
    print(f"  rotational delta:    {args.delta_r*1000:.1f} mrad/step  "
          f"({args.delta_r*180/3.14159:.2f} deg/step)")
    print()

    for name, pos_delta, neg_delta, description in selected:
        print(f"─── {name} ───  {description}")
        send_pulse(sender, pos_delta, args.steps, args.step_delay, f"+{name}")
        send_pulse(sender, neg_delta, args.steps, args.step_delay, f"-{name}")

        if not args.no_pause and selected[-1][0] != name:
            try:
                input("  Aligned? Press Enter for next axis  (Ctrl+C to abort) ...")
            except EOFError:
                pass
        print()

    print("All axes tested.")


def main():
    """Resolve the sender backend and run the axis check."""
    args = parse_args()

    if args.ros2_output:
        sys.path.insert(0, __file__.rsplit("/", 1)[0])
        from tcp_sender import ROS2EEFPublisher
        sender_ctx = ROS2EEFPublisher("/eef_delta")
    else:
        sys.path.insert(0, __file__.rsplit("/", 1)[0])
        from tcp_sender import EEFDeltaSender
        sender_ctx = EEFDeltaSender(args.host, args.port)

    try:
        with sender_ctx as sender:
            run(sender, args)
    except KeyboardInterrupt:
        print("\n[INFO] Aborted by user.")


if __name__ == "__main__":
    main()
