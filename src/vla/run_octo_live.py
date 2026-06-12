# run_octo_live.py — Live Octo VLA inference loop.
#
# Captures frames from a camera (MechEye, USB, or ROS2 topic), runs Octo
# inference, and sends 7-DOF EEF delta actions via TCP or ROS2 topic.
# Action format sent: dx;dy;dz;drx;dry;drz;gripper\n
#
# Pipeline: isaac/collect.py → training/pipeline.py (convert + finetune) → this script
#
# Usage:
#   python run_octo_live.py --instruction "pick up the circle"
#   python run_octo_live.py --goal-image /path/to/goal.png
#   python run_octo_live.py --model-path /path/to/checkpoint --dataset-name my_dataset

import argparse
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from mech_eye_camera import MechEyeCamera, ROS2Camera


def _find_latest_checkpoint() -> Path | None:
    """Return the most recently modified local finetune experiment dir, or None."""
    data_root = Path(__file__).parent.parent.parent / "data"
    if not data_root.exists():
        return None
    candidates = []
    for exp_dir in data_root.glob("*/checkpoint/octo_finetune/experiment_*"):
        if exp_dir.is_dir() and any(d.name.isdigit() for d in exp_dir.iterdir() if d.is_dir()):
            candidates.append(exp_dir)
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


class USBCamera:
    """Thin OpenCV wrapper matching the MechEyeCamera interface."""

    def __init__(self, device: int = 0):
        self.device = device
        self._cap   = None

    def connect(self):
        """Open the USB camera and discard warm-up frames."""
        self._cap = cv2.VideoCapture(self.device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open USB camera (device index {self.device}).")
        # Discard the first few frames — most USB cameras output black frames
        # while auto-exposure and white-balance settle.
        for _ in range(10):
            self._cap.read()
        print(f"[CAM] USB camera opened (device {self.device}).")

    def capture_rgb(self) -> np.ndarray:
        """Capture one frame and return it as HxWx3 RGB uint8."""
        ret, frame_bgr = self._cap.read()
        if not ret:
            raise RuntimeError("USB camera read failed.")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        """Release the camera."""
        if self._cap:
            self._cap.release()
            self._cap = None
            print("[CAM] USB camera released.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

from octo_policy import OctoPolicy, LinearHeadPolicy
from tcp_sender  import EEFDeltaSender, ROS2EEFPublisher, ROS2EEFStateSubscriber

# ------------------------------------------------------------------
# Defaults  (override via CLI)
# ------------------------------------------------------------------

CAMERA_IP    = "192.168.137.100"
CAMERA_PORT  = 50005

# SENDER_HOST  = "127.0.0.1"
# SENDER_PORT  = 9001

SENDER_HOST  = "192.168.1.244"
SENDER_PORT  = 9005

#MODEL_PATH   = "hf://rail-berkeley/octo-small-1.5"
MODEL_PATH   = "hf://rail-berkeley/octo-base-1.5"
DATASET_NAME = "bridge_dataset"

STEP_DELAY   = 0.2   # seconds between inference steps (~5 Hz)


def parse_args():
    """Parse CLI arguments for model, task, camera, and sender configuration."""
    ap = argparse.ArgumentParser(description="Octo VLA live test loop.")

    ap.add_argument("--model-path",   type=str, default=None,
                    help="Octo checkpoint path or HF URI (default: latest local finetune, "
                         f"or {MODEL_PATH} if none found)")
    ap.add_argument("--dataset-name", type=str, default=None,
                    help="Dataset key for unnormalisation stats "
                         "(default: ootf_synthetic for local checkpoints, "
                         f"{DATASET_NAME} for pretrained)")
    ap.add_argument("--step",         type=int, default=None,
                    help="Checkpoint step to load (default: latest)")
    ap.add_argument("--window-size",  type=int, default=2,
                    help="Observation history length passed to Octo (default: 2, must match training window_size)")

    task_grp = ap.add_mutually_exclusive_group()
    task_grp.add_argument("--instruction", type=str, default=None,
                          help="Natural-language instruction (language-conditioned mode)")
    task_grp.add_argument("--goal-image",  type=str, default=None,
                          help="Path to a goal image PNG/JPG (goal-conditioned mode)")

    ap.add_argument("--camera-ip",   type=str, default=CAMERA_IP)
    ap.add_argument("--camera-port", type=int, default=CAMERA_PORT)
    ap.add_argument("--host", type=str, default=SENDER_HOST,
                    help=f"Receiver host (default: {SENDER_HOST})")
    ap.add_argument("--port", type=int, default=SENDER_PORT,
                    help=f"Receiver port (default: {SENDER_PORT})")
    ap.add_argument("--steps",      type=int,   default=0,
                    help="Number of steps to run (0 = run until Ctrl+C)")
    ap.add_argument("--step-delay", type=float, default=STEP_DELAY,
                    help=f"Seconds between steps (default: {STEP_DELAY}). Only used "
                         "when frame-synced pacing is unavailable (no /eef_state).")
    ap.add_argument("--sync-frames", type=int, default=12,
                    help="Pace steps by waiting for this many /eef_state messages "
                         "(= sim render frames) after each action — matches the "
                         "12-frame action stride from training even when the sim "
                         "runs slower than real time. 0 = wall-clock pacing "
                         "(silently truncates actions on a slow sim). Requires "
                         "--ros2-camera. (default: 12)")
    ap.add_argument("--usb-camera",  type=int, default=None, metavar="DEVICE",
                    help="Use a USB webcam instead of MechEye (device index, usually 0)")
    ap.add_argument("--ros2-camera", action="store_true",
                    help="Use the Isaac Sim camera via ROS2 topic instead of a physical camera")
    ap.add_argument("--ros2-topic",  type=str, default="/mecheye/color",
                    help="ROS2 Image topic to subscribe to (default: /mecheye/color)")
    ap.add_argument("--ros2-output", action="store_true",
                    help="Publish EEF deltas to /eef_delta ROS2 topic instead of TCP")
    ap.add_argument("--show-camera", action="store_true",
                    help="Open an OpenCV window showing the live camera feed")
    ap.add_argument("--dry-run",    action="store_true",
                    help="Run without a camera (sends random noise images for testing Octo + TCP)")
    ap.add_argument("--action-horizon", type=int, default=4,
                    help="Number of actions per predicted chunk — must match training config (default: 4)")
    ap.add_argument("--ensemble-weight", type=float, default=0.0,
                    help="Exponential decay weight for temporal ensembling. "
                         "0 = uniform average across overlapping predictions (default: 0)")
    ap.add_argument("--no-temporal-ensemble", action="store_true",
                    help="Use only chunk[0] without averaging overlapping predictions. "
                         "Use this for debugging to see raw model output.")
    ap.add_argument("--grip-threshold", type=float, default=0.9,
                    help="Ensemble grip value must exceed this before closing the gripper. "
                         "Higher values prevent premature closes during approach (default: 0.9)")
    ap.add_argument("--grip-open-threshold", type=float, default=0.5,
                    help="Once closed, open only when the grip prediction drops below this "
                         "value — hysteresis against the close threshold (default: 0.5)")
    ap.add_argument("--grip-open-steps", type=int, default=3,
                    help="Consecutive sub-threshold grip predictions required to open. "
                         "Debounces diffusion sampling noise so a single low sample "
                         "cannot drop the object mid-transfer (default: 3)")
    ap.add_argument("--grip-hold-steps", type=int, default=3,
                    help="Once the gripper closes, hold it closed for at least this many "
                         "steps regardless of model output. Gives the surface gripper time "
                         "to latch before the model commands open (default: 10 = 2 s at 5 Hz)")
    ap.add_argument("--head-path", type=str, default=None,
                    help="Path to a trained linear_head.npz — uses LinearHeadPolicy "
                         "instead of the diffusion head (recommended)")

    return ap.parse_args()


def main():
    """Resolve model/camera/sender, load Octo, and run the inference loop."""
    args = parse_args()

    if args.model_path is None:
        if args.head_path:
            # Linear head mode: backbone is baked into the .npz.
            # Skip local checkpoint auto-detection — loading a finetuned checkpoint
            # here would produce different embeddings than the head was trained on.
            args.model_path = MODEL_PATH   # LinearHeadPolicy will override from npz
        else:
            local_ckpt = _find_latest_checkpoint()
            if local_ckpt:
                args.model_path = str(local_ckpt)
                print(f"[INFO] Using finetuned checkpoint: {local_ckpt}")
            else:
                args.model_path = MODEL_PATH
                print(f"[INFO] No local checkpoint found — using pretrained: {MODEL_PATH}")

    if args.dataset_name is None:
        args.dataset_name = (
            "ootf_synthetic" if not args.model_path.startswith("hf://") else DATASET_NAME
        )

    if args.head_path:
        policy = LinearHeadPolicy(
            model_path  = args.model_path,
            head_path   = args.head_path,
            window_size = args.window_size,
        )
    else:
        policy = OctoPolicy(
            model_path   = args.model_path,
            dataset_name = args.dataset_name,
            window_size  = args.window_size,
            step         = args.step,
        )

    if args.goal_image:
        goal_bgr = cv2.imread(args.goal_image)
        if goal_bgr is None:
            sys.exit(f"[ERR] Could not load goal image: {args.goal_image}")
        goal_rgb = cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2RGB)
        policy.set_task_goal_image(goal_rgb)

    elif args.instruction:
        policy.set_task_text(args.instruction)

    else:
        if not sys.stdin.isatty():
            sys.exit(
                "[ERR] No task specified and stdin is not a terminal (running as background process).\n"
                "      Pass --instruction or --goal-image when launching in vla/e2e mode.\n"
                "      Example: bash launch/launch.sh vla --instruction 'pick up the circle'"
            )

        # Interactive prompt — ask at runtime so you can change tasks without
        # restarting the script.
        print("\nTask mode:")
        print("  [l] Language instruction")
        print("  [g] Goal image path")
        choice = input("Choice (l/g): ").strip().lower()

        if choice == "l":
            text = input("Instruction: ").strip()
            policy.set_task_text(text)
        elif choice == "g":
            path = input("Goal image path: ").strip()
            goal_bgr = cv2.imread(path)
            if goal_bgr is None:
                sys.exit(f"[ERR] Could not load goal image: {path}")
            policy.set_task_goal_image(cv2.cvtColor(goal_bgr, cv2.COLOR_BGR2RGB))
        else:
            sys.exit("[ERR] Invalid choice.")

    if args.dry_run:
        print("[INFO] Dry-run mode — using random noise frames (no camera).")
        camera = None
    elif args.ros2_camera:
        camera = ROS2Camera(args.ros2_topic)
        camera.connect()
    elif args.usb_camera is not None:
        camera = USBCamera(args.usb_camera)
        camera.connect()
    else:
        camera = MechEyeCamera(args.camera_ip, args.camera_port)
        camera.connect()

    eef_state_sub = None
    if args.ros2_camera:
        eef_state_sub = ROS2EEFStateSubscriber()
        eef_state_sub.connect()

    sender_ctx = ROS2EEFPublisher() if args.ros2_output else EEFDeltaSender(args.host, args.port)
    with sender_ctx as sender:
        print(f"\n[INFO] Running. Ctrl+C to stop.")
        if args.ros2_output:
            print(f"[INFO] Publishing EEF deltas to /eef_delta  step_delay={args.step_delay}s\n")
        else:
            print(f"[INFO] Sending to {args.host}:{args.port}  step_delay={args.step_delay}s\n")

        if eef_state_sub is not None:
            print("[INFO] Waiting for first EEF state message...")
            while eef_state_sub.get_latest() is None:
                time.sleep(0.05)
            print("[INFO] EEF state received.")

        policy.reset()
        act_history        = deque(maxlen=args.action_horizon)
        step               = 0
        grip_hold_remaining = 0
        prev_grip           = 0.0
        low_grip_count      = 0

        if args.show_camera:
            cv2.namedWindow("Octo — camera feed", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Octo — camera feed", 640, 480)

        try:
            while True:
                t0 = time.time()

                if camera is not None:
                    frame_rgb = camera.capture_rgb()
                else:
                    frame_rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

                if args.show_camera:
                    display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Octo — camera feed", display)
                    # Pump the Qt event loop long enough to render the frame,
                    # then check for 'q'.  waitKey(1) after slow inference
                    # doesn't give Qt enough time to refresh the window.
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        print("[INFO] 'q' pressed — stopping.")
                        break

                proprio = eef_state_sub.get_latest() if eef_state_sub is not None else None
                chunk = policy.step(frame_rgb, proprio=proprio)   # (action_horizon, 7)

                if args.no_temporal_ensemble:
                    # Use only the first action in the chunk — each action represents
                    # one full action_stride window (0.2 s at stride=12).  Executing
                    # the full chunk at sub-stride intervals causes 4× overshoot.
                    action = chunk[0].copy()
                else:
                    # Temporal ensembling: blend this chunk with the last action_horizon
                    # predictions.  Each overlapping prediction contributes its action for
                    # the current timestep, weighted by exp(-ensemble_weight * age).
                    act_history.append(chunk[: args.action_horizon])
                    num_preds = len(act_history)
                    curr_act_preds = np.stack([
                        pred_actions[i]
                        for i, pred_actions in zip(range(num_preds - 1, -1, -1), act_history)
                    ])
                    weights = np.exp(-args.ensemble_weight * np.arange(num_preds))
                    weights = weights / weights.sum()
                    action = np.sum(weights[:, None] * curr_act_preds, axis=0)
                    # Never ensemble the gripper — future-chunk grip predictions bleed
                    # into the current step 4 steps early, causing the gripper to fire
                    # before the arm reaches the cube.  Use the raw current prediction.
                    action[6] = chunk[0][6]

                # Schmitt-trigger gripper: close above grip_threshold; once
                # closed, open only after grip_open_steps consecutive
                # predictions below grip_open_threshold.  A single noisy
                # diffusion sample must not release the object mid-transfer.
                if prev_grip < 0.5:
                    raw_grip = 1.0 if action[6] > args.grip_threshold else 0.0
                    low_grip_count = 0
                else:
                    if action[6] < args.grip_open_threshold:
                        low_grip_count += 1
                    else:
                        low_grip_count = 0
                    raw_grip = 0.0 if low_grip_count >= args.grip_open_steps else 1.0

                # Hold timer: keep gripper closed for grip_hold_steps after first close.
                if raw_grip > 0.5 and prev_grip < 0.5:
                    grip_hold_remaining = args.grip_hold_steps

                if grip_hold_remaining > 0:
                    action[6] = 1.0
                    grip_hold_remaining -= 1
                else:
                    action[6] = raw_grip

                prev_grip = action[6]
                sender.send(action)

                step += 1
                labels = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
                vals   = "  ".join(f"{l}={v:+.4f}" for l, v in zip(labels, action))
                hold_tag = f"  [hold {grip_hold_remaining}]" if grip_hold_remaining > 0 else ""
                print(f"[STEP {step:04d}]  {vals}{hold_tag}")

                if args.steps > 0 and step >= args.steps:
                    print(f"[INFO] Reached {args.steps} steps. Done.")
                    break

                # Frame-synced pacing: wait until the sim has actually executed
                # the 12 substeps of this action.  Wall-clock sleep undershoots
                # every action when the sim runs below real time.
                if eef_state_sub is not None and args.sync_frames > 0:
                    if not eef_state_sub.wait_frames(args.sync_frames):
                        print("[WARN] /eef_state stalled — falling back to step delay.")
                        time.sleep(args.step_delay)
                else:
                    time.sleep(args.step_delay)

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")

        finally:
            if args.show_camera:
                cv2.destroyAllWindows()
            if camera is not None:
                camera.close()
            if eef_state_sub is not None:
                eef_state_sub.close()


if __name__ == "__main__":
    main()
