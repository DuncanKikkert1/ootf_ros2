# =============================================================================
# Name        : run_octo_live.py
# Author      : Duncan Kikkert
# Created     : 16/4/2026
# Description : Test loop for the Octo VLA pipeline.
#               Captures frames from the MechEye 3D camera, runs Octo
#               inference, and sends 7-DOF EEF delta actions via TCP.
#
#               Action format sent: dx;dy;dz;drx;dry;drz;gripper\n
#
# Usage examples:
#   # Language-conditioned (prompted at runtime):
#   python run_octo_live.py
#
#   # Language-conditioned (instruction via CLI):
#   python run_octo_live.py --instruction "pick up the circle"
#
#   # Goal-image-conditioned:
#   python run_octo_live.py --goal-image /path/to/goal.png
#
#   # Use a locally finetuned checkpoint:
#   python run_octo_live.py --model-path /path/to/checkpoint \
#                           --dataset-name my_dataset
# =============================================================================

import argparse
import sys
import time
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
        self._cap = cv2.VideoCapture(self.device)
        if not self._cap.isOpened():
            raise RuntimeError(f"Could not open USB camera (device index {self.device}).")
        # Discard the first few frames — most USB cameras output black frames
        # while auto-exposure and white-balance settle.
        for _ in range(10):
            self._cap.read()
        print(f"[CAM] USB camera opened (device {self.device}).")

    def capture_rgb(self) -> np.ndarray:
        ret, frame_bgr = self._cap.read()
        if not ret:
            raise RuntimeError("USB camera read failed.")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def close(self):
        if self._cap:
            self._cap.release()
            self._cap = None
            print("[CAM] USB camera released.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

from octo_policy import OctoPolicy
from tcp_sender  import EEFDeltaSender, ROS2EEFPublisher

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


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Octo VLA live test loop.")

    # Model
    ap.add_argument("--model-path",   type=str, default=None,
                    help="Octo checkpoint path or HF URI (default: latest local finetune, "
                         f"or {MODEL_PATH} if none found)")
    ap.add_argument("--dataset-name", type=str, default=None,
                    help="Dataset key for unnormalisation stats "
                         "(default: ootf_synthetic for local checkpoints, "
                         f"{DATASET_NAME} for pretrained)")
    ap.add_argument("--step",         type=int, default=None,
                    help="Checkpoint step to load (default: latest)")
    ap.add_argument("--window-size",  type=int, default=1,
                    help="Observation history length passed to Octo (default: 1)")

    # Task — mutually exclusive
    task_grp = ap.add_mutually_exclusive_group()
    task_grp.add_argument("--instruction", type=str, default=None,
                          help="Natural-language instruction (language-conditioned mode)")
    task_grp.add_argument("--goal-image",  type=str, default=None,
                          help="Path to a goal image PNG/JPG (goal-conditioned mode)")

    # Camera
    ap.add_argument("--camera-ip",   type=str, default=CAMERA_IP)
    ap.add_argument("--camera-port", type=int, default=CAMERA_PORT)

    # TCP sender
    ap.add_argument("--host", type=str, default=SENDER_HOST,
                    help=f"Receiver host (default: {SENDER_HOST})")
    ap.add_argument("--port", type=int, default=SENDER_PORT,
                    help=f"Receiver port (default: {SENDER_PORT})")

    # Loop control
    ap.add_argument("--steps",      type=int,   default=0,
                    help="Number of steps to run (0 = run until Ctrl+C)")
    ap.add_argument("--step-delay", type=float, default=STEP_DELAY,
                    help=f"Seconds between steps (default: {STEP_DELAY})")
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

    return ap.parse_args()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    args = parse_args()

    # ---- resolve model path and dataset name ----
    if args.model_path is None:
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

    # ---- load Octo ----
    policy = OctoPolicy(
        model_path   = args.model_path,
        dataset_name = args.dataset_name,
        window_size  = args.window_size,
        step         = args.step,
    )

    # ---- set task ----
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

    # ---- connect camera & TCP sender ----
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

    sender_ctx = ROS2EEFPublisher() if args.ros2_output else EEFDeltaSender(args.host, args.port)
    with sender_ctx as sender:
        print(f"\n[INFO] Running. Ctrl+C to stop.")
        if args.ros2_output:
            print(f"[INFO] Publishing EEF deltas to /eef_delta  step_delay={args.step_delay}s\n")
        else:
            print(f"[INFO] Sending to {args.host}:{args.port}  step_delay={args.step_delay}s\n")

        policy.reset()
        step = 0

        if args.show_camera:
            cv2.namedWindow("Octo — camera feed", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Octo — camera feed", 640, 480)

        try:
            while True:
                t0 = time.time()

                # ---- capture frame ----
                if camera is not None:
                    frame_rgb = camera.capture_rgb()
                else:
                    frame_rgb = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)

                # ---- show live feed (before inference so Qt can paint the frame) ----
                if args.show_camera:
                    display = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                    cv2.imshow("Octo — camera feed", display)
                    # Pump the Qt event loop long enough to render the frame,
                    # then check for 'q'.  The larger waitKey here is what actually
                    # makes the window refresh; waitKey(1) after slow inference
                    # doesn't give Qt enough time.
                    if cv2.waitKey(30) & 0xFF == ord('q'):
                        print("[INFO] 'q' pressed — stopping.")
                        break

                # ---- Octo inference ----
                action = policy.step(frame_rgb)
                # action: [dx, dy, dz, drx, dry, drz, gripper]

                # ---- send action ----
                sender.send(action)

                step += 1
                labels = ["dx", "dy", "dz", "drx", "dry", "drz", "grip"]
                vals   = "  ".join(f"{l}={v:+.4f}" for l, v in zip(labels, action))
                print(f"[STEP {step:04d}]  {vals}")

                if args.steps > 0 and step >= args.steps:
                    print(f"[INFO] Reached {args.steps} steps. Done.")
                    break

                # ---- pace the loop ----
                elapsed = time.time() - t0
                wait    = args.step_delay - elapsed
                if wait > 0:
                    time.sleep(wait)

        except KeyboardInterrupt:
            print("\n[INFO] Stopped by user.")

        finally:
            if args.show_camera:
                cv2.destroyAllWindows()
            if camera is not None:
                camera.close()


if __name__ == "__main__":
    main()
