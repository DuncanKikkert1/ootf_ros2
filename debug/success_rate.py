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

# SortingTask target platforms (world frame) — MUST match task.py
# SORT_PLATFORM_CENTRES and sim_node.py.  Used for the --sort-target placement-
# accuracy metric.  Platform top surface ≈ 0.860 m; a 10 cm cube resting on it
# centres at z ≈ 0.91.  (Phased sorting uses the SAME instructions as pick-place
# — only the place LOCATION differs — so no instruction change is needed.)
SORT_PLATFORM_CENTRES = {
    "cube":     np.array([-0.5, -1.28, 0.855]),
    "cylinder": np.array([ 0.0, -1.28, 0.855]),
    "pyramid":  np.array([ 0.5, -1.28, 0.855]),
}


def _xyz(v):
    """Format a world XYZ vector for the result log, or — if missing."""
    return f"[{v[0]:+.3f} {v[1]:+.3f} {v[2]:+.3f}]" if v is not None else "—"


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
    ap.add_argument("--reset-settle", type=float, default=3.5, help="Seconds for the arm to home + cube to settle after reset")
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
    ap.add_argument("--grasp-dwell-steps",   type=int,   default=40,
                    help="Hold + retry the gripper this many frames at the "
                         "pick→place transition until the suction latches "
                         "(mirrors collection's gripper_wait). 0 disables.")
    ap.add_argument("--pick-dz-bias",         type=float, default=0.0,
                    help="Constant Z delta added to every PICK-phase action "
                         "(negative = down). Default 0 = off, so the harness "
                         "measures raw model behaviour. Set negative to "
                         "compensate the model's measured ~+0.0025 m/step upward "
                         "dz bias (holdout eval) that compounds into a ~4-6cm "
                         "hover above the cube, so the cup reaches contact before "
                         "the gripper fires. Pick only; stops at grip commit; "
                         "backstopped by the sim's MIN_EEF_Z_WORLD floor.")
    ap.add_argument("--pick-dx-bias",         type=float, default=0.0,
                    help="Constant X delta added to every PICK-phase action to "
                         "cancel the model's systematic lateral overshoot (eval "
                         "showed the cup lands ~+15cm in +X of the cube). Negative "
                         "pulls the reach back. Applied ALSO under --contact-gate "
                         "(the gate only handles Z). 0 disables.")
    ap.add_argument("--pick-dy-bias",         type=float, default=0.0,
                    help="Constant Y delta added to every PICK-phase action to "
                         "cancel the model's systematic +Y overshoot (eval showed "
                         "the cup lands ~+17cm in +Y of the cube). Negative pulls "
                         "the reach back. Applied ALSO under --contact-gate. 0 disables.")
    ap.add_argument("--place-on-miss", action="store_true",
                    help="Run the place phase even when the grasp didn't latch. "
                         "Default: end the attempt on a missed grasp, since the "
                         "empty place swing just thrashes IK toward the out-of-"
                         "reach conveyor (HOME stalls) with no cube in hand.")
    ap.add_argument("--contact-gate", action="store_true",
                    help="Contact-gated grasp: commit + latch the moment the sim "
                         "reports the cup is over the cube within suction range "
                         "(/grasp_align in_range), instead of the model's grip "
                         "timing or the open-loop dz bias. Stops the descent "
                         "overshooting past the cube into the pallet. Overrides "
                         "--pick-dz-bias (vertical is handled by the gate).")
    ap.add_argument("--place-gate", action="store_true",
                    help="Place-gated release (symmetric to --contact-gate): OPEN "
                         "the gripper the instant the held cube is over the place "
                         "target and low enough to drop, instead of the model's "
                         "imprecise place (which rams the belt / skips home). "
                         "Target = the --sort-target platform if given, else the "
                         "conveyor belt region (cube y < --belt-y-max).")
    ap.add_argument("--place-z", type=float, default=0.85,
                    help="Place-target surface height (m) for --place-gate "
                         "(conveyor/belt default; matches collect --place-z).")
    ap.add_argument("--place-drop-clear", type=float, default=0.08,
                    help="--place-gate releases when the cube is 0..this many m "
                         "above the place-target surface (drop clearance).")
    ap.add_argument("--temporal-ensemble", action="store_true",
                    help="ACT-style temporal ensembling: average the overlapping "
                         "predictions the model makes for each execution step "
                         "(it forecasts a 4-step chunk each step) to low-pass the "
                         "per-step noise and reduce closed-loop drift. Position/"
                         "rotation only; the gripper signal stays crisp.")
    ap.add_argument("--ensemble-decay", "--ensemble-m", type=float, default=0.1,
                    dest="ensemble_decay",
                    help="Temporal-ensemble weight decay: w=exp(-decay*age) over the "
                         "overlapping predictions (age 0 = newest). Lower = more "
                         "smoothing (uniform), higher = favour the newest. "
                         "(--ensemble-m is a deprecated alias.)")
    ap.add_argument("--record-dagger",    default=None, metavar="DIR",
                    help="Record per-step on-policy rollout traces to DIR for "
                         "DAgger relabeling: each attempt → one npz of (images, "
                         "proprios, cube_world, phases, actions_taken, outcome). "
                         "The offline oracle relabeler turns these policy-visited "
                         "states into corrective action labels.")
    # success regions (defaults match sim_node pick centre + collect place-y belt)
    ap.add_argument("--lift-z-min",  type=float, default=0.05, help="Cube lift (m) for grasp success")
    ap.add_argument("--belt-y-max",  type=float, default=-0.80, help="Cube y below this = on belt side")
    ap.add_argument("--floor-z-min", type=float, default=0.30, help="Cube z above this = not dropped to floor")
    # sorting eval: place-accuracy onto a designated platform (needs the sim
    # launched with OOTF_SORT_PLATFORMS=1 so the platforms exist).
    ap.add_argument("--sort-target", choices=["cube", "cylinder", "pyramid"], default=None,
                    help="Score placement against this object's sort platform "
                         "(distance to platform centre) instead of the belt region.")
    ap.add_argument("--sort-xy-tol", type=float, default=0.06,
                    help="Cube-centre XY distance (m) to the platform centre to "
                         "count as correctly placed (platform is 10cm wide).")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.model_path is None:
        args.model_path = _latest_checkpoint()
        if args.model_path is None:
            sys.exit("[ERR] No local checkpoint found — pass --model-path.")
        print(f"[INFO] Using checkpoint: {args.model_path}")

    import rclpy
    from std_msgs.msg import Float64MultiArray, Empty, String

    rclpy.init()
    node      = rclpy.create_node("success_rate_harness")
    delta_pub = node.create_publisher(Float64MultiArray, "/eef_delta",   10)
    reset_pub = node.create_publisher(Empty,             "/reset_scene",  1)
    cube      = {"v": None}
    eefst     = {"v": None, "frames": 0}
    latched   = {"v": False}   # is the pick cube currently held by the gripper?
    node.create_subscription(Float64MultiArray, "/cube_pose",
                             lambda m: cube.update(v=np.array(m.data[:3])), 10)
    def _eef_cb(m):
        eefst["v"] = np.array(m.data[:7]); eefst["frames"] += 1
    node.create_subscription(Float64MultiArray, "/eef_state", _eef_cb, 10)
    # /gripper_status = "status|/World/pick_cube,..." — cube path appears once latched.
    node.create_subscription(String, "/gripper_status",
                             lambda m: latched.update(v=("pick_cube" in m.data)), 10)
    # /grasp_align = [lateral_xy, vertical_clearance, in_range] — cup-vs-cube
    # geometry in world frame, for the contact-gated grasp.
    align = {"in_range": False, "lat": None, "vclear": None, "cup": None}
    node.create_subscription(Float64MultiArray, "/grasp_align",
                             lambda m: align.update(
                                 lat=m.data[0], vclear=m.data[1], in_range=(m.data[2] > 0.5),
                                 cup=(np.array(m.data[3:6]) if len(m.data) >= 6 else None)), 10)
    # /cube_pose_rf = cube position in the robot/base frame (same frame as
    # proprio + actions) — the DAgger oracle's target.
    cube_rf = {"v": None}
    node.create_subscription(Float64MultiArray, "/cube_pose_rf",
                             lambda m: cube_rf.update(v=np.array(m.data[:3])), 10)

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

    def grasp_dwell(n):
        """Hold the pose with the gripper closed until the suction latches (or
        n frames elapse).  Mirrors collection's gripper_wait: without it the
        place phase lifts the arm away before a near-miss grasp can latch."""
        hold = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]   # zero motion, gripper closed
        for _ in range(n):
            delta_pub.publish(Float64MultiArray(data=hold))
            wait_frames(args.sync_frames if args.sync_frames > 0 else 1)
            if latched["v"]:
                return True
        return latched["v"]

    policy = OctoPolicy(model_path=args.model_path, dataset_name=args.dataset_name,
                        window_size=args.window_size, step=args.step)

    # --- temporal ensembling state ---------------------------------------
    # The model forecasts a (horizon, 7) chunk each step.  At execution step t,
    # the chunks issued at t, t-1, … t-(H-1) each contain a prediction for t
    # (at chunk index = age).  Averaging them (ACT-style) low-passes the per-
    # step prediction noise that compounds into closed-loop XY drift.
    ens_buf = []   # recent chunks, newest last; trimmed to the action horizon

    def policy_reset():
        """Reset the policy AND drop the ensemble buffer — predictions across a
        phase switch (pick→place→home) are a different distribution and must
        not be averaged together."""
        policy.reset()
        ens_buf.clear()

    def get_action(frame, proprio):
        chunk = np.asarray(policy.step(frame, proprio=proprio), dtype=np.float64)  # (H,7)
        if not args.temporal_ensemble:
            return chunk[0].copy()
        ens_buf.append(chunk)
        if len(ens_buf) > chunk.shape[0]:
            ens_buf.pop(0)
        # gather each buffered chunk's prediction for the CURRENT step (idx=age)
        acts, ws = [], []
        L = len(ens_buf)
        for p, ch in enumerate(ens_buf):
            age = (L - 1) - p
            if age < ch.shape[0]:
                acts.append(ch[age])
                ws.append(np.exp(-args.ensemble_decay * age))
        ws  = np.array(ws); ws /= ws.sum()
        out = (ws[:, None] * np.array(acts)).sum(axis=0)
        out[6] = ens_buf[-1][0, 6]   # keep the gripper decision crisp (newest)
        return out.copy()

    def apply_phase(phase):
        policy.set_task_text(PHASE_INSTRUCTIONS[phase].replace("{obj}", args.phase_object))

    camera = ROS2Camera(args.ros2_topic)
    camera.connect()
    print("[INFO] Waiting for /cube_pose and /eef_state …")
    while cube["v"] is None or eefst["v"] is None:
        rclpy.spin_once(node, timeout_sec=0.1)

    dagger_dir = None
    if args.record_dagger:
        dagger_dir = Path(args.record_dagger)
        dagger_dir.mkdir(parents=True, exist_ok=True)
        run_stamp  = time.strftime("%Y%m%d_%H%M%S")
        print(f"[DAGGER] Recording rollout traces → {dagger_dir}")

    results = []
    try:
        for attempt in range(1, args.attempts + 1):
            print(f"\n========== ATTEMPT {attempt}/{args.attempts} ==========")
            reset_pub.publish(Empty())
            spin(args.reset_settle)
            cube_start = cube["v"].copy()
            print(f"[RESET] cube start: {cube_start.round(3)}")

            policy_reset()
            if args.phased:
                apply_phase("pick")
            elif args.instruction:
                policy.set_task_text(args.instruction)
            else:
                apply_phase("pick")   # fall back to phased pick text

            phase = "pick"; prev_grip = 0.0; grip_hold = 0; low_grip = 0
            cube_lift = 0.0
            grip_fired = False        # did the gripper ever commit closed?
            min_xy = float("inf")     # legacy cross-frame proxy (link_6 vs world cube) — UNRELIABLE
            min_lat = float("inf")    # REAL closest cup→cube lateral dist (/grasp_align, world frame)
            min_vclear = float("inf") # lowest cup-tip clearance above cube top (<0 = rammed through it)
            commit_lat = commit_vclear = None  # cup↔cube geometry AT the grip-commit (= when the
                                               # suction rays fire) — the position that decides the grasp
            # RAW positions (world XYZ): cup vs cube, so the distance is unambiguous.
            min_dist = float("inf"); min_cup = min_cube = None   # closest 3D approach + the positions
            commit_cup = commit_cube = None                      # cup & cube world XYZ at grip-commit
            # DAgger trace: per-step policy-visited states for offline relabeling
            rec_img, rec_prop, rec_cube, rec_phase, rec_act = [], [], [], [], []
            rec_align = []   # [lat, vclear, in_range] from /grasp_align (NaN if absent)
            rec_cube_rf = []  # cube in robot/base frame — the oracle target

            for step in range(1, args.max_steps + 1):
                frame = camera.capture_rgb()
                proprio = eefst["v"]
                action = get_action(frame, proprio)
                if args.zero_rotation:
                    action[3:6] = 0.0

                # Contact-gated grasp: the instant the sim reports the cup is
                # over the cube and within suction range, commit + latch HERE —
                # before the open-loop descent overshoots past the cube into the
                # pallet (the attempt-19 failure: dead-on XY but dz-bias drove
                # the cup down past the cube, ray hit the pallet, no latch).
                if (args.contact_gate and args.phased and phase == "pick"
                        and align["in_range"]):
                    grip_fired = True
                    commit_lat, commit_vclear = align["lat"], align["vclear"]
                    commit_cup = align["cup"]; commit_cube = cube["v"]
                    latched_now = grasp_dwell(args.grasp_dwell_steps)
                    if not latched_now and not args.place_on_miss:
                        break
                    phase = "place"; apply_phase(phase); policy_reset()
                    prev_grip = 1.0
                    continue

                # Place-gated release (symmetric to the pick gate above): the
                # instant the held cube is over the place target AND low enough to
                # drop, OPEN the gripper HERE instead of trusting the model's
                # imprecise place (which rams the belt / skips home).  Additive —
                # if it never triggers, the model's own release below still works.
                if (args.place_gate and args.phased and phase == "place"
                        and cube["v"] is not None):
                    _cp = cube["v"]
                    if args.sort_target:
                        _tgt   = SORT_PLATFORM_CENTRES[args.sort_target]
                        _over  = float(np.linalg.norm(_cp[:2] - _tgt[:2])) < args.sort_xy_tol
                        _tgt_z = float(_tgt[2])
                    else:
                        _over  = _cp[1] < args.belt_y_max     # over the conveyor belt
                        _tgt_z = args.place_z
                    if _over and 0.0 <= (_cp[2] - _tgt_z) <= args.place_drop_clear:
                        action[6] = 0.0                        # open → release
                        delta_pub.publish(Float64MultiArray(
                            data=[float(v) for v in action]))
                        if args.sync_frames > 0:
                            wait_frames(args.sync_frames)
                        else:
                            spin(args.step_delay)
                        phase = "home"; apply_phase(phase); policy_reset()
                        prev_grip = 0.0
                        continue

                # Compensate the model's systematic upward dz bias (holdout:
                # pred dz mean ~+0.002 vs GT ~-0.001) which compounds into a
                # ~4cm hover, so the gripper fires above the cube.  Pick-phase
                # approach only — place/home need the model's raw vertical
                # motion; the gripper commit below ends the pick phase.  Skipped
                # under --contact-gate (the gate handles the vertical instead).
                if phase == "pick":
                    # Lateral overshoot correction — applied even under --contact-gate
                    # (the gate only handles the vertical), since the cup must still
                    # be brought OVER the cube in XY.
                    action[0] += args.pick_dx_bias
                    action[1] += args.pick_dy_bias
                    # Z bias only when the gate isn't handling the descent.
                    if args.pick_dz_bias != 0.0 and not args.contact_gate:
                        action[2] += args.pick_dz_bias

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

                if args.phased and phase == "pick" and prev_grip < 0.5 and action[6] > 0.5:
                    # Grasp committed: hold + keep closing until the suction
                    # latches BEFORE the place phase lifts the arm.
                    grip_fired = True
                    commit_lat, commit_vclear = align["lat"], align["vclear"]
                    commit_cup = align["cup"]; commit_cube = cube["v"]
                    latched_now = grasp_dwell(args.grasp_dwell_steps)
                    if not latched_now and not args.place_on_miss:
                        # Missed grasp — skip the place phase.  With no cube
                        # latched it would just swing the empty arm at the
                        # out-of-reach conveyor, spamming IK failures/HOME
                        # stalls.  cube_lift (measured below) stays ~0 = a
                        # correctly-scored miss.
                        break
                    phase = "place"; apply_phase(phase); policy_reset()
                    prev_grip = 1.0
                    continue
                elif args.phased and phase == "place" and prev_grip > 0.5 and action[6] < 0.5:
                    phase = "home"; apply_phase(phase); policy_reset()

                if action[6] > 0.5:
                    grip_fired = True
                prev_grip = action[6]
                delta_pub.publish(Float64MultiArray(data=[float(v) for v in action]))

                # DAgger: record the state the policy just acted from (image it
                # saw + proprio fed + world cube pose + phase) and the action it
                # took.  The offline oracle relabeler replaces actions_taken with
                # the corrective expert delta; image/proprio/cube are the inputs.
                if dagger_dir is not None and cube["v"] is not None:
                    rec_img.append(cv2.resize(frame, (128, 128)).astype(np.uint8))
                    rec_prop.append(np.asarray(proprio, dtype=np.float32))
                    rec_cube.append(np.asarray(cube["v"], dtype=np.float32))
                    rec_phase.append(phase)
                    rec_act.append(np.asarray(action, dtype=np.float32))
                    _nan = float("nan")
                    rec_align.append(np.array(
                        [align["lat"] if align["lat"] is not None else _nan,
                         align["vclear"] if align["vclear"] is not None else _nan,
                         1.0 if align["in_range"] else 0.0], dtype=np.float32))
                    rec_cube_rf.append(np.asarray(
                        cube_rf["v"] if cube_rf["v"] is not None else [_nan]*3,
                        dtype=np.float32))

                if cube["v"] is not None:
                    cube_lift = max(cube_lift, cube["v"][2] - cube_start[2])
                    # eef_state proprio xy ≈ EEF in robot frame; cube_pose is world.
                    # Not the same frame, but the per-attempt MINIMUM is still a
                    # useful relative proxy for "did it get close in XY".
                    if eefst["v"] is not None:
                        min_xy = min(min_xy, float(np.linalg.norm(
                            eefst["v"][:2] - cube["v"][:2])))
                # REAL geometry from the sim (/grasp_align, world frame): how close
                # did the suction cup actually get to the cube, and how far did it
                # descend past the cube top (negative vclear = rammed through).
                if align["lat"] is not None:
                    min_lat    = min(min_lat, align["lat"])
                    min_vclear = min(min_vclear, align["vclear"])
                if align["cup"] is not None and cube["v"] is not None:
                    _d3 = float(np.linalg.norm(align["cup"] - cube["v"]))
                    if _d3 < min_dist:
                        min_dist, min_cup, min_cube = _d3, align["cup"].copy(), cube["v"].copy()

                if args.sync_frames > 0:
                    wait_frames(args.sync_frames)
                else:
                    spin(args.step_delay)

            cp = cube["v"]
            grasped = cube_lift > args.lift_z_min
            if args.sort_target:
                tgt = SORT_PLATFORM_CENTRES[args.sort_target]
                place_err = float(np.linalg.norm(cp[:2] - tgt[:2])) if cp is not None else float("inf")
                placed = (cp is not None and place_err < args.sort_xy_tol and cp[2] > args.floor_z_min)
            else:
                place_err = None
                placed = (cp is not None and cp[1] < args.belt_y_max and cp[2] > args.floor_z_min)
            results.append((grasped, placed, place_err, min_lat, min_vclear,
                            commit_lat, commit_vclear, min_dist))
            _perr = f"  place_err={place_err*1000:.0f}mm" if place_err not in (None, float("inf")) else ""
            print(f"[RESULT] grasped={grasped}  placed={placed}{_perr}  "
                  f"grip_fired={grip_fired}  max_lift={cube_lift*1000:.0f}mm")
            # RAW world XYZ — exactly how far the suction cup is from the cube.
            if min_cup is not None:
                _d = (min_cup - min_cube) * 1000
                print(f"   closest:  cup={_xyz(min_cup)}  cube={_xyz(min_cube)}  "
                      f"dist={min_dist*1000:.0f}mm  Δ(x,y,z)=({_d[0]:+.0f},{_d[1]:+.0f},{_d[2]:+.0f})mm")
            if commit_cup is not None:
                _c = (commit_cup - commit_cube) * 1000
                print(f"   grip@:    cup={_xyz(commit_cup)}  cube={_xyz(commit_cube)}  "
                      f"dist={float(np.linalg.norm(commit_cup-commit_cube))*1000:.0f}mm  "
                      f"Δ(x,y,z)=({_c[0]:+.0f},{_c[1]:+.0f},{_c[2]:+.0f})mm")
            else:
                print("   grip@:    (gripper never committed)")

            if dagger_dir is not None and rec_img:
                out = dagger_dir / f"rollout_{run_stamp}_{attempt:03d}.npz"
                np.savez_compressed(
                    out,
                    images        = np.stack(rec_img),                       # (N,128,128,3) u8
                    proprios      = np.stack(rec_prop),                      # (N,7) f32 robot frame
                    cube_world    = np.stack(rec_cube),                      # (N,3) f32 world
                    cube_rf       = np.stack(rec_cube_rf),                   # (N,3) f32 robot/base frame
                    phases        = np.array(rec_phase),                     # (N,) str
                    actions_taken = np.stack(rec_act),                       # (N,7) f32 (policy's action)
                    align         = np.stack(rec_align),                     # (N,3) f32 [lat,vclear,in_range]
                    grasped       = np.array(grasped),
                    placed        = np.array(placed),
                    cube_start    = cube_start.astype(np.float32),
                )
                print(f"[DAGGER] saved {len(rec_img)} steps → {out.name}")

        # tuple = (grasped, placed, place_err, min_lat, min_vclear, commit_lat, commit_vclear)
        g = sum(1 for r in results if r[0])
        p = sum(1 for r in results if r[1])
        n = len(results)
        print("\n══════════ SUCCESS RATE ══════════")
        print(f"Attempts : {n}")
        print(f"Grasped  : {g}/{n}  ({100*g/n:.0f}%)")
        if args.sort_target:
            print(f"Placed on {args.sort_target} platform : {p}/{n}  ({100*p/n:.0f}%)  "
                  f"[XY tol {args.sort_xy_tol*1000:.0f}mm]")
            errs = [r[2] for r in results if r[0] and r[2] not in (None, float("inf"))]
            if errs:
                print(f"Placement XY error (grasped) : mean={np.mean(errs)*1000:.0f}mm  "
                      f"min={np.min(errs)*1000:.0f}mm  max={np.max(errs)*1000:.0f}mm")
        else:
            print(f"Placed   : {p}/{n}  ({100*p/n:.0f}%)")
        # THE metric that matters: where the cup was WHEN the gripper fired (rays).
        # A grasp needs lat≈0 (over the cube) AND vclear≈0 (at the cube top).
        clat = [r[5] for r in results if r[5] not in (None, float("inf"))]
        cvcl = [r[6] for r in results if r[6] not in (None, float("inf"))]
        if clat:
            print(f"Gripper @ commit (n={len(clat)}/{n} fired) : "
                  f"lateral mean={np.mean(clat)*1000:.0f}mm min={np.min(clat)*1000:.0f}mm  |  "
                  f"vclear mean={np.mean(cvcl)*1000:+.0f}mm min={np.min(cvcl)*1000:+.0f}mm")
            print(f"  (lat<30mm AND |vclear|<20mm on commit : "
                  f"{sum(1 for la,vc in zip(clat,cvcl) if la<0.03 and abs(vc)<0.02)}/{len(clat)} "
                  f"= grasp-ready commits)")
        else:
            print("Gripper @ commit : never fired the gripper in any attempt")
        # RAW closest 3D cup→cube distance over the run (no derivation, just XYZ norm)
        d3 = [r[7] for r in results if r[7] != float("inf")]
        if d3:
            print(f"Closest cup→cube 3D distance : mean={np.mean(d3)*1000:.0f}mm  "
                  f"min={np.min(d3)*1000:.0f}mm   [<30mm on {sum(1 for d in d3 if d<0.03)}/{len(d3)}]")
        # context: deepest ram over the run (vclear<0 = through the cube into pallet)
        vcls = [r[4] for r in results if r[4] != float("inf")]
        if vcls:
            print(f"Deepest descent past cube top (context) : {np.min(vcls)*1000:+.0f}mm  "
                  f"[rammed <-10mm on {sum(1 for v in vcls if v<-0.01)}/{len(vcls)}]")
    except KeyboardInterrupt:
        print("\n[INFO] Stopped by user.")
    finally:
        camera.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
