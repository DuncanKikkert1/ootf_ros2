# =============================================================================
# debug/debug_run.py — Dry-run episode collector for trajectory verification.
#
# Mirrors the full collect.py pipeline (scene setup, IK pre-solve, joint-space
# execution, camera capture) but writes no .npz files to disk.
#
# Per-episode output:
#   - Object type, dimensions, and surface grab angle
#   - For each waypoint: name, target joints (degrees), gripper state
#   - Arrival EEF world position after each waypoint completes
#
# Usage:
#   python debug/debug_run.py --seq seq1          # 1 cube episode
#   python debug/debug_run.py --seq seq2          # 1 cylinder episode
#   python debug/debug_run.py --seq seq3          # 1 pyramid episode
#   python debug/debug_run.py --seq seq4          # cube → cylinder → pyramid
#   python debug/debug_run.py                     # 3 random episodes (default)
#   python debug/debug_run.py --seq seq4 --seed 7
# =============================================================================

import argparse
import sys
from pathlib import Path

# Allow importing from src/isaac.
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "isaac"))

from collect import DataCollector
from task import PickAndPlaceTask

# ── Predefined episode sequences ──────────────────────────────────────────────
SEQUENCES = {
    "seq1": dict(obj_types=["cube"],                          n_episodes=1),
    "seq2": dict(obj_types=["cylinder"],                      n_episodes=1),
    "seq3": dict(obj_types=["pyramid"],                       n_episodes=1),
    "seq4": dict(obj_types=["cube", "cylinder", "pyramid"],   n_episodes=3),
}


def parse_args():
    ap = argparse.ArgumentParser(
        description="Dry-run trajectory verifier — no files written."
    )
    ap.add_argument(
        "--seq",
        choices=list(SEQUENCES),
        default=None,
        help=(
            "seq1: 1×cube  |  seq2: 1×cylinder  |  "
            "seq3: 1×pyramid  |  seq4: cube→cylinder→pyramid"
        ),
    )
    ap.add_argument("--n-episodes",   type=int,   default=3,
                    help="Ignored when --seq is set.")
    ap.add_argument("--time-scale",   type=int,   default=1,
                    help="Multiply each waypoint's n_steps by this factor (e.g. 3 = 3× slower).")
    ap.add_argument("--image-size",   type=int,   default=128)
    ap.add_argument("--seed",         type=int,   default=0)
    ap.add_argument("--scene-usd",    default=None)
    ap.add_argument("--pick-x",       type=float, nargs=2, default=[0.59, 1.26])
    ap.add_argument("--pick-y",       type=float, nargs=2, default=[-0.3, 0.8])
    ap.add_argument("--surface-z",    type=float, default=0.43)
    ap.add_argument("--place-x",      type=float, nargs=2, default=[-0.95, 0.85])
    ap.add_argument("--place-y",      type=float, nargs=2, default=[-1.55, -1.0])
    ap.add_argument("--place-z",      type=float, default=0.85)
    ap.add_argument("--max-reach-xy", type=float, default=1.65)
    ap.add_argument("--eef-z-offset",      type=float, default=0.215)
    ap.add_argument("--gripper-wait-sec",  type=float, default=3.0,
                    help="Seconds to hold the arm at pick and retry gripper close (default 3.0).")
    return ap.parse_args()


def main():
    import traceback as _tb
    args = parse_args()

    # Resolve sequence settings.
    if args.seq:
        seq_cfg       = SEQUENCES[args.seq]
        n_episodes    = seq_cfg["n_episodes"]
        forced_seq    = seq_cfg["obj_types"]
        print(f"[DEBUG] Running predefined sequence '{args.seq}': "
              f"{' → '.join(forced_seq)}", flush=True)
    else:
        n_episodes = args.n_episodes
        forced_seq = None
        print(f"[DEBUG] Running {n_episodes} random episode(s).", flush=True)

    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False})

    _failed = False
    try:
        DataCollector(
            raw_dir    = "/tmp/debug_run_scratch",
            task       = PickAndPlaceTask(
                pick_x       = tuple(args.pick_x),
                pick_y       = tuple(args.pick_y),
                surface_z    = args.surface_z,
                place_x      = tuple(args.place_x),
                place_y      = tuple(args.place_y),
                place_z      = args.place_z,
                max_reach_xy = args.max_reach_xy,
                eef_z_offset = args.eef_z_offset,
            ),
            n_episodes           = n_episodes,
            image_size           = args.image_size,
            seed                 = args.seed,
            scene_usd            = args.scene_usd,
            dry_run              = True,
            forced_obj_sequence  = forced_seq,
            time_scale           = args.time_scale,
            gripper_wait_steps   = int(args.gripper_wait_sec * 60),
        ).run()
    except Exception:
        print("\n[DEBUG] ── FATAL ERROR ──────────────────────────", flush=True)
        _tb.print_exc()
        print("[DEBUG] ──────────────────────────────────────────\n", flush=True)
        _failed = True
    finally:
        simulation_app.close()

    if _failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
