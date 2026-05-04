# =============================================================================
# collect.py — Episode collection in Isaac Sim with Replicator randomisation.
#
# Opens the project USD scene directly (no manual scene construction) and
# records scripted pick-and-place trajectories as .npz files.
# Replicator varies lighting and object appearance between episodes.
#
# Usage:
#   bash launch/collect.sh --output-dir data/exp_01 --n-episodes 500
# =============================================================================

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from task import PickAndPlaceTask


# ─────────────────────────────────────────────────────────────────────────────
# Episode buffer
# ─────────────────────────────────────────────────────────────────────────────

class EpisodeBuffer:
    def __init__(self):
        self._images:  list = []
        self._actions: list = []
        self._instr:   str  = ""

    def add_step(self, image: np.ndarray, action: np.ndarray, instruction: str):
        self._images.append(image)
        self._actions.append(action.astype(np.float32))
        self._instr = instruction

    def save(self, path: Path):
        np.savez_compressed(
            path,
            images      = np.stack(self._images,  axis=0),
            actions     = np.stack(self._actions, axis=0),
            instruction = self._instr,
        )

    def __len__(self):
        return len(self._images)


# ─────────────────────────────────────────────────────────────────────────────
# Data collector
# ─────────────────────────────────────────────────────────────────────────────

class DataCollector:
    ROBOT_PRIM    = "/World/h2017"
    EE_FRAME      = "link_6"
    WRIST_CAM_PRIM = "/World/h2017/link_6/MechEye/Camera"
    OBJ_PRIM      = "/World/pick_object"
    OBJ_SIZE      = 0.04
    HOME_POS      = [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]
    HOME_SETTLE   = 60
    IK_POS_TOL    = 0.005
    IK_ORI_TOL    = 0.01

    def __init__(
        self,
        raw_dir:    Path,
        task:       PickAndPlaceTask,
        n_episodes: int,
        image_size: int = 128,
        seed:       int = 42,
        scene_usd:  str = None,
        urdf_path:  str = None,
        lula_desc:  str = None,
    ):
        root = Path(__file__).parent.parent.parent
        self.raw_dir    = Path(raw_dir)
        self.task       = task
        self.n_episodes = n_episodes
        self.image_size = image_size
        self.rng        = np.random.default_rng(seed)
        self.scene_usd  = scene_usd or str(root / "scenes/usd/doosan_BIC.usd")
        self.urdf_path  = urdf_path or str(root / "scenes/h2017/urdf/h2017.urdf")
        self.lula_desc  = lula_desc or str(root / "scenes/h2017/urdf/h2017_lula.yaml")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        import omni.usd
        import omni.replicator.core as rep
        from isaacsim.core.api import World
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.api.objects import DynamicCuboid
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
        from isaacsim.sensors.camera import Camera

        # ── Scene ─────────────────────────────────────────────────────────────
        # Open the USD file as the authoritative stage. All geometry, materials,
        # lighting, and the wrist camera come from the file.
        omni.usd.get_context().open_stage(self.scene_usd)

        # The USD ActionGraph contains ROS2 camera nodes that require the
        # isaacsim.ros2.bridge extension, which is not loaded during headless
        # collection. Deactivate it before World init so neither the physics
        # step nor Replicator attempt to execute those nodes.
        stage = omni.usd.get_context().get_stage()
        _ag = stage.GetPrimAtPath("/World/ActionGraph")
        if _ag.IsValid():
            _ag.SetActive(False)
            print("[COLLECT] Disabled /World/ActionGraph (ROS2 bridge not loaded)", flush=True)

        world = World(physics_dt=1/60, rendering_dt=1/60, stage_units_in_meters=1.0)
        robot = world.scene.add(Robot(prim_path=self.ROBOT_PRIM, name="robot"))

        # Wrap the existing USD wrist camera for image capture.
        # Resolution is set here for training image size; pose comes from the USD.
        wrist_cam = Camera(
            prim_path  = self.WRIST_CAM_PRIM,
            resolution = (self.image_size, self.image_size),
        )
        world.scene.add(wrist_cam)

        # Physics pick object — repositioned per episode.
        # Replicator randomises its visual appearance; no fixed color here.
        obj = world.scene.add(DynamicCuboid(
            prim_path = self.OBJ_PRIM,
            name      = "pick_object",
            position  = np.array([0.908, 0.24, self.task.pick_z]),
            scale     = np.array([self.OBJ_SIZE] * 3),
        ))

        print("[COLLECT] world.reset() ...", flush=True)
        world.reset()
        print("[COLLECT] wrist_cam.initialize() ...", flush=True)
        wrist_cam.initialize()
        print("[COLLECT] camera ready", flush=True)

        n_dof = robot.num_dof
        robot.get_articulation_controller().set_gains(
            kps=np.full(n_dof, 1e6), kds=np.full(n_dof, 1e5),
        )
        print(f"[COLLECT] robot gains set  (n_dof={n_dof})", flush=True)

        if not Path(self.lula_desc).exists():
            self._generate_lula(self.urdf_path, self.lula_desc)

        print("[COLLECT] LulaKinematicsSolver init ...", flush=True)
        ik = LulaKinematicsSolver(
            robot_description_path=self.lula_desc,
            urdf_path=self.urdf_path,
        )
        print("[COLLECT] IK solver ready", flush=True)

        robot_world_pos, robot_world_quat = robot.get_world_pose()
        rq = robot_world_quat
        robot_R = Rotation.from_quat([rq[1], rq[2], rq[3], rq[0]]).as_matrix()

        # ── Replicator ────────────────────────────────────────────────────────
        # Randomisation only — no render product needed.
        # rep.orchestrator.step() applies these to the scene without conflicting
        # with the USD's existing ActionGraph camera nodes.
        print("[COLLECT] Replicator setup ...", flush=True)
        with rep.trigger.on_frame():
            with rep.get.light():
                rep.modify.attribute(
                    name  = "inputs:intensity",
                    value = rep.distribution.uniform(300, 4000),
                )
                rep.modify.attribute(
                    name  = "inputs:colorTemperature",
                    value = rep.distribution.uniform(3200, 7500),
                )
            with rep.get.prims(path_pattern=self.OBJ_PRIM):
                rep.randomizer.color(
                    colors=rep.distribution.uniform((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
                )
        print("[COLLECT] Replicator ready", flush=True)

        # ── Episode loop ──────────────────────────────────────────────────────
        ep_idx = len(list(self.raw_dir.glob("*.npz")))
        print(f"[COLLECT] {self.n_episodes} episodes (resuming from {ep_idx})", flush=True)
        consecutive_failures = 0
        MAX_CONSECUTIVE_FAILURES = 20

        while ep_idx < self.n_episodes:
            waypoints, instruction, pick_pos, _ = self.task.sample(self.rng)
            obj.set_world_pose(position=pick_pos)

            # New lighting + object appearance for this episode.
            rep.orchestrator.step(rt_subframes=4, pause_timeline=False)

            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=self.HOME_POS)
            )
            for _ in range(self.HOME_SETTLE):
                world.step(render=False)

            buf, prev_pos, prev_rot, prev_rgb, ok = EpisodeBuffer(), None, None, None, True

            for wp in waypoints:
                wp_pos_robot = robot_R.T @ (wp.position - robot_world_pos)
                q_target, ik_ok = ik.compute_inverse_kinematics(
                    self.EE_FRAME, wp_pos_robot, wp.orientation,
                    position_tolerance=self.IK_POS_TOL,
                    orientation_tolerance=self.IK_ORI_TOL,
                )
                if not ik_ok:
                    print(f"[COLLECT] ep {ep_idx}: IK failed — "
                          f"robot-frame target={wp_pos_robot.round(3)} "
                          f"(world={wp.position.round(3)}) — skipping", flush=True)
                    ok = False
                    break

                q_start = robot.get_joint_positions().copy()
                for s in range(wp.n_steps):
                    # Capture image BEFORE the move so each (obs, action) pair
                    # represents the state the model must act on, not the result.
                    rgba    = wrist_cam.get_rgba()
                    cur_rgb = (
                        (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
                        if rgba is not None and rgba.ndim >= 3 else None
                    )

                    alpha = (s + 1) / wp.n_steps
                    robot.get_articulation_controller().apply_action(
                        ArticulationAction(
                            joint_positions=(1 - alpha) * q_start + alpha * q_target
                        )
                    )
                    world.step(render=True)

                    cur_pos, cur_rot = ik.compute_forward_kinematics(
                        self.EE_FRAME, robot.get_joint_positions()
                    )
                    if prev_pos is None:
                        prev_pos, prev_rot, prev_rgb = cur_pos, cur_rot.copy(), cur_rgb
                        continue

                    d_pos   = cur_pos - prev_pos
                    d_euler = Rotation.from_matrix(cur_rot @ prev_rot.T).as_euler('xyz')
                    action  = np.concatenate([d_pos, d_euler, [wp.gripper]])

                    if prev_rgb is not None:
                        buf.add_step(prev_rgb, action.astype(np.float32), instruction)
                    prev_pos, prev_rot, prev_rgb = cur_pos, cur_rot.copy(), cur_rgb

            if ok and len(buf) > 0:
                buf.save(self.raw_dir / f"episode_{ep_idx:06d}.npz")
                ep_idx += 1
                consecutive_failures = 0
                print(f"[COLLECT] {ep_idx}/{self.n_episodes}  "
                      f"steps={len(buf)}  '{instruction}'", flush=True)
            else:
                consecutive_failures += 1
                if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                    raise RuntimeError(
                        f"{MAX_CONSECUTIVE_FAILURES} consecutive episode failures. "
                        f"Check IK, camera, and workspace bounds."
                    )

    @staticmethod
    def _generate_lula(urdf_path: str, output_path: str):
        import xml.etree.ElementTree as ET
        root      = ET.parse(urdf_path).getroot()
        joints    = [j.get('name') for j in root.findall('joint')
                     if j.get('type') == 'revolute']
        parent    = root.find("joint/parent")
        root_link = parent.get('link') if parent is not None else 'base_link'
        n = len(joints)
        Path(output_path).write_text(
            "api_version: 1.0\n\ncspace:\n"
            + "".join(f"    - {j}\n" for j in joints)
            + f"\nroot_link: {root_link}\n\n"
            f"default_q: [{', '.join(['0.0']*n)}]\n\n"
            f"acceleration_limits: [{', '.join(['40.0']*n)}]\n"
            f"jerk_limits: [{', '.join(['500.0']*n)}]\n"
        )
        print(f"[COLLECT] Generated LULA description → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Collect synthetic episodes in Isaac Sim with Replicator domain randomisation."
    )
    ap.add_argument("--output-dir",   required=True, help="Directory for .npz episode files")
    ap.add_argument("--n-episodes",   type=int,   default=200)
    ap.add_argument("--image-size",   type=int,   default=128)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--scene-usd",    default=None)
    ap.add_argument("--pick-x",       type=float, nargs=2, default=[0.7,  1.2],  metavar=("MIN", "MAX"))
    ap.add_argument("--pick-y",       type=float, nargs=2, default=[-0.1, 0.6],  metavar=("MIN", "MAX"))
    ap.add_argument("--pick-z",       type=float, default=0.6)
    ap.add_argument("--place-x",      type=float, nargs=2, default=[-1.0,  0.0], metavar=("MIN", "MAX"))
    ap.add_argument("--place-y",      type=float, nargs=2, default=[-1.5, -1.0], metavar=("MIN", "MAX"))
    ap.add_argument("--place-z",      type=float, default=0.806)
    ap.add_argument("--max-reach-xy", type=float, default=1.65)
    return ap.parse_args()


def main():
    import traceback as _tb
    args = parse_args()
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": True})
    _failed = False
    try:
        DataCollector(
            raw_dir    = args.output_dir,
            task       = PickAndPlaceTask(
                pick_x       = tuple(args.pick_x),
                pick_y       = tuple(args.pick_y),
                pick_z       = args.pick_z,
                place_x      = tuple(args.place_x),
                place_y      = tuple(args.place_y),
                place_z      = args.place_z,
                max_reach_xy = args.max_reach_xy,
            ),
            n_episodes = args.n_episodes,
            image_size = args.image_size,
            seed       = args.seed,
            scene_usd  = args.scene_usd,
        ).run()
    except Exception:
        # Print before close() because simulation_app.close() calls sys.exit(0)
        # internally, which would otherwise swallow the traceback entirely.
        print("\n[COLLECT] ── FATAL ERROR ──────────────────────────", flush=True)
        _tb.print_exc()
        print("[COLLECT] ──────────────────────────────────────────\n", flush=True)
        _failed = True
    finally:
        simulation_app.close()
    if _failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
