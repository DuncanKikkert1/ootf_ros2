# =============================================================================
# collect.py — Episode collection in Isaac Sim with Replicator randomisation.
#
# Opens the project USD scene directly (no manual scene construction) and
# records scripted pick-and-place trajectories as .npz files.
# Replicator varies lighting and object appearance between episodes.
# Each episode randomly uses either a cube or a sphere as the pick object.
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
    ROBOT_PRIM     = "/World/h2017"
    EE_FRAME       = "link_6"
    WRIST_CAM_PRIM = "/World/h2017/link_6/MechEye/MechEye/Camera"
    CUBE_PRIM      = "/World/pick_cube"
    SPHERE_PRIM    = "/World/pick_sphere"
    PARK_POS       = np.array([0.0, 0.0, -5.0])   # underground — inactive object
    HOME_POS       = [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]
    HOME_SETTLE    = 60
    IK_POS_TOL     = 0.005
    IK_ORI_TOL     = 0.01

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
        self.scene_usd  = scene_usd or str(root / "scenes/usd/sim2.usd")
        self.urdf_path  = urdf_path or str(root / "scenes/h2017/urdf/h2017.urdf")
        self.lula_desc  = lula_desc or str(root / "scenes/h2017/urdf/h2017_lula.yaml")
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        from gripper import SurfaceGripperController
        import omni.usd
        import omni.replicator.core as rep
        from isaacsim.core.api import World
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.api.objects import DynamicCuboid, DynamicSphere
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
        from isaacsim.sensors.camera import Camera

        # ── Scene ─────────────────────────────────────────────────────────────
        omni.usd.get_context().open_stage(self.scene_usd)

        # Deactivate the ROS2 ActionGraph — isaacsim.ros2.bridge is not loaded
        # during headless collection and the graph nodes would error on step.
        stage = omni.usd.get_context().get_stage()
        _ag = stage.GetPrimAtPath("/World/ActionGraph")
        if _ag.IsValid():
            _ag.SetActive(False)
            print("[COLLECT] Disabled /World/ActionGraph (ROS2 bridge not loaded)", flush=True)

        # ── Surface gripper Phase 1: prims before world.reset() ───────────────
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.robot.surface_gripper")
        gripper = SurfaceGripperController()
        gripper.create_prims(stage)
        _prev_gripper = 0.0

        world = World(physics_dt=1/60, rendering_dt=1/60, stage_units_in_meters=1.0)
        robot = world.scene.add(Robot(prim_path=self.ROBOT_PRIM, name="robot"))

        wrist_cam = Camera(
            prim_path  = self.WRIST_CAM_PRIM,
            resolution = (self.image_size, self.image_size),
        )
        world.scene.add(wrist_cam)

        # Pre-allocate both pick objects at fixed size. Only one is active per
        # episode; the other is parked underground.
        s = self.task.obj_half_size
        cube = world.scene.add(DynamicCuboid(
            prim_path = self.CUBE_PRIM,
            name      = "pick_cube",
            position  = self.PARK_POS.copy(),
            scale     = np.array([s * 2] * 3),
        ))
        sphere = world.scene.add(DynamicSphere(
            prim_path = self.SPHERE_PRIM,
            name      = "pick_sphere",
            position  = self.PARK_POS.copy(),
            radius    = s,
        ))

        print("[COLLECT] world.reset() ...", flush=True)
        world.reset()

        # ── Surface gripper Phase 2: acquire interface after world.reset() ────
        gripper.acquire_interface()

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
        # Randomise lighting and object colour each frame. Both pick object
        # prims are covered — only the active (visible) one matters.
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
            with rep.get.prims(path_pattern=self.CUBE_PRIM):
                rep.randomizer.color(
                    colors=rep.distribution.uniform((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))
                )
            with rep.get.prims(path_pattern=self.SPHERE_PRIM):
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
            waypoints, instruction, pick_pos, _, obj_type, obj_half_size = self.task.sample(self.rng)

            # Activate chosen object at pick position; park the other underground.
            if obj_type == 'cube':
                cube.set_world_pose(position=pick_pos)
                sphere.set_world_pose(position=self.PARK_POS)
            else:
                sphere.set_world_pose(position=pick_pos)
                cube.set_world_pose(position=self.PARK_POS)

            # New lighting + object appearance for this episode.
            rep.orchestrator.step(rt_subframes=4, pause_timeline=False)

            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=self.HOME_POS)
            )
            for _ in range(self.HOME_SETTLE):
                world.step(render=False)

            buf, prev_pos, prev_rot, prev_rgb, ok = EpisodeBuffer(), None, None, None, True

            for wp in waypoints:
                # Send gripper command only on state transitions
                if wp.gripper != _prev_gripper:
                    if wp.gripper > 0.5:
                        gripper.close()
                    else:
                        gripper.open()
                    _prev_gripper = wp.gripper

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
                    # Sync kinematic gripper body to link_6 and keep close-raycast alive.
                    # Both must happen every step — sync_to_link6 keeps scan origins
                    # current; close_gripper must be re-issued each step because the
                    # surface-gripper extension does not self-maintain Closing state.
                    _pre_pos, _pre_rot = ik.compute_forward_kinematics(
                        self.EE_FRAME, robot.get_joint_positions()
                    )
                    _l6_pos  = robot_world_pos + robot_R @ _pre_pos
                    _l6_xyzw = Rotation.from_matrix(robot_R @ _pre_rot).as_quat()
                    gripper.sync_to_link6(
                        world_pos       = _l6_pos,
                        world_quat_wxyz = np.array([_l6_xyzw[3], _l6_xyzw[0],
                                                     _l6_xyzw[1], _l6_xyzw[2]]),
                    )
                    if wp.gripper > 0.5:
                        gripper.close()

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
                      f"steps={len(buf)}  obj={obj_type}  '{instruction}'", flush=True)
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
    ap.add_argument("--pick-x",       type=float, nargs=2, default=[0.59, 1.26], metavar=("MIN", "MAX"))
    ap.add_argument("--pick-y",       type=float, nargs=2, default=[-0.3, 0.8],  metavar=("MIN", "MAX"))
    ap.add_argument("--surface-z",     type=float, default=0.43,
                    help="Pallet surface Z in world frame (object bottom face)")
    ap.add_argument("--place-x",       type=float, nargs=2, default=[-0.95, 0.85],  metavar=("MIN", "MAX"))
    ap.add_argument("--place-y",       type=float, nargs=2, default=[-1.55, -1.0],  metavar=("MIN", "MAX"))
    ap.add_argument("--place-z",       type=float, default=0.85,
                    help="Conveyor surface Z in world frame (object bottom face)")
    ap.add_argument("--max-reach-xy",  type=float, default=1.65)
    ap.add_argument("--eef-z-offset",  type=float, default=0.185,
                    help="link_6 → cup-tip distance along gripper -Z (tune via red markers in sim)")
    ap.add_argument("--obj-half-size",  type=float, default=0.05,
                    help="Object half-size in metres (cube half-edge / sphere radius)")
    return ap.parse_args()


def main():
    import traceback as _tb
    args = parse_args()
    from isaacsim import SimulationApp
    simulation_app = SimulationApp({"headless": False})
    _failed = False
    try:
        DataCollector(
            raw_dir    = args.output_dir,
            task       = PickAndPlaceTask(
                pick_x       = tuple(args.pick_x),
                pick_y       = tuple(args.pick_y),
                surface_z    = args.surface_z,
                place_x      = tuple(args.place_x),
                place_y      = tuple(args.place_y),
                place_z      = args.place_z,
                max_reach_xy = args.max_reach_xy,
                eef_z_offset  = args.eef_z_offset,
                obj_half_size = args.obj_half_size,
            ),
            n_episodes = args.n_episodes,
            image_size = args.image_size,
            seed       = args.seed,
            scene_usd  = args.scene_usd,
        ).run()
    except Exception:
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
