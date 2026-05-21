# =============================================================================
# collect.py — Episode collection in Isaac Sim 5.1.0 with Replicator.
#
# Three object types are supported:
#   cube     — 10×10×10 cm DynamicCuboid, straight-down EEF.
#   cylinder — 10 cm dia × 8 cm DynamicCylinder, straight-down EEF.
#   pyramid  — true apex pyramid (rectangular base L×W, no flat top).
#              Dimensions are sampled fresh each episode and the USD mesh is
#              rebuilt between episodes using force_load_physics_from_usd()
#              so PhysX picks up the new geometry without a full world.reset().
#              EEF tilts to the most-horizontal graspable face normal.
#
# Usage:
#   bash launch/pipeline.sh --output-dir data/exp_01 --n-episodes 500
# =============================================================================

import argparse
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from task import PickAndPlaceTask, Waypoint, _OBJ_PARAMS, WAYPOINT_NAMES


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

    CUBE_PRIM     = "/World/pick_cube"
    CYLINDER_PRIM = "/World/pick_cylinder"
    PYRAMID_PRIM  = "/World/pick_pyramid"

    PARK_POSITIONS = {
        'cube':     np.array([0.0, -0.4, -4.945]),
        'cylinder': np.array([0.0,  0.0, -4.955]),
        'pyramid':  np.array([0.0,  0.4, -4.995]),
    }

    HOME_POS    = [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]
    HOME_SETTLE = 60

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
        dry_run:             bool = False,
        forced_obj_sequence: list = None,
        time_scale:          int  = 1,
        gripper_wait_steps:  int  = 0,
    ):
        root = Path(__file__).parent.parent.parent
        self.raw_dir             = Path(raw_dir)
        self.task                = task
        self.n_episodes          = n_episodes
        self.image_size          = image_size
        self.rng                 = np.random.default_rng(seed)
        self.scene_usd           = scene_usd or str(root / "scenes/usd/sim2.usd")
        self.urdf_path           = urdf_path or str(root / "scenes/h2017/urdf/h2017.urdf")
        self.lula_desc           = lula_desc or str(root / "scenes/h2017/urdf/h2017_lula.yaml")
        self.dry_run             = dry_run
        self.forced_obj_sequence = forced_obj_sequence
        self.time_scale          = time_scale
        self.gripper_wait_steps  = gripper_wait_steps
        self.raw_dir.mkdir(parents=True, exist_ok=True)

    # ── run ───────────────────────────────────────────────────────────────────

    def run(self):
        import carb
        from gripper import SurfaceGripperController
        import omni.usd
        import omni.replicator.core as rep
        from isaacsim.core.api import World
        from isaacsim.core.api.robots import Robot
        from isaacsim.core.api.objects import DynamicCuboid, DynamicCylinder
        from isaacsim.core.utils.types import ArticulationAction
        from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
        from isaacsim.sensors.camera import Camera

        # ── Scene ─────────────────────────────────────────────────────────────
        omni.usd.get_context().open_stage(self.scene_usd)
        stage = omni.usd.get_context().get_stage()

        _ag = stage.GetPrimAtPath("/World/ActionGraph")
        if _ag.IsValid():
            _ag.SetActive(False)
            print("[COLLECT] Disabled /World/ActionGraph", flush=True)

        # ── Surface gripper (Phase 1: before world.reset) ─────────────────────
        from isaacsim.core.utils.extensions import enable_extension
        enable_extension("isaacsim.robot.surface_gripper")
        gripper = SurfaceGripperController(verbose=True)
        gripper.create_prims(stage)
        _prev_gripper  = 0.0
        _close_pending = False   # deferred gripper close, fires once arm settles

        world = World(physics_dt=1/60, rendering_dt=1/60, stage_units_in_meters=1.0)
        robot = world.scene.add(Robot(prim_path=self.ROBOT_PRIM, name="robot"))

        wrist_cam = Camera(
            prim_path  = self.WRIST_CAM_PRIM,
            resolution = (self.image_size, self.image_size),
        )
        world.scene.add(wrist_cam)

        # ── Pre-allocate cube and cylinder ────────────────────────────────────
        cube_h   = _OBJ_PARAMS['cube']['height']
        cyl_h    = _OBJ_PARAMS['cylinder']['height']

        cube = world.scene.add(DynamicCuboid(
            prim_path = self.CUBE_PRIM,
            name      = "pick_cube",
            position  = self.PARK_POSITIONS['cube'].copy(),
            scale     = np.array([cube_h] * 3),
            mass      = 0.20,
        ))
        cylinder = world.scene.add(DynamicCylinder(
            prim_path = self.CYLINDER_PRIM,
            name      = "pick_cylinder",
            position  = self.PARK_POSITIONS['cylinder'].copy(),
            radius    = 0.05,
            height    = cyl_h,
            mass      = 0.20,
        ))

        # Pyramid is NOT pre-allocated; it is built fresh each episode between
        # episodes using force_load_physics_from_usd() so its L×W×H can vary.

        print("[COLLECT] world.reset() ...", flush=True)
        world.reset()

        # ── Surface gripper (Phase 2: after world.reset) ──────────────────────
        gripper.acquire_interface()

        # ── Disable collision on the SMC gripper mesh attached to link_6 ──────
        # The SMC_gripper mesh under link_6 has active collision geometry.
        # Its collision shape completely surrounds the cup tips, so every
        # surface-gripper raycast hits link_6 at distance=0 and never reaches
        # the pick cube.  Disabling collision on those meshes lets the scan
        # pass through to the object below.
        from pxr import Usd as _Usd
        _smc_root = stage.GetPrimAtPath("/World/h2017/link_6/SMC_gripper")
        _n_disabled = 0
        if _smc_root.IsValid():
            for _p in _Usd.PrimRange(_smc_root):
                _col = _p.GetAttribute("physics:collisionEnabled")
                if _col.IsValid() and _col.Get():
                    _col.Set(False)
                    _n_disabled += 1
        print(f"[COLLECT] SMC_gripper collision disabled on {_n_disabled} prim(s)", flush=True)

        # ── Park the scene's Pickables/Cube so it never enters the scan path ──
        # sim2.usd contains /World/Pickables/Cube as a dynamic rigid body.  It
        # sits at some fixed scene position which may overlap with the pick zone.
        # Moving it underground prevents the surface gripper from accidentally
        # attaching to it instead of the episode's /World/pick_cube.
        from isaacsim.core.prims import SingleRigidPrim as _SRP
        _scene_cube_prim = stage.GetPrimAtPath("/World/Pickables/Cube")
        if _scene_cube_prim.IsValid():
            _sc = _SRP(prim_path="/World/Pickables/Cube", name="scene_cube_park")
            _sc_pos, _ = _sc.get_world_pose()
            print(f"[COLLECT] /World/Pickables/Cube found at {_sc_pos.round(3)} — parking underground", flush=True)
            _sc.set_world_pose(position=np.array([0.0, 0.8, -5.0]))
            _sc.set_linear_velocity(np.zeros(3))
            _sc.set_angular_velocity(np.zeros(3))
            world.step(render=False)
        else:
            print("[COLLECT] /World/Pickables/Cube not in scene — nothing to park", flush=True)

        print("[COLLECT] wrist_cam.initialize() ...", flush=True)
        wrist_cam.initialize()

        n_dof = robot.num_dof
        robot.get_articulation_controller().set_gains(
            kps=np.full(n_dof, 1e6), kds=np.full(n_dof, 1e5),
        )
        print(f"[COLLECT] robot gains set (n_dof={n_dof})", flush=True)

        self._generate_lula(self.urdf_path, self.lula_desc)

        # Place-zone IK solver seeded toward J1≈-130°, J2≈+60°, J3≈+100° so LULA
        # starts in the elbow-up branch when reaching behind the robot.
        # J1=-130° faces the place zone; J2/J3 positive keeps the arm above the
        # shoulder plane so it never needs to sweep through the vertical singularity.
        place_lula = str(Path(self.lula_desc).with_suffix('')) + '_place.yaml'
        self._generate_lula(
            self.urdf_path, place_lula,
            default_q=list(np.radians([-130.0, 60.0, 100.0, 0.0, 20.0, 0.0])),
        )

        ik = LulaKinematicsSolver(
            robot_description_path=self.lula_desc,
            urdf_path=self.urdf_path,
        )
        ik_place = LulaKinematicsSolver(
            robot_description_path=place_lula,
            urdf_path=self.urdf_path,
        )
        print("[COLLECT] IK solvers ready (pick + place)", flush=True)

        robot_world_pos, robot_world_quat = robot.get_world_pose()
        rq = robot_world_quat
        robot_R = Rotation.from_quat([rq[1], rq[2], rq[3], rq[0]]).as_matrix()

        # ── Replicator (cube + cylinder; pyramid colour set manually) ─────────
        print("[COLLECT] Replicator setup ...", flush=True)
        with rep.trigger.on_frame():
            with rep.get.light():
                rep.modify.attribute("inputs:intensity",
                                     rep.distribution.uniform(300, 4000))
                rep.modify.attribute("inputs:colorTemperature",
                                     rep.distribution.uniform(3200, 7500))
            for pp in (self.CUBE_PRIM, self.CYLINDER_PRIM):
                with rep.get.prims(path_pattern=pp):
                    rep.randomizer.color(
                        colors=rep.distribution.uniform((0,0,0), (1,1,1))
                    )
        print("[COLLECT] Replicator ready", flush=True)

        # obj_map is updated dynamically when a pyramid episode is encountered.
        obj_map: dict = {'cube': cube, 'cylinder': cylinder}
        pyramid = None   # SingleRigidPrim, created on first pyramid episode

        # ── Episode loop ──────────────────────────────────────────────────────
        ep_idx = len(list(self.raw_dir.glob("*.npz")))
        print(f"[COLLECT] target={self.n_episodes}  resuming from ep {ep_idx}", flush=True)
        consecutive_failures = 0
        MAX_FAILURES = 20

        while ep_idx < self.n_episodes:
            force_type = (
                self.forced_obj_sequence[ep_idx % len(self.forced_obj_sequence)]
                if self.forced_obj_sequence else None
            )
            # Sample yaw before task.sample() so the IK can align J6 to it.
            # Pyramid ignores pick_yaw (it uses face-normal orientation instead).
            _ep_yaw = float(self.rng.uniform(0, 2 * np.pi))
            waypoints, instruction, pick_pos, _, obj_type, meta = \
                self.task.sample(self.rng, ik, robot_world_pos, robot_R, self.EE_FRAME,
                                 force_obj_type=force_type, ik_place=ik_place,
                                 pick_yaw=_ep_yaw)

            if waypoints is None:
                print(f"[COLLECT] ep {ep_idx}: IK pre-solve failed obj={obj_type} — resampling",
                      flush=True)
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    raise RuntimeError(
                        f"{MAX_FAILURES} consecutive failures. "
                        "Check IK workspace bounds and pyramid tilt constraints."
                    )
                continue

            if self.time_scale != 1:
                waypoints = [Waypoint(wp.position, wp.gripper, wp.n_steps * self.time_scale)
                             for wp in waypoints]

            # ── Dry-run episode header ─────────────────────────────────────
            if self.dry_run:
                dims = (f"  L={meta['L']:.2f} W={meta['W']:.2f} H={meta['H']:.2f}"
                        if meta else "")
                if obj_type == 'pyramid':
                    n_ow = meta['n_outward']
                    grab_angle = np.degrees(np.arccos(np.clip(float(n_ow[2]), -1.0, 1.0)))
                    surface_info = f"  grab_angle={grab_angle:.1f}° from vertical"
                else:
                    surface_info = "  grab_angle=0.0° (flat surface)"
                print(f"\n[DRY-RUN] ── Episode {ep_idx + 1}/{self.n_episodes} "
                      f"obj={obj_type}{dims}{surface_info}", flush=True)

            # ── Pyramid: rebuild USD prim between episodes ─────────────────
            if obj_type == 'pyramid':
                pyramid = self._rebuild_pyramid(
                    stage, world, meta, pyramid, obj_map
                )

            # ── Robot home FIRST so idle objects settle on the platform ───
            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=self.HOME_POS)
            )
            for _ in range(self.HOME_SETTLE):
                world.step(render=False)

            # ── Teleport active object to pallet; park the rest ────────────
            # Cube/cylinder yaw was already sampled above and baked into the IK
            # so J6 tracks the cube orientation.  Pyramid uses identity (face normal
            # orientation is handled separately in _build_pyramid_waypoints).
            _yaw_for_pose = _ep_yaw if obj_type != 'pyramid' else 0.0
            _cy, _sy = np.cos(_yaw_for_pose / 2), np.sin(_yaw_for_pose / 2)
            _pick_ori = np.array([_cy, 0.0, 0.0, _sy])  # wxyz, rotation around Z

            for name, obj in obj_map.items():
                if name == obj_type:
                    obj.set_world_pose(position=pick_pos, orientation=_pick_ori)
                else:
                    obj.set_world_pose(position=self.PARK_POSITIONS.get(
                        name, np.array([0.0, 0.0, -5.0])
                    ).copy())
                obj.set_linear_velocity(np.zeros(3))
                obj.set_angular_velocity(np.zeros(3))

            # ── Zero robot joint velocities to prevent cross-episode leakage ─
            robot.get_articulation_controller().apply_action(
                ArticulationAction(
                    joint_positions=self.HOME_POS,
                    joint_velocities=np.zeros(n_dof),
                )
            )
            for _ in range(5):
                world.step(render=False)

            # ── Colour-randomise pyramid manually (Replicator doesn't track
            #    dynamically recreated prims) ─────────────────────────────
            if obj_type == 'pyramid':
                self._randomise_pyramid_colour(stage)

            # ── Replicator step: lighting + cube/cylinder colours ──────────
            rep.orchestrator.step(rt_subframes=4, pause_timeline=False)

            if self.dry_run:
                _obj = obj_map.get(obj_type)
                if _obj is not None:
                    _obj_pos, _obj_ori_wxyz = _obj.get_world_pose()
                    _obj_euler = np.degrees(
                        Rotation.from_quat([
                            _obj_ori_wxyz[1], _obj_ori_wxyz[2],
                            _obj_ori_wxyz[3], _obj_ori_wxyz[0],
                        ]).as_euler('xyz')
                    )
                    print(
                        f"[DRY-RUN]   {obj_type} pos = "
                        f"({_obj_pos[0]:.4f}, {_obj_pos[1]:.4f}, {_obj_pos[2]:.4f}) m\n"
                        f"[DRY-RUN]   {obj_type} ori = "
                        f"({_obj_euler[0]:.1f}°, {_obj_euler[1]:.1f}°, {_obj_euler[2]:.1f}°) XYZ Euler",
                        flush=True,
                    )

            buf = EpisodeBuffer()
            prev_pos = prev_rot = prev_rgb = None

            n_wps = len(waypoints)
            _gripper_scan_done = False
            for wp_idx, wp in enumerate(waypoints, 1):
                _gripper_close_this_wp = False
                if wp.gripper != _prev_gripper:
                    if wp.gripper > 0.5:
                        _close_pending = True   # arm still approaching — close fires in step loop
                        _gripper_close_this_wp = True
                    else:
                        gripper.open()
                        # 20-frame physics dwell lets suction release cleanly
                        # before the arm begins its retract motion.
                        for _ in range(20):
                            world.step(render=True)
                    _prev_gripper = wp.gripper

                if self.dry_run:
                    name = WAYPOINT_NAMES[wp_idx - 1] if wp_idx <= len(WAYPOINT_NAMES) \
                           else f"wp{wp_idx}"
                    gripper_str = "closed" if wp.gripper > 0.5 else "open"
                    joints_deg  = np.degrees(wp.position).round(1)
                    print(f"  → [{wp_idx}/{n_wps}] {name:<26}  "
                          f"gripper={gripper_str:<6}  "
                          f"joints(°)={joints_deg}", flush=True)

                start_joints = robot.get_joint_positions().copy()
                next_wp      = waypoints[wp_idx] if wp_idx < n_wps else None
                _blend_joints = None
                for s in range(wp.n_steps):
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
                    # Fire close once the arm has settled (~70% into the dwell).
                    # Firing at the waypoint boundary would scan when the arm is
                    # still far above the cube (still descending from pre-pick),
                    # which puts the cup tips outside the 5 cm scan range.
                    # Wait until 50% of dwell before starting to scan (arm near target).
                    # Then call close() every step — the surface gripper extension
                    # requires per-step calls to keep the raycast alive; it does not
                    # self-maintain Closing state between physics steps.
                    if _close_pending and s >= int(wp.n_steps * 0.5):
                        _close_pending = False
                        if self.dry_run:
                            # Print actual scan geometry at the moment scanning starts.
                            _d_pos, _d_rot = ik.compute_forward_kinematics(
                                self.EE_FRAME, robot.get_joint_positions()
                            )
                            _l6w   = robot_world_pos + robot_R @ _d_pos
                            _Rw    = robot_R @ _d_rot
                            # Z=0.215 matches _CUP_TIPS in gripper.py
                            _origA = _l6w + _Rw @ np.array([-0.022, -0.019, 0.215])
                            _origB = _l6w + _Rw @ np.array([ 0.020,  0.0235, 0.215])
                            _dir   = _Rw @ np.array([0.0, 0.0, 1.0])
                            _tcp   = _l6w + _Rw @ np.array([0.0, 0.0, 0.215])
                            print(f"[SCAN-DIAG] step={s}  link6_world  = {_l6w.round(4)}", flush=True)
                            print(f"[SCAN-DIAG] cup_A_origin = {_origA.round(4)}  scan_bottom={(_origA[2]-0.08):.4f}", flush=True)
                            print(f"[SCAN-DIAG] cup_B_origin = {_origB.round(4)}  scan_bottom={(_origB[2]-0.08):.4f}", flush=True)
                            print(f"[SCAN-DIAG] TCP_tip      = {_tcp.round(4)}", flush=True)
                            print(f"[SCAN-DIAG] scan_dir     = {_dir.round(4)}", flush=True)
                    # Do NOT call close() here — the arm is still converging toward
                    # wp.position during the dwell.  Firing the surface gripper while
                    # the body is moving causes repeated attachment failures that put
                    # the extension in a non-responsive Open state.  close() is called
                    # only after the arm has fully settled, in the GRIPPER-WAIT block.

                    rgba    = wrist_cam.get_rgba()
                    cur_rgb = (
                        (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
                        if rgba is not None and rgba.ndim >= 3 else None
                    )

                    # Dwell waypoint: same IK target as previous (e.g. pick
                    # contact → pick dwell).  Use a loose tolerance (0.05 rad
                    # ≈ 3°) so that even if the arm hasn't fully settled during
                    # the previous waypoint's hold phase it still holds from
                    # step 0 instead of re-ramping (which causes visible EEF
                    # tilt mid-ramp for certain kinematic configurations).
                    is_dwell = np.allclose(wp.position, start_joints, atol=0.05)

                    # Reserve the last _HOLD_STEPS of every waypoint as a hold
                    # phase so the PD controller converges before the next
                    # waypoint starts.  Without this the arm is still in motion
                    # at waypoint boundaries and never reaches the IK targets.
                    _HOLD_STEPS = 20
                    _ramp_steps = max(1, wp.n_steps - _HOLD_STEPS)

                    if is_dwell:
                        q_joint_cmd = wp.position
                    elif wp.blend > 0 and next_wp is not None and s >= wp.n_steps - wp.blend:
                        if _blend_joints is None:
                            _blend_joints = robot.get_joint_positions().copy()
                        b           = (s - (wp.n_steps - wp.blend) + 1) / wp.blend
                        diff        = next_wp.position - _blend_joints
                        diff        = (diff + np.pi) % (2 * np.pi) - np.pi
                        q_joint_cmd = _blend_joints + b * diff
                    elif s < _ramp_steps:
                        alpha       = (s + 1) / _ramp_steps
                        diff        = wp.position - start_joints
                        diff        = (diff + np.pi) % (2 * np.pi) - np.pi
                        q_joint_cmd = start_joints + alpha * diff
                    else:
                        q_joint_cmd = wp.position

                    robot.get_articulation_controller().apply_action(
                        ArticulationAction(joint_positions=q_joint_cmd)
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

                # After the arm fully arrives at the pick dwell, hold position and
                # retry gripper.close() for gripper_wait_steps steps before retract.
                # Mirrors GRIP_WAIT_SEC in the gripper test.
                if _gripper_close_this_wp and self.gripper_wait_steps > 0:
                    # Diagnostic: show TCP tip world position vs object center so we
                    # can verify the scan origin is above (not inside) the object.
                    _w_pos, _w_rot = ik.compute_forward_kinematics(
                        self.EE_FRAME, robot.get_joint_positions()
                    )
                    _w_l6   = robot_world_pos + robot_R @ _w_pos
                    _w_Rw   = robot_R @ _w_rot
                    _w_tcp  = _w_l6 + _w_Rw @ np.array([0.0, 0.0, 0.215])
                    _w_obj  = obj_map.get(obj_type)
                    _w_obj_z = _w_obj.get_world_pose()[0][2] if _w_obj is not None else float('nan')
                    print(f"[GRIPPER-WAIT] TCP_tip   = {_w_tcp.round(4)}", flush=True)
                    print(f"[GRIPPER-WAIT] obj_ctr_z = {_w_obj_z:.4f}  "
                          f"TCP_z - obj_z = {(_w_tcp[2] - _w_obj_z):.4f} m", flush=True)
                    print(f"[GRIPPER-WAIT] status before wait = {gripper.status()}", flush=True)
                    # Do NOT call open() here — open_gripper() resets the D6 joint
                    # body1 reference inside the extension, which makes close_gripper()
                    # silently no-op on every subsequent call.  The gripper is already
                    # in Open state (close() was never called during the step loop).
                    # Phase 1: arm settle — hold at wp.position, NO close() calls.
                    # The PD controller needs ~30 steps to finish converging after
                    # the main loop ends.  Calling close() before the arm is truly
                    # stationary floods the extension with failed attachment attempts
                    # and puts it in a non-responsive Open state.
                    _settle_steps = min(30, self.gripper_wait_steps)
                    _grip_steps   = self.gripper_wait_steps - _settle_steps
                    print(f"[GRIPPER-WAIT] Phase 1: settling {_settle_steps} steps "
                          f"(arm converging, no close)...", flush=True)
                    for _ in range(_settle_steps):
                        _gw_pos, _gw_rot = ik.compute_forward_kinematics(
                            self.EE_FRAME, robot.get_joint_positions()
                        )
                        _gw_l6   = robot_world_pos + robot_R @ _gw_pos
                        _gw_xyzw = Rotation.from_matrix(robot_R @ _gw_rot).as_quat()
                        gripper.sync_to_link6(
                            world_pos       = _gw_l6,
                            world_quat_wxyz = np.array([_gw_xyzw[3], _gw_xyzw[0],
                                                        _gw_xyzw[1], _gw_xyzw[2]]),
                        )
                        robot.get_articulation_controller().apply_action(
                            ArticulationAction(joint_positions=wp.position)
                        )
                        world.step(render=True)

                    # Phase 2: grip — sync_to_link6() must be called every step
                    # so the kinematic body registers as "moved" to PhysX.
                    # A frozen kinematic body is treated as static and the
                    # surface gripper extension skips its raycast entirely.
                    # With the 3 cm pick clearance in task.py the arm is now
                    # above the cube and won't press it, so continuous sync
                    # is safe again.
                    # ── Raw PhysX raycast diagnostic ─────────────────────────
                    # Bypass the surface gripper extension completely and ask
                    # PhysX directly what (if anything) is below the cup tip.
                    try:
                        from omni.physx import get_physx_scene_query_interface as _gpsqi
                        _rc_pos, _rc_rot = ik.compute_forward_kinematics(
                            self.EE_FRAME, robot.get_joint_positions()
                        )
                        _rc_tcp = robot_world_pos + robot_R @ (
                            _rc_pos + _rc_rot @ np.array([0.0, 0.0, 0.215])
                        )
                        _rc_hit = _gpsqi().raycast_closest(
                            carb.Float3(_rc_tcp[0], _rc_tcp[1], _rc_tcp[2]),
                            carb.Float3(0.0, 0.0, -1.0),
                            0.15,
                        )
                        print(f"[RAYCAST-DIAG] origin={_rc_tcp.round(4)}", flush=True)
                        print(f"[RAYCAST-DIAG] hit={_rc_hit}", flush=True)
                    except Exception as _e:
                        print(f"[RAYCAST-DIAG] error: {_e}", flush=True)
                    # ─────────────────────────────────────────────────────────

                    # Print actual scan direction so we can confirm the EEF
                    # is pointing straight down (not tilted) before scanning.
                    _sd_pos, _sd_rot = ik.compute_forward_kinematics(
                        self.EE_FRAME, robot.get_joint_positions()
                    )
                    _sd_dir = (robot_R @ _sd_rot) @ np.array([0.0, 0.0, 1.0])
                    print(f"[GRIPPER-WAIT] scan_dir at Phase 2 = {_sd_dir.round(4)}", flush=True)

                    print(f"[GRIPPER-WAIT] Phase 2: gripping {_grip_steps} steps "
                          f"(sync active, close active)...", flush=True)
                    for _gw in range(_grip_steps):
                        _gw_pos, _gw_rot = ik.compute_forward_kinematics(
                            self.EE_FRAME, robot.get_joint_positions()
                        )
                        _gw_l6   = robot_world_pos + robot_R @ _gw_pos
                        _gw_xyzw = Rotation.from_matrix(robot_R @ _gw_rot).as_quat()
                        gripper.sync_to_link6(
                            world_pos       = _gw_l6,
                            world_quat_wxyz = np.array([_gw_xyzw[3], _gw_xyzw[0],
                                                        _gw_xyzw[1], _gw_xyzw[2]]),
                        )
                        gripper.close()
                        robot.get_articulation_controller().apply_action(
                            ArticulationAction(joint_positions=wp.position)
                        )
                        world.step(render=True)
                        _gw_status = gripper.status()
                        if _gw % 30 == 0:
                            print(f"[GRIPPER-WAIT] step {_gw:3d}  status={_gw_status}", flush=True)
                        if _gw_status == "closed":
                            print(f"[GRIPPER-WAIT] latched at step {_gw}", flush=True)
                            break

                if self.dry_run:
                    _arr_pos, _arr_rot = ik.compute_forward_kinematics(
                        self.EE_FRAME, robot.get_joint_positions()
                    )
                    # Cup tips are 0.215 m along link_6 +Z — transform into world space.
                    _tip_world = robot_world_pos + robot_R @ (
                        _arr_pos + _arr_rot @ np.array([0.0, 0.0, 0.215])
                    )
                    print(f"    ✓ arrived  TCP_tip=({_tip_world[0]:.3f}, "
                          f"{_tip_world[1]:.3f}, {_tip_world[2]:.3f}) m", flush=True)
                    if wp.gripper > 0.5 and not _gripper_scan_done:
                        import omni.usd as _ousd
                        _stage = _ousd.get_context().get_stage()
                        _jp    = _stage.GetPrimAtPath("/World/SurfaceGripperJoints/D6Joint_00")
                        _b1    = _jp.GetRelationship("physics:body1").GetTargets() if _jp.IsValid() else []
                        _hit   = str(_b1[0]) if _b1 and str(_b1[0]) != "/World/h2017/link_6" else None
                        print(f"    gripper scan: {_hit if _hit else 'No object within range'}", flush=True)
                        _gripper_scan_done = True

            # Settle at home — PD controller needs extra frames to physically
            # reach the commanded position after the last Lerp step.
            for _ in range(40):
                robot.get_articulation_controller().apply_action(
                    ArticulationAction(joint_positions=self.HOME_POS)
                )
                world.step(render=True)

            if len(buf) > 0:
                ep_idx += 1
                consecutive_failures = 0
                dims = (f"L={meta['L']:.2f} W={meta['W']:.2f} H={meta['H']:.2f}"
                        if meta else "")
                tag = "[DRY-RUN]" if self.dry_run else "[COLLECT]"
                print(f"{tag} {ep_idx}/{self.n_episodes}  "
                      f"steps={len(buf)}  obj={obj_type}  {dims}  '{instruction}'",
                      flush=True)
                if not self.dry_run:
                    buf.save(self.raw_dir / f"episode_{ep_idx - 1:06d}.npz")
            else:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    raise RuntimeError(
                        f"{MAX_FAILURES} consecutive failures. "
                        "Check IK workspace bounds and pyramid tilt constraints."
                    )

    # ── pyramid helpers ────────────────────────────────────────────────────────

    def _rebuild_pyramid(self, stage, world, meta: dict, pyramid, obj_map: dict):
        """Delete the existing pyramid prim (if any) and build a fresh one with
        the sampled dimensions, then reload it into the running PhysX scene.

        In Isaac Sim 5.1.0, force_load_physics_from_usd() recompiles only the
        changed prim without requiring a full world.reset().
        """
        from isaacsim.core.prims import SingleRigidPrim
        from omni.physx import get_physx_interface

        L, W, H = meta['L'], meta['W'], meta['H']

        # Remove old prim from Isaac scene tracking and USD stage.
        if pyramid is not None:
            try:
                world.scene.remove_object("pick_pyramid")
            except Exception:
                pass
        if stage.GetPrimAtPath(self.PYRAMID_PRIM).IsValid():
            stage.RemovePrim(self.PYRAMID_PRIM)

        # Build the new apex pyramid mesh with the sampled dimensions.
        self._build_pyramid_usd(stage, self.PYRAMID_PRIM, L, W, H, mass=0.25)

        # Tell PhysX to reparse the USD stage and compile the new rigid body.
        # This is cheaper than world.reset() and sufficient for a single new prim.
        get_physx_interface().force_load_physics_from_usd()

        # Wrap in SingleRigidPrim and re-register with the scene so the PhysX
        # tensor simulation view is rebuilt around the new rigid body before
        # any velocity queries are issued.
        new_pyramid = SingleRigidPrim(prim_path=self.PYRAMID_PRIM, name="pick_pyramid")
        world.scene.add(new_pyramid)
        obj_map['pyramid'] = new_pyramid

        # Reinitialise physics so Isaac Sim rebuilds its tensor view to include
        # the new rigid body, then flush two frames for the memory swap to settle.
        world.initialize_physics()
        for _ in range(2):
            world.step(render=False)

        return new_pyramid

    def _randomise_pyramid_colour(self, stage):
        """Set a random display colour on the pyramid mesh for visual diversity."""
        from pxr import Gf, UsdGeom
        prim = stage.GetPrimAtPath(self.PYRAMID_PRIM)
        if not prim.IsValid():
            return
        color = self.rng.uniform(0.0, 1.0, size=3)
        UsdGeom.Mesh(prim).GetDisplayColorAttr().Set(
            [Gf.Vec3f(float(color[0]), float(color[1]), float(color[2]))]
        )

    @staticmethod
    def _build_pyramid_usd(stage, prim_path: str,
                           L: float, W: float, H: float, mass: float):
        """True apex pyramid: rectangular base L×W at Z=0, apex at (0,0,H).

        Local origin = base centre. set_world_pose([px,py,pz]) places the base
        centre at [px,py,pz], so pz=surface_z puts the base on the pallet.

        Face winding (right-hand rule → outward normals):
          Base   0,3,2,1  →  normal -Z  (downward, resting on pallet)
          Front  0,1,4    →  normal ∝ (0,−2H,W)   (−Y side)
          Right  1,2,4    →  normal ∝ (+2H,0,L)   (+X side)
          Back   2,3,4    →  normal ∝ (0,+2H,W)   (+Y side)
          Left   3,0,4    →  normal ∝ (−2H,0,L)   (−X side)

        Z-component of ±X face normal: L/√(4H²+L²)
        Z-component of ±Y face normal: W/√(4H²+W²)
        These match _pyramid_best_face() in task.py exactly.
        """
        from pxr import Gf, UsdGeom, UsdPhysics

        mesh = UsdGeom.Mesh.Define(stage, prim_path)

        hl, hw = L / 2.0, W / 2.0
        verts = [
            Gf.Vec3f(-hl, -hw, 0.0),   # 0  base corner
            Gf.Vec3f( hl, -hw, 0.0),   # 1
            Gf.Vec3f( hl,  hw, 0.0),   # 2
            Gf.Vec3f(-hl,  hw, 0.0),   # 3
            Gf.Vec3f(0.0, 0.0,   H),   # 4  apex
        ]
        mesh.GetPointsAttr().Set(verts)
        mesh.GetFaceVertexCountsAttr().Set([4, 3, 3, 3, 3])
        mesh.GetFaceVertexIndicesAttr().Set([
            0, 3, 2, 1,   # base  (normal −Z)
            0, 1, 4,      # front (normal ∝ (0,−2H,W))
            1, 2, 4,      # right (normal ∝ (+2H,0,L))
            2, 3, 4,      # back  (normal ∝ (0,+2H,W))
            3, 0, 4,      # left  (normal ∝ (−2H,0,L))
        ])

        prim = stage.GetPrimAtPath(prim_path)
        UsdPhysics.RigidBodyAPI.Apply(prim)
        UsdPhysics.CollisionAPI.Apply(prim)
        UsdPhysics.MassAPI.Apply(prim)
        prim.GetAttribute("physics:mass").Set(mass)

        try:
            from pxr import PhysxSchema
            col = PhysxSchema.PhysxMeshCollisionAPI.Apply(prim)
            col.GetCollisionMeshAttr().Set("convexHull")
        except Exception:
            pass

    @staticmethod
    def _generate_lula(urdf_path: str, output_path: str, default_q=None):
        import xml.etree.ElementTree as ET
        root      = ET.parse(urdf_path).getroot()
        joints    = [j.get('name') for j in root.findall('joint')
                     if j.get('type') == 'revolute']
        parent    = root.find("joint/parent")
        root_link = parent.get('link') if parent is not None else 'base_link'
        n = len(joints)
        seed   = list(default_q) if default_q is not None else [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]
        home_q = (seed + [0.0] * n)[:n]
        Path(output_path).write_text(
            "api_version: 1.0\n\ncspace:\n"
            + "".join(f"    - {j}\n" for j in joints)
            + f"\nroot_link: {root_link}\n\n"
            f"default_q: [{', '.join(str(q) for q in home_q)}]\n\n"
            f"acceleration_limits: [{', '.join(['40.0']*n)}]\n"
            f"jerk_limits: [{', '.join(['500.0']*n)}]\n"
        )
        print(f"[COLLECT] Generated LULA description → {output_path}")


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    ap = argparse.ArgumentParser(
        description="Collect synthetic episodes in Isaac Sim 5.1.0."
    )
    ap.add_argument("--output-dir",   required=True)
    ap.add_argument("--n-episodes",   type=int,   default=200)
    ap.add_argument("--image-size",   type=int,   default=128)
    ap.add_argument("--seed",         type=int,   default=42)
    ap.add_argument("--scene-usd",    default=None)
    ap.add_argument("--pick-x",       type=float, nargs=2, default=[0.59, 1.26])
    ap.add_argument("--pick-y",       type=float, nargs=2, default=[-0.3, 0.8])
    ap.add_argument("--surface-z",    type=float, default=0.43)
    ap.add_argument("--place-x",      type=float, nargs=2, default=[-0.95, 0.85])
    ap.add_argument("--place-y",      type=float, nargs=2, default=[-1.55, -1.0])
    ap.add_argument("--place-z",      type=float, default=0.85)
    ap.add_argument("--max-reach-xy", type=float, default=1.65)
    ap.add_argument("--eef-z-offset", type=float, default=0.215)
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
                eef_z_offset = args.eef_z_offset,
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
