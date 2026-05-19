from pathlib import Path
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

# All imports must come after SimulationApp — Kit initialises the Python
# environment that makes pxr, omni.*, and isaacsim.* importable.
import time
import numpy as np
from scipy.spatial.transform import Rotation
from pxr import Usd
import omni.usd
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float64MultiArray, String, Empty
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.prims import SingleRigidPrim
from isaacsim.core.utils.extensions import enable_extension
from gripper import (SurfaceGripperController, _GRIPPER_PRIM_PATH, _CUP_TIPS,
                     _MAX_GRIP_DISTANCE)
from isaacsim.core.utils.stage import is_stage_loading
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from isaacsim.sensors.camera import Camera

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ROBOT_PRIM     = "/World/h2017"
EE_FRAME       = "link_6"
WRIST_CAM_PRIM = "/World/h2017/link_6/MechEye/MechEye/Camera"
URDF_PATH      = Path(__file__).parent.parent.parent / "scenes" / "h2017" / "urdf" / "h2017.urdf"
LULA_DESC      = Path(__file__).parent.parent.parent / "scenes" / "h2017" / "urdf" / "h2017_lula.yaml"
JOINT_NAMES    = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
HOME_POSITION  = [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]
MAX_POS        = 0.05   # metres per step — EEF delta safety clamp
MAX_ROT        = 0.05   # radians per step (~2.9° — tighter to prevent outlier spasms)
ROT_EMA_ALPHA  = 0.5    # EMA smoothing for rotation — blends new prediction with previous

# Pickable cube reset — fill in the prim path and world position of the cube
# so that /reset_scene teleports it back to the pick position.
# Find the prim path by clicking the cube in the Isaac Sim stage panel.
PICKUP_PRIM_PATH = "/World/Pickables/Cube"
PICKUP_POSITION  = [1.04598, 0.53071, 0.47985]
PICKUP_ORIENT_WXYZ = [1.0, 0.0, 0.0, 0.0]    # identity — no rotation

# -----------------------------------------------------------------------------
# Scene preprocessing — strip OmniGraph prims from the USD before loading.
# RemovePrim() is required: SetActive(False) does not stop the Kit OmniGraph
# runtime from executing the nodes, which crashes omni.graph.image.core while
# Fabric is still syncing robot mesh prims.
# -----------------------------------------------------------------------------
_RAW_SCENE   = Path(__file__).parent.parent.parent / "scenes" / "usd" / "sim2.usd"
_CLEAN_SCENE = _RAW_SCENE.parent / (_RAW_SCENE.stem + "_sim.usd")

if (not _CLEAN_SCENE.exists() or
        _RAW_SCENE.stat().st_mtime > _CLEAN_SCENE.stat().st_mtime):
    _stage = Usd.Stage.Open(str(_RAW_SCENE))
    _paths = [p.GetPath() for p in _stage.Traverse()
              if p.GetTypeName() in ("OmniGraph", "ComputeGraph")]
    for _path in _paths:
        _stage.RemovePrim(_path)
    _stage.GetRootLayer().Export(str(_CLEAN_SCENE))
    print(f"[SCENE] Built {_CLEAN_SCENE.name}: removed {len(_paths)} OmniGraph prim(s)")
else:
    print(f"[SCENE] Using cached {_CLEAN_SCENE.name}")

# -----------------------------------------------------------------------------
# Stage loading — poll without update() so omni.graph.image.core is never
# triggered while Fabric is still syncing prims from USD sublayers.
# -----------------------------------------------------------------------------
omni.usd.get_context().open_stage(str(_CLEAN_SCENE))

print("[SCENE] Waiting for stage to load...")
_t0 = time.monotonic()
while is_stage_loading():
    if time.monotonic() - _t0 > 120:
        raise RuntimeError("Stage loading timed out after 120 s")
    time.sleep(0.05)
print(f"[SCENE] Stage loaded in {time.monotonic() - _t0:.1f} s")

for _ in range(60):
    simulation_app.update()

# -----------------------------------------------------------------------------
# Surface gripper — Phase 1: create USD prims BEFORE world.reset()
# -----------------------------------------------------------------------------
enable_extension("isaacsim.robot.surface_gripper")
gripper = SurfaceGripperController()
gripper.create_prims(omni.usd.get_context().get_stage())

# Disable collision on the SMC gripper visual mesh BEFORE world.reset() so
# PhysX never registers it as a scene-query shape.  The mesh wraps the cup
# tips, causing the surface-gripper scan ray to self-terminate at distance=0
# before it can reach the cube.  Physics suction is driven by the extension,
# not by the mesh collision.
_smc_stage = omni.usd.get_context().get_stage()
_smc_n = 0
for _smc_prim in _smc_stage.Traverse():
    if str(_smc_prim.GetPath()).startswith("/World/h2017/link_6/SMC_gripper"):
        _smc_attr = _smc_prim.GetAttribute("physics:collisionEnabled")
        if _smc_attr.IsValid():
            _smc_attr.Set(False)
            _smc_n += 1
print(f"[SCENE] SMC gripper: disabled {_smc_n} collision prim(s) — scan path cleared")

# -----------------------------------------------------------------------------
# Physics world + robot
# -----------------------------------------------------------------------------
world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
robot = world.scene.add(Robot(prim_path=ROBOT_PRIM, name="robot"))
world.reset()
print(f"[ROBOT] {robot.num_dof} DOF")

# Surface gripper — Phase 2: bind to PhysX runtime AFTER world.reset()
gripper.acquire_interface()

# Pickable cube — used by /reset_scene to teleport the block back
_pickup_cube = None
_stage = omni.usd.get_context().get_stage()
if _stage.GetPrimAtPath(PICKUP_PRIM_PATH).IsValid():
    _pickup_cube = SingleRigidPrim(prim_path=PICKUP_PRIM_PATH, name="pickup_cube")
    print(f"[SCENE] Pickup cube registered: {PICKUP_PRIM_PATH}")
else:
    print(f"[SCENE] WARNING: PICKUP_PRIM_PATH='{PICKUP_PRIM_PATH}' not found — "
          "set it in sim_node.py to enable /reset_scene")

robot.get_articulation_controller().set_gains(
    kps=np.full(robot.num_dof, 1e6),
    kds=np.full(robot.num_dof, 1e5),
)
robot.get_articulation_controller().apply_action(
    ArticulationAction(joint_positions=HOME_POSITION)
)
for _ in range(60):
    world.step(render=False)

# -----------------------------------------------------------------------------
# Camera
# -----------------------------------------------------------------------------
if not omni.usd.get_context().get_stage().GetPrimAtPath(WRIST_CAM_PRIM).IsValid():
    raise RuntimeError(f"Camera prim not found: {WRIST_CAM_PRIM}")

camera = Camera(prim_path=WRIST_CAM_PRIM, resolution=(128, 128))
camera.initialize()
for _ in range(30):
    world.step(render=True)
print("[CAMERA] Ready")

# -----------------------------------------------------------------------------
# LULA IK solver
# -----------------------------------------------------------------------------
def _generate_lula_description(urdf_path: Path, output_path: Path):
    import xml.etree.ElementTree as ET
    root = ET.parse(str(urdf_path)).getroot()
    joints, root_link = [], None
    for joint in root.findall('joint'):
        if joint.get('type') == 'revolute':
            joints.append(joint.get('name'))
            if root_link is None:
                parent = joint.find('parent')
                if parent is not None:
                    root_link = parent.get('link')
    n = len(joints)
    lines = [
        "api_version: 1.0\n\n",
        "cspace:\n",
        *[f"    - {j}\n" for j in joints],
        f"\nroot_link: {root_link or 'base_link'}\n\n",
        f"default_q: [{', '.join(['0.0'] * n)}]\n\n",
        f"acceleration_limits: [{', '.join(['40.0'] * n)}]\n",
        f"jerk_limits: [{', '.join(['500.0'] * n)}]\n",
    ]
    output_path.write_text("".join(lines))
    print(f"[IK] Generated LULA description → {output_path}")

if not LULA_DESC.exists():
    _generate_lula_description(URDF_PATH, LULA_DESC)

ik_solver = LulaKinematicsSolver(
    robot_description_path=str(LULA_DESC),
    urdf_path=str(URDF_PATH),
)
print(f"[IK] LULA solver ready — end-effector frame: {EE_FRAME}")

# -----------------------------------------------------------------------------
# ROS2 node
# -----------------------------------------------------------------------------
class RobotROSNode(Node):
    def __init__(self):
        super().__init__('h2017_ros_node')
        self.joint_positions  = None
        self.eef_delta        = None
        self.gripper_cmd      = None   # float: >0.5 → close, ≤0.5 → open
        self.do_reset         = False
        self.create_subscription(JointState,        '/joint_command', self._joint_cb,     10)
        self.create_subscription(Float64MultiArray, '/eef_delta',     self._eef_delta_cb, 10)
        self.create_subscription(Empty,             '/reset_scene',   self._reset_cb,     1)
        self.state_publisher   = self.create_publisher(JointState, '/joint_states',    10)
        self.camera_publisher  = self.create_publisher(Image,      '/mecheye/color',   1)
        self.gripper_publisher = self.create_publisher(String,     '/gripper_status',  1)
        self.get_logger().info("ROS2 node ready")

    def _joint_cb(self, msg):
        self.joint_positions = list(msg.position)

    def _eef_delta_cb(self, msg):
        if len(msg.data) >= 6:
            self.eef_delta = list(msg.data)
            # 7th element (index 6) carries the gripper command if present
            if len(msg.data) >= 7:
                self.gripper_cmd = float(msg.data[6])

    def _reset_cb(self, _msg):
        self.do_reset = True

    def publish_joint_states(self, positions, velocities, efforts):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = JOINT_NAMES
        msg.position = list(positions)
        msg.velocity = list(velocities)
        msg.effort   = list(efforts)
        self.state_publisher.publish(msg)

    def publish_camera(self, rgb: np.ndarray):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.height   = rgb.shape[0]
        msg.width    = rgb.shape[1]
        msg.encoding = "rgb8"
        msg.step     = rgb.shape[1] * 3
        msg.data     = rgb.tobytes()
        self.camera_publisher.publish(msg)

enable_extension("isaacsim.ros2.bridge")
rclpy.init()
ros_node = RobotROSNode()

Path("/tmp/sim_node_ready").touch()
print("[ROS2] Sentinel written — entering main loop")

_smoothed_rot = np.zeros(3)

# -----------------------------------------------------------------------------
# Simulation loop
# -----------------------------------------------------------------------------
_gripper_state = False   # tracks last commanded state
_loop_count    = 0

while simulation_app.is_running():
    rclpy.spin_once(ros_node, timeout_sec=0)

    # Gripper command.  The extension does NOT self-maintain Closing state
    # between physics steps — close_gripper() must be called every step to
    # keep the raycast alive.  RETRY_INTERVAL gates how often it scans, not
    # a self-managed timer.  open() is one-shot (transition only).
    if ros_node.gripper_cmd is not None:
        cmd = ros_node.gripper_cmd
        ros_node.gripper_cmd = None
        new_state = (cmd > 0.5)
        if new_state != _gripper_state:
            _gripper_state = new_state
            if not _gripper_state:
                gripper.open()

    if _gripper_state:
        gripper.close()

    if ros_node.eef_delta is not None:
        raw = np.array(ros_node.eef_delta[:6])
        ros_node.eef_delta = None
        ros_node.joint_positions = None   # eef_delta overrides stale joint command

        if np.any(np.abs(raw[:3]) > MAX_POS) or np.any(np.abs(raw[3:]) > MAX_ROT):
            ros_node.get_logger().warn(
                f"[SAFETY] Action clamped: pos={raw[:3].round(4)} rot={raw[3:].round(4)}"
            )
            raw[:3] = np.clip(raw[:3], -MAX_POS, MAX_POS)
            raw[3:] = np.clip(raw[3:], -MAX_ROT, MAX_ROT)

        # EMA smoothing on rotation to dampen closed-loop feedback oscillation
        _smoothed_rot[:] = ROT_EMA_ALPHA * raw[3:] + (1 - ROT_EMA_ALPHA) * _smoothed_rot
        dx, dy, dz = raw[:3]
        drx, dry, drz = _smoothed_rot

        q_now = robot.get_joint_positions()
        ee_pos, ee_rot_mat = ik_solver.compute_forward_kinematics(EE_FRAME, q_now)

        target_pos  = ee_pos + np.array([dx, dy, dz])
        xyzw        = (Rotation.from_euler('xyz', [drx, dry, drz]) *
                       Rotation.from_matrix(ee_rot_mat)).as_quat()
        target_quat = xyzw[[3, 0, 1, 2]]

        q_target, success = ik_solver.compute_inverse_kinematics(
            EE_FRAME, target_pos, target_quat,
            position_tolerance=0.005,
            orientation_tolerance=0.01,
        )
        if success:
            robot.get_articulation_controller().apply_action(
                ArticulationAction(joint_positions=q_target)
            )
        else:
            ros_node.get_logger().warn("[IK] IK failed — skipping step")

    elif ros_node.joint_positions is not None:
        robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=ros_node.joint_positions)
        )

    ros_node.publish_joint_states(
        positions=robot.get_joint_positions(),
        velocities=robot.get_joint_velocities(),
        efforts=robot.get_applied_joint_efforts(),
    )

    # Scene reset: teleport cube back to pick position
    if ros_node.do_reset:
        ros_node.do_reset = False
        if _pickup_cube is not None:
            _pickup_cube.set_world_pose(
                position=np.array(PICKUP_POSITION, dtype=float),
                orientation=np.array(PICKUP_ORIENT_WXYZ, dtype=float),
            )
            _pickup_cube.set_linear_velocity(np.zeros(3))
            _pickup_cube.set_angular_velocity(np.zeros(3))
            print("[SCENE] Cube reset to pick position")
        else:
            print(f"[SCENE] Reset requested but cube prim not found — "
                  f"set PICKUP_PRIM_PATH in sim_node.py")

    # Publish gripper status ('open'|'closing'|'closed') + gripped objects
    _gs_msg = String()
    _gripped = gripper._interface.get_gripped_objects(_GRIPPER_PRIM_PATH) if gripper._interface else []
    _gs_msg.data = f"{gripper.status()}|{','.join(_gripped)}"
    ros_node.gripper_publisher.publish(_gs_msg)

    # Keep the kinematic gripper body co-located with link_6 so scan origins
    # are current and the red markers don't drift away from the physical gripper.
    _sync_q = robot.get_joint_positions()
    _sync_pos_rf, _sync_rot_rf = ik_solver.compute_forward_kinematics(EE_FRAME, _sync_q)
    _sync_base_pos, _sync_base_quat = robot.get_world_pose()   # wxyz
    _sync_base_mat = Rotation.from_quat([_sync_base_quat[1], _sync_base_quat[2],
                                         _sync_base_quat[3], _sync_base_quat[0]]).as_matrix()
    _sync_l6_pos  = _sync_base_pos + _sync_base_mat @ _sync_pos_rf
    _sync_l6_xyzw = (Rotation.from_matrix(_sync_base_mat @ _sync_rot_rf)).as_quat()
    gripper.sync_to_link6(
        world_pos=_sync_l6_pos,
        world_quat_wxyz=np.array([_sync_l6_xyzw[3], _sync_l6_xyzw[0],
                                   _sync_l6_xyzw[1], _sync_l6_xyzw[2]]),
    )

    world.step(render=True)
    _loop_count += 1
    if _loop_count % 120 == 0:   # ~every 2 s at 60 fps
        print(f"[GRIPPER] status={gripper.status()} state={'closed' if _gripper_state else 'open'} gripped={_gripped}")
        if _gripper_state:
            _q = robot.get_joint_positions()
            _ee_pos_rf, _ee_rot_rf = ik_solver.compute_forward_kinematics(EE_FRAME, _q)
            _rb_pos, _rb_quat = robot.get_world_pose()   # position=(3,), quat wxyz=(4,)
            _rb_mat = Rotation.from_quat([_rb_quat[1], _rb_quat[2], _rb_quat[3], _rb_quat[0]]).as_matrix()
            for _i, _tip in enumerate(_CUP_TIPS):
                _t_rf      = _ee_pos_rf + _ee_rot_rf @ np.array([_tip[0], _tip[1], _tip[2]])
                _t_w       = _rb_pos + _rb_mat @ _t_rf
                _scan_dir_w = _rb_mat @ (_ee_rot_rf @ np.array([0.0, 0.0, 1.0]))
                _scan_end_w = _t_w + _scan_dir_w * _MAX_GRIP_DISTANCE
                print(f"[GRIPPER-DIAG] cup{_i} tip  WORLD: {_t_w.round(4)}")
                print(f"[GRIPPER-DIAG] cup{_i} scan dir  : {_scan_dir_w.round(4)}")
                print(f"[GRIPPER-DIAG] cup{_i} scan end  : {_scan_end_w.round(4)}")
                # Direct PhysX raycast — reports what is actually in the scan path
                try:
                    import omni.physx as _physx_mod
                    import carb as _carb_mod
                    _qi  = _physx_mod.get_physx_scene_query_interface()
                    _hit = _qi.raycast_closest(
                        _carb_mod.Float3(float(_t_w[0]), float(_t_w[1]), float(_t_w[2])),
                        _carb_mod.Float3(float(_scan_dir_w[0]), float(_scan_dir_w[1]), float(_scan_dir_w[2])),
                        float(_MAX_GRIP_DISTANCE),
                    )
                    print(f"[GRIPPER-DIAG] PhysX ray hit   : {_hit}")
                except Exception as _physx_err:
                    print(f"[GRIPPER-DIAG] PhysX ray error : {_physx_err}")
            print(f"[GRIPPER-DIAG] robot base world : {_rb_pos.round(4)}")
            print(f"[GRIPPER-DIAG] cube center      : {PICKUP_POSITION}")
            if _pickup_cube is not None:
                _cube_actual_pos, _ = _pickup_cube.get_world_pose()
                print(f"[GRIPPER-DIAG] cube actual pos  : {_cube_actual_pos.round(4)}")

    rgba = camera.get_rgba()
    if rgba is not None and rgba.size > 0:
        rgb = (rgba[:, :, :3] * 255).clip(0, 255).astype(np.uint8)
        ros_node.publish_camera(rgb)

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
ros_node.destroy_node()
rclpy.shutdown()
simulation_app.close()
