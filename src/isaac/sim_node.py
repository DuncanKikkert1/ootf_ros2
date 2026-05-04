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
from std_msgs.msg import Float64MultiArray
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import is_stage_loading
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
from isaacsim.sensors.camera import Camera

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
ROBOT_PRIM     = "/World/h2017"
EE_FRAME       = "link_6"
WRIST_CAM_PRIM = "/World/h2017/link_6/MechEye/Camera"
URDF_PATH      = Path(__file__).parent.parent.parent / "scenes" / "h2017" / "urdf" / "h2017.urdf"
LULA_DESC      = Path(__file__).parent.parent.parent / "scenes" / "h2017" / "urdf" / "h2017_lula.yaml"
JOINT_NAMES    = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
HOME_POSITION  = [0.0, 0.0, 1.57, 0.0, 1.57, 0.0]
MAX_POS        = 0.05   # metres per step — EEF delta safety clamp
MAX_ROT        = 0.15   # radians per step

# -----------------------------------------------------------------------------
# Scene preprocessing — strip OmniGraph prims from the USD before loading.
# RemovePrim() is required: SetActive(False) does not stop the Kit OmniGraph
# runtime from executing the nodes, which crashes omni.graph.image.core while
# Fabric is still syncing robot mesh prims.
# -----------------------------------------------------------------------------
_RAW_SCENE   = Path(__file__).parent.parent.parent / "scenes" / "usd" / "doosan_BIC.usd"
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
# Physics world + robot
# -----------------------------------------------------------------------------
world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
robot = world.scene.add(Robot(prim_path=ROBOT_PRIM, name="robot"))
world.reset()
print(f"[ROBOT] {robot.num_dof} DOF")

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
_cam_prim = omni.usd.get_context().get_stage().GetPrimAtPath(WRIST_CAM_PRIM)
if not _cam_prim.IsValid():
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
        self.joint_positions = None
        self.eef_delta       = None
        self.create_subscription(JointState,        '/joint_command', self._joint_cb,     10)
        self.create_subscription(Float64MultiArray, '/eef_delta',     self._eef_delta_cb, 10)
        self.state_publisher  = self.create_publisher(JointState, '/joint_states', 10)
        self.camera_publisher = self.create_publisher(Image, '/mecheye/color', 1)
        self.get_logger().info("ROS2 node ready")

    def _joint_cb(self, msg):
        self.joint_positions = list(msg.position)

    def _eef_delta_cb(self, msg):
        if len(msg.data) >= 6:
            self.eef_delta = list(msg.data)

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

# -----------------------------------------------------------------------------
# Simulation loop
# -----------------------------------------------------------------------------
while simulation_app.is_running():
    rclpy.spin_once(ros_node, timeout_sec=0)

    if ros_node.eef_delta is not None:
        raw = np.array(ros_node.eef_delta[:6])
        ros_node.eef_delta = None

        if np.any(np.abs(raw[:3]) > MAX_POS) or np.any(np.abs(raw[3:]) > MAX_ROT):
            ros_node.get_logger().warn(
                f"[SAFETY] Action clamped: pos={raw[:3].round(4)} rot={raw[3:].round(4)}"
            )
            raw[:3] = np.clip(raw[:3], -MAX_POS, MAX_POS)
            raw[3:] = np.clip(raw[3:], -MAX_ROT, MAX_ROT)
        dx, dy, dz, drx, dry, drz = raw

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
    world.step(render=True)
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
