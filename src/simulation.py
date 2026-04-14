# =============================================================================
# Name        : simulation.py
# Author      : Duncan Kikkert
# Created     : 14/4/2026
# Last Update : 14/4/2026
# Version     : 1.0
# Description : Isaac Sim entry point. Imports the H2017 robot from URDF,
#               sets up the physics world, and runs the simulation loop.
#               Subscribes to /joint_command to move the robot and publishes
#               live joint states to /joint_states.
#
# Dependencies : Isaac Sim (Python 3.11), ROS2 Jazzy, rclpy, sensor_msgs
# Usage        : bash launch/launch_isaacsim.sh
# =============================================================================

from pathlib import Path
from isaacsim import SimulationApp

simulation_app = SimulationApp({"headless": False})

import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.asset.importer.urdf import _urdf
from pxr import UsdGeom, UsdLux, UsdShade, Sdf, Gf
import omni.usd

# Enable ROS2 bridge before setting up the world
enable_extension("isaacsim.ros2.bridge")

# -----------------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------------
SCENES_DIR = Path(__file__).parent.parent / "scenes"
ROBOT_DIR  = SCENES_DIR / "h2017"
URDF_PATH  = ROBOT_DIR / "h2017.urdf"

# -----------------------------------------------------------------------------
# ROS2 subscriber node
# -----------------------------------------------------------------------------
JOINT_NAMES = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']

class RobotROSNode(Node):
    def __init__(self):
        super().__init__('h2017_ros_node')
        self.joint_positions = None
        self.create_subscription(JointState, '/joint_command', self._command_callback, 10)
        self.state_publisher = self.create_publisher(JointState, '/joint_states', 10)
        self.get_logger().info("Subscribed to /joint_command, publishing to /joint_states")

    def _command_callback(self, msg):
        self.joint_positions = list(msg.position)

    def publish_joint_states(self, positions, velocities, efforts):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name = JOINT_NAMES
        msg.position = list(positions)
        msg.velocity = list(velocities)
        msg.effort = list(efforts)
        self.state_publisher.publish(msg)

# -----------------------------------------------------------------------------
# World and physics setup — must exist before importing URDF
# -----------------------------------------------------------------------------
world = World(physics_dt=1.0 / 60.0, rendering_dt=1.0 / 60.0, stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

stage = omni.usd.get_context().get_stage()

# -----------------------------------------------------------------------------
# Lighting — dome light for uniform ambient light with no shadows
# -----------------------------------------------------------------------------
dome_light = UsdLux.DomeLight.Define(stage, "/World/DomeLight")
dome_light.CreateIntensityAttr(300.0)
dome_light.CreateColorAttr(Gf.Vec3f(1.0, 1.0, 1.0))
dome_light.GetPrim().CreateAttribute("inputs:shadow:enable", Sdf.ValueTypeNames.Bool).Set(False)

# -----------------------------------------------------------------------------
# Ground — light grey material applied to default ground plane
# -----------------------------------------------------------------------------
mat = UsdShade.Material.Define(stage, "/World/GroundMaterial")
shader = UsdShade.Shader.Define(stage, "/World/GroundMaterial/Shader")
shader.CreateIdAttr("UsdPreviewSurface")
shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.85, 0.85, 0.85))
shader.CreateInput("roughness",    Sdf.ValueTypeNames.Float).Set(0.9)
mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
ground = stage.GetPrimAtPath("/World/defaultGroundPlane")
UsdShade.MaterialBindingAPI(ground).Bind(mat)

# -----------------------------------------------------------------------------
# Scene assets — 3 stacked pallets
# -----------------------------------------------------------------------------
PALLET_URL    = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/5.1/Isaac/Props/Pallet/pallet.usd"
PALLET_HEIGHT = 0.144  # metres — adjust if pallets overlap or gap after loading

# Rotations per pallet — alternating 0°/90° for realistic stacking
PALLET_ROTATIONS = [89.5, 90.7, 90.0]

for i in range(3):
    prim_path = f"/World/Pallet_{i}"
    add_reference_to_stage(usd_path=PALLET_URL, prim_path=prim_path)
    xform = UsdGeom.Xformable(stage.GetPrimAtPath(prim_path))
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(1.0, 0.0, PALLET_HEIGHT * i))
    xform.AddRotateXYZOp().Set(Gf.Vec3f(0.0, 0.0, PALLET_ROTATIONS[i]))
    xform.AddScaleOp().Set(Gf.Vec3f(1.0, 1.0, 1.0))

# -----------------------------------------------------------------------------
# URDF import configuration
# -----------------------------------------------------------------------------
urdf_interface = _urdf.acquire_urdf_interface()
import_config   = _urdf.ImportConfig()

import_config.fix_base                        = True
import_config.import_inertia_tensor           = True
import_config.self_collision                  = False
import_config.default_drive_type             = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
import_config.default_drive_strength         = 1e6
import_config.default_position_drive_damping = 1e5

# Parse and import the URDF into the active stage
robot_urdf = urdf_interface.parse_urdf(str(ROBOT_DIR), URDF_PATH.name, import_config)
prim_path  = urdf_interface.import_robot(str(ROBOT_DIR), URDF_PATH.name, robot_urdf, import_config)

if not prim_path:
    raise RuntimeError(f"Failed to import URDF from {URDF_PATH}")

print(f"Robot imported at: {prim_path}")

# -----------------------------------------------------------------------------
# Wrap the imported prim as a Robot and configure joint gains
# -----------------------------------------------------------------------------
robot = world.scene.add(Robot(prim_path=prim_path, name="robot"))

world.reset()

num_joints = robot.num_dof
print(f"Robot has {num_joints} DOF")

robot.get_articulation_controller().set_gains(
    kps=np.full(num_joints, 1e6),
    kds=np.full(num_joints, 1e5),
)

# -----------------------------------------------------------------------------
# ROS2 initialisation
# -----------------------------------------------------------------------------
rclpy.init()
ros_node = RobotROSNode()

# -----------------------------------------------------------------------------
# Simulation loop
# -----------------------------------------------------------------------------
while simulation_app.is_running():
    # Process any incoming ROS2 messages without blocking
    rclpy.spin_once(ros_node, timeout_sec=0)

    if ros_node.joint_positions is not None:
        robot.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=ros_node.joint_positions)
        )

    ros_node.publish_joint_states(
        positions=robot.get_joint_positions(),
        velocities=robot.get_joint_velocities(),
        efforts=robot.get_applied_joint_efforts(),
    )

    world.step(render=True)

# -----------------------------------------------------------------------------
# Cleanup
# -----------------------------------------------------------------------------
ros_node.destroy_node()
rclpy.shutdown()
simulation_app.close()
