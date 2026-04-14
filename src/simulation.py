###
# This script imports a URDF robot into Isaac Sim, sets up the world and
# physics, configures joint drive gains, and runs the simulation loop.
# ROS2 joint commands received on /joint_command are applied to the robot.
#
# Place your robot's .urdf file in scenes/<robot>/ before running.
# Run with: bash launch/launch_isaacsim.sh
###

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
from isaacsim.asset.importer.urdf import _urdf

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
