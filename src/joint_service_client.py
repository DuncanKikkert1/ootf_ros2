# =============================================================================
# Name        : joint_service_client.py
# Author      : Duncan Kikkert
# Created     : 15/4/2026
# Last Update : 15/4/2026
# Version     : 1.0
# Description : Subscribes to /joint_command and forwards joint positions to
#               the Doosan MoveJoint ROS2 service. Joint positions are
#               converted from radians (ROS2 convention) to degrees (Doosan).
#
# Dependencies : rclpy, sensor_msgs, dsr_msgs2
# Usage        : bash launch/launch_bridge.sh (or run standalone with ROS2 sourced)
# =============================================================================

import math
import rclpy
from rclpy.node import Node
from rclpy.callback_groups import ReentrantCallbackGroup
from sensor_msgs.msg import JointState
from dsr_msgs2.srv import MoveJoint

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SUBSCRIBE_TOPIC  = '/joint_command'
SERVICE_NAME     = '/dsr01/motion/move_joint'

DEFAULT_VEL        = 30.0   # deg/sec
DEFAULT_ACC        = 30.0   # deg/sec²
DEFAULT_TIME       = 0.0    # sec  (0 = use vel/acc)
DEFAULT_RADIUS     = 0.0    # mm   (blending radius)
DEFAULT_MODE       = 0      # MOVE_MODE_ABSOLUTE
DEFAULT_BLEND_TYPE = 0      # BLENDING_SPEED_TYPE_DUPLICATE
DEFAULT_SYNC_TYPE  = 0      # SYNC

# -----------------------------------------------------------------------------
# Node
# -----------------------------------------------------------------------------
class JointServiceClient(Node):
    def __init__(self):
        super().__init__('joint_service_client')

        self.callback_group = ReentrantCallbackGroup()

        self.client = self.create_client(
            MoveJoint,
            SERVICE_NAME,
            callback_group=self.callback_group
        )

        self.create_subscription(
            JointState,
            SUBSCRIBE_TOPIC,
            self._joint_command_callback,
            10,
            callback_group=self.callback_group
        )

        self.get_logger().info(f"Subscribed to {SUBSCRIBE_TOPIC}")
        self.get_logger().info(f"Waiting for service {SERVICE_NAME}...")
        self.client.wait_for_service()
        self.get_logger().info("Service ready.")

    def _joint_command_callback(self, msg):
        if len(msg.position) < 6:
            self.get_logger().warn(f"Expected 6 joint positions, got {len(msg.position)} — skipping.")
            return

        # Convert radians to degrees for the Doosan service
        pos_deg = [math.degrees(p) for p in msg.position[:6]]

        request = MoveJoint.Request()
        request.pos        = pos_deg
        request.vel        = DEFAULT_VEL
        request.acc        = DEFAULT_ACC
        request.time       = DEFAULT_TIME
        request.radius     = DEFAULT_RADIUS
        request.mode       = DEFAULT_MODE
        request.blend_type = DEFAULT_BLEND_TYPE
        request.sync_type  = DEFAULT_SYNC_TYPE

        self.get_logger().info(f"Sending MoveJoint request: {[round(p, 2) for p in pos_deg]} deg")

        future = self.client.call_async(request)
        future.add_done_callback(self._response_callback)

    def _response_callback(self, future):
        try:
            response = future.result()
            if response.success:
                self.get_logger().info("MoveJoint succeeded.")
            else:
                self.get_logger().warn("MoveJoint returned success=False.")
        except Exception as e:
            self.get_logger().error(f"MoveJoint service call failed: {e}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    rclpy.init()
    node = JointServiceClient()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        print("\nShutting down cleanly...")
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
