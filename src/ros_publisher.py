# =============================================================================
# Name        : ros_publisher.py
# Author      : Duncan Kikkert
# Created     : 13/4/2026
# Last update : 14/4/2026
# Version     : 1.1
# Description : Receives parsed TCP messages forwarded by receiver.py and
#               publishes them to a ROS2 topic as a sensor_msgs/JointState message.
#               msg.position is populated from the named fields x, y, z, aw, ap, ar.
#
# Requirements: Must be run with ROS2 Jazzy sourced:
#               source /opt/ros/jazzy/setup.bash && python3 ros_publisher.py
# =============================================================================

import json
import socket
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
FORWARD_PORT  = 9001
ROS_TOPIC     = '/joint_command'

# -----------------------------------------------------------------------------
# ROS2 publisher node
# -----------------------------------------------------------------------------
class CommandPublisher(Node):
    def __init__(self):
        super().__init__('doosan_command_publisher')
        self.publisher_ = self.create_publisher(JointState, ROS_TOPIC, 10)
        self.get_logger().info(f"Publishing to ROS2 topic: {ROS_TOPIC}")

    def publish(self, parsed):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.position = [
            parsed['x'],
            parsed['y'],
            parsed['z'],
            parsed['aw'],
            parsed['ap'],
            parsed['ar'],
        ]
        msg.name = [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]
        msg.velocity = [
            .5,
            .5,
            .5,
            .5,
            .5,
            .5,
        ]
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published positions: {msg.position}")

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    rclpy.init()
    node = CommandPublisher()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(('127.0.0.1', FORWARD_PORT))
        s.listen(5)
        print(f"Waiting for forwarded messages on port {FORWARD_PORT}...")

        try:
            while True:
                conn, addr = s.accept()
                with conn:
                    data = conn.recv(1024)
                    if data:
                        parsed = json.loads(data.decode('utf-8').strip())
                        node.publish(parsed)

        except KeyboardInterrupt:
            print("\nShutting down ROS publisher...")
        finally:
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
