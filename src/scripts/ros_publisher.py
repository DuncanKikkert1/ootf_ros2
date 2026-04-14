# =============================================================================
# Name        : ros_publisher.py
# Author      : Duncan Kikkert
# Date        : 13/4/2026
# Version     : 1.0
# Description : Receives validated TCP messages forwarded by receiver.py and
#               publishes them to a ROS2 topic as a std_msgs/String message.
#
# Requirements: Must be run with ROS2 Jazzy sourced:
#               source /opt/ros/jazzy/setup.bash && python3 ros_publisher.py
# =============================================================================

import socket
import rclpy
from rclpy.node import Node
from std_msgs.msg import String

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
FORWARD_PORT  = 9001
ROS_TOPIC     = '/Affix/commands'

# -----------------------------------------------------------------------------
# ROS2 publisher node
# -----------------------------------------------------------------------------
class CommandPublisher(Node):
    def __init__(self):
        super().__init__('doosan_command_publisher')
        self.publisher_ = self.create_publisher(String, ROS_TOPIC, 10)
        self.get_logger().info(f"Publishing to ROS2 topic: {ROS_TOPIC}")

    def publish(self, message):
        msg = String()
        msg.data = message
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published: {message}")

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
                        message = data.decode('utf-8').strip()
                        node.publish(message)

        except KeyboardInterrupt:
            print("\nShutting down ROS publisher...")
        finally:
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()