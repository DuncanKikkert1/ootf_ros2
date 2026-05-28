# tcp_ros_bridge.py — TCP receiver that publishes joint positions to /joint_command.
#
# Listens on port 9002 for semicolon-separated messages, validates them,
# and publishes joint positions as sensor_msgs/JointState.
#
# Message format: key_cmd;status;extra_num;gripper;x;y;z;aw;ap;ar
# Example:        250;254;0;0;1.57;0;1.57;1.57;-1.57;0

import re
import socket
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
HOST      = '0.0.0.0'
PORT      = 9002
ROS_TOPIC = '/joint_command'

FIELD_NAMES = ['key_cmd', 'status', 'extra_num', 'gripper', 'x', 'y', 'z', 'aw', 'ap', 'ar']

# Fields 1-3: integer only.  Fields 4-10: optional sign + digits + optional decimal.
PATTERN = re.compile(
    r'^\d+;'
    r'\d+;'
    r'\d+;'
    r'[+\-]?\d+(\.\d+)?;'
    r'[+\-]?\d+(\.\d+)?;'
    r'[+\-]?\d+(\.\d+)?;'
    r'[+\-]?\d+(\.\d+)?;'
    r'[+\-]?\d+(\.\d+)?;'
    r'[+\-]?\d+(\.\d+)?;'
    r'[+\-]?\d+(\.\d+)?$'
)


def parse_message(message):
    """Split a validated message string into a named-field dict."""
    fields = message.split(';')
    return {
        'key_cmd':   int(fields[0]),
        'status':    int(fields[1]),
        'extra_num': int(fields[2]),
        'gripper':   float(fields[3]),
        'x':         float(fields[4]),
        'y':         float(fields[5]),
        'z':         float(fields[6]),
        'aw':        float(fields[7]),
        'ap':        float(fields[8]),
        'ar':        float(fields[9]),
    }


class JointCommandPublisher(Node):
    """ROS2 node that publishes parsed joint positions to /joint_command."""

    def __init__(self):
        super().__init__('tcp_ros_bridge')
        self.publisher_ = self.create_publisher(JointState, ROS_TOPIC, 10)
        self.get_logger().info(f"Publishing to {ROS_TOPIC}")

    def publish(self, parsed):
        """Publish a JointState from a parsed message dict."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
        msg.position = [parsed['x'], parsed['y'], parsed['z'], parsed['aw'], parsed['ap'], parsed['ar']]
        self.publisher_.publish(msg)
        self.get_logger().info(f"Published: {msg.position}")


def main():
    """Accept one TCP connection and relay validated messages to ROS2."""
    rclpy.init()
    node = JointCommandPublisher()

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind((HOST, PORT))
        s.listen(1)
        print(f"Listening on port {PORT}...")

        try:
            conn, addr = s.accept()
            with conn:
                print(f"Welcome {addr}, please have a drink and stay a while...")
                while True:
                    data = conn.recv(1024)
                    if not data:
                        print("Sad to see you go, closing connection...")
                        break

                    message = data.decode('utf-8').strip()

                    if PATTERN.match(message):
                        parsed = parse_message(message)
                        print(f"Valid message received: {message}")
                        print(f"  key_cmd   : {parsed['key_cmd']}")
                        print(f"  status    : {parsed['status']}")
                        print(f"  extra_num : {parsed['extra_num']}")
                        print(f"  gripper   : {parsed['gripper']}")
                        print(f"  x         : {parsed['x']}")
                        print(f"  y         : {parsed['y']}")
                        print(f"  z         : {parsed['z']}")
                        print(f"  aw        : {parsed['aw']}")
                        print(f"  ap        : {parsed['ap']}")
                        print(f"  ar        : {parsed['ar']}")
                        node.publish(parsed)
                    else:
                        fields = message.split(';')
                        if len(fields) != 10:
                            print(f"Invalid message rejected: expected 10 fields, got {len(fields)}: {message}")
                        else:
                            for name, value in zip(FIELD_NAMES, fields):
                                if not re.fullmatch(r'\d+' if name in ('key_cmd', 'status', 'extra_num') else r'[+\-]?\d+(\.\d+)?', value):
                                    print(f"Invalid message rejected: field '{name}' has invalid value '{value}'")

        except KeyboardInterrupt:
            print("\nShutting down cleanly...")
        finally:
            node.destroy_node()
            rclpy.shutdown()

if __name__ == '__main__':
    main()
