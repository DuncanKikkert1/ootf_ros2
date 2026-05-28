# tcp_sender.py — Persistent TCP client for sending 7-DOF EEF delta actions.
#
# Message format (newline-terminated, 7 semicolon-separated floats):
#   dx;dy;dz;drx;dry;drz;gripper\n
#
# Usage:
#   with EEFDeltaSender('127.0.0.1', 9001) as sender:
#       sender.send(action)

import socket


class ROS2EEFPublisher:
    """Publishes 7-DOF EEF delta actions to a ROS2 Float64MultiArray topic."""

    def __init__(self, topic: str = "/eef_delta"):
        self.topic  = topic
        self._node  = None
        self._pub   = None

    def connect(self):
        """Initialise ROS2 and create the publisher node."""
        import rclpy
        from std_msgs.msg import Float64MultiArray
        if not rclpy.ok():
            rclpy.init()
        self._node = rclpy.create_node("octo_eef_publisher")
        self._pub  = self._node.create_publisher(Float64MultiArray, self.topic, 10)
        print(f"[ROS2] Publishing EEF deltas on {self.topic}")

    def send(self, action) -> None:
        """Publish a 7-value EEF delta action as a Float64MultiArray message."""
        from std_msgs.msg import Float64MultiArray
        msg      = Float64MultiArray()
        msg.data = [float(v) for v in action]
        self._pub.publish(msg)

    def close(self):
        """Destroy the publisher node."""
        if self._node is not None:
            self._node.destroy_node()
            self._node = None
            print("[ROS2] EEF publisher node destroyed.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()


class EEFDeltaSender:
    """Persistent TCP connection for sending 7-DOF EEF delta actions."""

    FIELDS = ["dx", "dy", "dz", "drx", "dry", "drz", "gripper"]

    # def __init__(self, host: str = "127.0.0.1", port: int = 9001): # For local
    def __init__(self, host: str = "192.168.1.244", port: int = 9005): # For laptop Duncan
        self.host  = host
        self.port  = port
        self._sock = None

    def connect(self):
        """Open the TCP connection."""
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.connect((self.host, self.port))
        print(f"[TCP] Connected to {self.host}:{self.port}")

    def close(self):
        """Close the TCP connection."""
        if self._sock:
            self._sock.close()
            self._sock = None
            print("[TCP] Connection closed.")

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, *_):
        self.close()

    def send(self, action) -> None:
        """Send a 7-value EEF delta action as a semicolon-separated string."""
        if self._sock is None:
            raise RuntimeError("Not connected. Call connect() or use as a context manager.")

        action = list(action)
        if len(action) != 7:
            raise ValueError(f"Expected 7 action values, got {len(action)}.")

        msg = ";".join(f"{v:.6f}" for v in action) + "\n"
        self._sock.sendall(msg.encode("utf-8"))
