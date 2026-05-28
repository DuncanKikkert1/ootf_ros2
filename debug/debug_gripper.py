#!/usr/bin/env python3
# debug_gripper.py — Interactive gripper test (requires sim_node.py running).
#
# Sends the robot to a fixed joint configuration, closes the gripper,
# returns home, opens the gripper, then resets the cube to its pick position.
#
# Usage:
#   python3 debug/debug_gripper.py
#   ./run.sh debug gripper-test

import math
import time

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Empty, Float64MultiArray, String

# ── Configuration ────────────────────────────────────────────────────────────

# Joint target in DEGREES — edit these to match your pick position
JOINT_DEG = [27.0, 84.0, 70.5, 0.0, 25.0, 34.5]

JOINT_NAMES   = ['joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5', 'joint_6']
SETTLE_SEC    = 3.0    # seconds to wait for the robot to reach the target
GRIP_WAIT_SEC = 3.0    # seconds to hold after closing (allow gripper retries)
RETURN_SEC    = 3.0    # seconds to wait while moving back to home
RELEASE_SEC   = 1.0    # seconds after opening the gripper before reset
POS_TOL_RAD   = 0.05   # radians — "close enough" threshold per joint

# Home position to return to after gripping (degrees)
HOME_DEG = [0.0, 0.0, 90.0, 0.0, 90.0, 0.0]

# ─────────────────────────────────────────────────────────────────────────────

JOINT_RAD = [math.radians(d) for d in JOINT_DEG]
HOME_RAD  = [math.radians(d) for d in HOME_DEG]


class GripperTester(Node):
    """ROS2 node that drives the robot through a pick-grip-return-release sequence."""

    def __init__(self):
        """Subscribe to joint states and gripper status; advertise control topics."""
        super().__init__('gripper_tester')
        self._current_positions = None
        self._gripper_status    = "unknown"
        self._gripped_objects   = []

        self._joint_pub  = self.create_publisher(JointState,        '/joint_command', 10)
        self._delta_pub  = self.create_publisher(Float64MultiArray, '/eef_delta',     10)
        self._reset_pub  = self.create_publisher(Empty,             '/reset_scene',   1)
        self._state_sub  = self.create_subscription(
            JointState, '/joint_states', self._state_cb, 10
        )
        self._status_sub = self.create_subscription(
            String, '/gripper_status', self._gripper_status_cb, 10
        )

    def _state_cb(self, msg: JointState):
        self._current_positions = list(msg.position)

    def _gripper_status_cb(self, msg: String):
        parts = msg.data.split("|", 1)
        self._gripper_status  = parts[0]   # 'open' | 'closing' | 'closed'
        self._gripped_objects = [o for o in parts[1].split(",") if o] if len(parts) > 1 else []

    def send_joints(self, positions_rad: list):
        """Publish a JointState command."""
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = JOINT_NAMES
        msg.position = positions_rad
        self._joint_pub.publish(msg)

    def send_gripper(self, close: bool):
        """Send gripper command via the 7th element of /eef_delta."""
        msg = Float64MultiArray()
        msg.data = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0 if close else 0.0]
        self._delta_pub.publish(msg)

    def send_reset(self):
        """Teleport the pick cube back to its configured pick position."""
        self._reset_pub.publish(Empty())

    def wait_for_joint_state(self, timeout=5.0):
        """Block until a /joint_states message arrives."""
        deadline = time.monotonic() + timeout
        while self._current_positions is None:
            rclpy.spin_once(self, timeout_sec=0.05)
            if time.monotonic() > deadline:
                raise RuntimeError("No /joint_states received — is sim_node.py running?")

    def at_target(self) -> bool:
        """Return True if all joints are within POS_TOL_RAD of JOINT_RAD."""
        if self._current_positions is None:
            return False
        return all(
            abs(cur - tgt) < POS_TOL_RAD
            for cur, tgt in zip(self._current_positions, JOINT_RAD)
        )

    def spin_for(self, seconds: float):
        """Process ROS2 callbacks for the given duration."""
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)

    def run(self):
        """Execute the full pick-grip-return-release test sequence."""
        print("\n── Gripper test ─────────────────────────────────────────────")
        print(f"Target joints (deg): {JOINT_DEG}")
        print(f"Target joints (rad): {[round(r, 4) for r in JOINT_RAD]}")

        print("\n[1/6] Waiting for sim_node to be ready...")
        self.wait_for_joint_state()
        print(f"      Current positions: {[round(p, 4) for p in self._current_positions]}")

        print(f"\n[2/6] Moving to pick position — waiting {SETTLE_SEC}s...")
        deadline = time.monotonic() + SETTLE_SEC
        while time.monotonic() < deadline:
            self.send_joints(JOINT_RAD)
            self.spin_for(0.1)

        at = self.at_target()
        print(f"      At target: {at}")
        if self._current_positions:
            errs = [abs(c - t) for c, t in zip(self._current_positions, JOINT_RAD)]
            print(f"      Joint errors (rad): {[round(e, 4) for e in errs]}")

        print(f"\n[3/6] Closing gripper — waiting {GRIP_WAIT_SEC}s for retries...")
        self.send_gripper(close=True)
        self.spin_for(GRIP_WAIT_SEC)

        print(f"      Gripper status : {self._gripper_status}")
        if self._gripper_status == "closing":
            print("      → Still retrying — cube may be out of range.")
            print("        Check [GRIPPER-DIAG] lines in T1 for cup tip vs cube world positions.")
        elif self._gripper_status == "closed":
            print(f"      → Grip confirmed!")
        else:
            print("      → Never entered 'closing' state — check gripper prim setup.")

        if self._gripped_objects:
            print(f"      Gripped objects: {self._gripped_objects}")

        print(f"\n[4/6] Returning to home {HOME_DEG} — waiting {RETURN_SEC}s (gripper holding)...")
        deadline = time.monotonic() + RETURN_SEC
        while time.monotonic() < deadline:
            self.send_joints(HOME_RAD)
            self.spin_for(0.1)

        gripped_at_home = self._gripper_status == "closed"
        if gripped_at_home:
            print("      Cube is attached — gripper still closed.")
        else:
            print("      Gripper is open (cube not carried or was dropped).")

        print(f"\n[5/6] Opening gripper — releasing object...")
        self.send_gripper(close=False)
        self.spin_for(RELEASE_SEC)
        print(f"      Gripper status: {self._gripper_status}")

        print(f"\n[6/6] Resetting cube to pick position...")
        self.send_reset()
        self.spin_for(0.5)
        print("      Reset command sent.")

        print("\nResult")
        if gripped_at_home:
            print("      SUCCESS — cube moved with the arm and was released at home.")
        else:
            print("      The cube did not travel with the arm.")
            print("      Check red markers in the viewport against the cube surface.")
        print("─────────────────────────────────────────────────────────────\n")


def main():
    rclpy.init()
    node = GripperTester()
    try:
        node.run()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
