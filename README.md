# ootf_ros2

A ROS2-based pipeline for controlling a Doosan H2017 robot arm in NVIDIA Isaac Sim via TCP commands. Robot joint positions are received over TCP, validated, and published directly to a ROS2 topic, then applied to a physics simulation in real time.

---

## Architecture

```
TCP Client
    │
    ▼
tcp_ros_bridge.py   — Validates and parses TCP messages, publishes to /joint_command
    │
    ▼ (/joint_command)
simulation.py       — Isaac Sim simulation, applies joint positions to robot
    │
    ▼ (/joint_states)
Any ROS2 subscriber — Live joint state feedback from the simulation
```

---

## Dependencies

### System
- Ubuntu 24.04
- [NVIDIA Isaac Sim](https://developer.nvidia.com/isaac-sim) (Python 3.11, via pyenv)
- ROS2 Jazzy
- [Isaac Sim ROS2 Workspace](https://github.com/isaac-sim/IsaacSim-ros_workspaces)

### Python
- `rclpy` (provided by Isaac Sim's internal ROS2 bridge)
- `sensor_msgs` (provided by Isaac Sim's internal ROS2 bridge)
- `isaacsim` (installed in pyenv environment)

No additional pip installs are required. All dependencies are either built into Isaac Sim or provided by the ROS2 workspace.

---

## Repository Structure

```
ootf_ros2/
├── launch/
│   ├── launch_isaacsim.sh      # Start Isaac Sim with the robot
│   └── launch_bridge.sh        # Start the TCP/ROS2 bridge
├── scenes/
│   └── h2017/
│       ├── h2017.urdf          # Doosan H2017 robot description
│       ├── meshes_white/       # Visual mesh files (.dae)
│       └── meshes_collision/   # Collision mesh files (.dae)
└── src/
    ├── simulation.py           # Isaac Sim simulation entry point
    └── tcp_ros_bridge.py       # Combined TCP receiver and ROS2 publisher
```

---

## Message Format

Commands are sent as semicolon-separated strings over TCP:

```
key_cmd;status;extra_num;gripper;x;y;z;aw;ap;ar
```

| Field       | Type  | Description              |
|-------------|-------|--------------------------|
| `key_cmd`   | int   | Command code             |
| `status`    | int   | Status code              |
| `extra_num` | int   | Extra parameter          |
| `gripper`   | float | Gripper value            |
| `x`         | float | Joint 1 position (rad)   |
| `y`         | float | Joint 2 position (rad)   |
| `z`         | float | Joint 3 position (rad)   |
| `aw`        | float | Joint 4 position (rad)   |
| `ap`        | float | Joint 5 position (rad)   |
| `ar`        | float | Joint 6 position (rad)   |

Example:
```
250;254;0;0;0;0;0;0;0;0
```

---

## Usage

Open two terminals and run each launch script in order.

**Terminal 1 — Isaac Sim**
```bash
bash launch/launch_isaacsim.sh
```
Wait until the Isaac Sim viewport is fully loaded before starting the bridge.

**Terminal 2 — TCP/ROS2 Bridge**
```bash
bash launch/launch_bridge.sh
```

Then connect a TCP client to port `9000` on the machine's IP and start sending commands in the format described above.

---

## Verifying the Pipeline

Check that joint states are being published:
```bash
ros2 topic echo /joint_states
```

Check that joint commands are being received:
```bash
ros2 topic echo /joint_command
```
