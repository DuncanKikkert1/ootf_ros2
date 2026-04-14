# ootf_ros2

A ROS2-based pipeline for controlling a Doosan H2017 robot arm in NVIDIA Isaac Sim via TCP commands. Robot joint positions are received over TCP, validated, forwarded to a ROS2 topic, and applied to a physics simulation in real time.

---

## Architecture

```
TCP Client
    │
    ▼
receiver.py         — Validates incoming TCP messages, parses joint positions
    │
    ▼ (JSON over TCP)
ros_publisher.py    — Publishes joint positions to /joint_command (ROS2)
    │
    ▼ (/joint_command)
test.py             — Isaac Sim simulation, applies positions to robot joints
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
│   ├── launch_publisher.sh     # Start the ROS2 joint command publisher
│   └── launch_receiver.sh      # Start the TCP receiver
├── scenes/
│   ├── h2017.urdf              # Doosan H2017 robot description
│   ├── h2017_white/            # Visual mesh files (.dae)
│   └── h2017_collision/        # Collision mesh files (.dae)
└── src/scripts/
    ├── test.py                 # Isaac Sim simulation entry point
    ├── ros_publisher.py        # ROS2 publisher node
    └── receiver.py             # TCP server and message validator
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

Open three terminals from the project root and run each launch script in order.

**Terminal 1 — Isaac Sim**
```bash
bash launch/launch_isaacsim.sh
```
Wait until the Isaac Sim viewport is fully loaded before starting the other terminals.

**Terminal 2 — ROS2 Publisher**
```bash
bash launch/launch_publisher.sh
```

**Terminal 3 — TCP Receiver**
```bash
bash launch/launch_receiver.sh
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
