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

## Installation

### 1. System requirements
- Ubuntu 22.04 or 24.04
- ROS2 (Humble for Ubuntu 22.04, Jazzy for Ubuntu 24.04)
- Python 3.11

### 2. Install ROS2
Follow the official guide for your distro:
- [ROS2 Humble (Ubuntu 22.04)](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- [ROS2 Jazzy (Ubuntu 24.04)](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)

### 3. Install NVIDIA Isaac Sim
Install via pip into a Python 3.11 environment:
```bash
pip install isaacsim --extra-index-url https://pypi.nvidia.com
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit isaacsim-extscache-kit-sdk --extra-index-url https://pypi.nvidia.com
```

### 4. Set up the Isaac Sim ROS2 workspace
Clone and build the Isaac Sim ROS2 workspace for your distro:
```bash
git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git ~/IsaacSim-ros_workspaces
cd ~/IsaacSim-ros_workspaces
# Follow the build instructions in that repo for your ROS2 distro
```

### 5. Clone this repository
```bash
git clone https://github.com/DuncanKikkert1/ootf_ros2.git
cd ootf_ros2
```

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

The launch scripts auto-detect your ROS2 distro, Python environment, and Isaac Sim paths. If your Isaac Sim ROS2 workspace is in a non-standard location, set `ISAAC_WS` before running:
```bash
ISAAC_WS=/path/to/workspace bash launch/launch_isaacsim.sh
```

Open two terminals from the project root and run in order:

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
