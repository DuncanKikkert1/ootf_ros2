# ootf_ros2

A ROS2-based pipeline for controlling a Doosan H2017 robot arm — either in NVIDIA Isaac Sim (virtual mode) or on the physical robot (real mode) — via TCP commands. Joint positions are received over TCP, validated, and published to a ROS2 topic.

---

## Architecture

**Virtual mode** (Isaac Sim):
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

**Real mode** (physical Doosan robot):
```
TCP Client
    │
    ▼
tcp_ros_bridge.py       — Validates and parses TCP messages, publishes to /joint_command
    │
    ▼ (/joint_command)
joint_service_client.py — Converts positions (rad→deg), calls /dsr01/motion/move_joint
    │
    ▼ (MoveJoint service)
Doosan H2017            — Physical robot executes movement
```

---

## Installation

### 1. System requirements
- Ubuntu 22.04 or 24.04
- ROS2 (Humble for Ubuntu 22.04, Jazzy for Ubuntu 24.04)
- Python 3.11 (required by Isaac Sim — use pyenv or a virtual environment)

### 2. Install ROS2
Follow the official guide for your distro:
- [ROS2 Humble (Ubuntu 22.04)](https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html)
- [ROS2 Jazzy (Ubuntu 24.04)](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html)

### 3. Install Python 3.11 (for Isaac Sim)
Isaac Sim requires Python 3.11. If your system Python differs, use pyenv:
```bash
pyenv install 3.11.9
pyenv global 3.11.9
```

### 4. Install NVIDIA Isaac Sim
Install via pip into your Python 3.11 environment:
```bash
pip install isaacsim==4.5.0 --extra-index-url https://pypi.nvidia.com
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit isaacsim-extscache-kit-sdk --extra-index-url https://pypi.nvidia.com
```

> **Note:** Replace `4.5.0` with the Isaac Sim version you want to use. Check available versions at [pypi.nvidia.com](https://pypi.nvidia.com).

### 5. Set up the Isaac Sim ROS2 workspace
Clone and build the Isaac Sim ROS2 workspace for your distro:
```bash
git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git ~/IsaacSim-ros_workspaces
cd ~/IsaacSim-ros_workspaces
# Follow the build instructions in that repo for your ROS2 distro
```

### 6. (Real mode only) Install the Doosan ROS2 package
Install the `dsr_msgs2` package for the Doosan robot service interfaces:
```bash
# Follow the installation guide at:
# https://github.com/doosan-robotics/doosan-robot2
```

### 7. Clone this repository
```bash
git clone https://github.com/DuncanKikkert1/ootf_ros2.git
cd ootf_ros2
```

---

## Repository Structure

```
ootf_ros2/
├── launch/
│   ├── launch.sh               # Top-level launcher (virtual or real mode)
│   ├── launch_isaacsim.sh      # Start Isaac Sim with the robot
│   ├── launch_bridge.sh        # Start the TCP/ROS2 bridge
│   └── launch_joint_client.sh  # Start the Doosan joint service client
├── scenes/
│   └── h2017/
│       ├── h2017.urdf          # Doosan H2017 robot description
│       ├── meshes_white/       # Visual mesh files (.dae)
│       └── meshes_collision/   # Collision mesh files (.dae)
└── src/
    ├── simulation.py           # Isaac Sim simulation entry point
    ├── tcp_ros_bridge.py       # Combined TCP receiver and ROS2 publisher
    └── joint_service_client.py # Forwards joint commands to the Doosan MoveJoint service
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

Use the top-level `launch.sh` to start the pipeline in one command:

```bash
bash launch/launch.sh virtual   # TCP bridge + Isaac Sim
bash launch/launch.sh real      # TCP bridge + Doosan joint service client
```

Both programs start in parallel. Press `Ctrl+C` to shut both down cleanly.

If your Isaac Sim ROS2 workspace is in a non-standard location, set `ISAAC_WS` before running:
```bash
ISAAC_WS=/path/to/workspace bash launch/launch.sh virtual
```

Then connect a TCP client to port `9000` on the machine's IP and start sending commands in the format described above.

---

## Verifying the Pipeline

Check that joint commands are being received:
```bash
ros2 topic echo /joint_command
```

Check that joint states are being published (virtual mode only):
```bash
ros2 topic echo /joint_states
```
