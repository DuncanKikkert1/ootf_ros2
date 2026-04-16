# ootf_ros2

A ROS2-based pipeline for controlling a Doosan H2017 robot arm using the Octo Vision-Language-Action (VLA) model. Supports three modes: virtual (Isaac Sim), real (physical robot), and end-to-end (Octo VLA inference driving the real robot over TCP).

---

## Architecture

**Virtual mode** (Isaac Sim):
```
tcp_ros_bridge.py   — Listens on TCP port 9000, publishes to /joint_command
    │
    ▼ (/joint_command)
simulation.py       — Isaac Sim simulation, applies joint positions to robot
    │
    ▼ (/joint_states)
Any ROS2 subscriber — Live joint state feedback
```

**Real mode** (physical Doosan robot):
```
tcp_ros_bridge.py       — Listens on TCP port 9000, publishes to /joint_command
    │
    ▼ (/joint_command)
joint_service_client.py — Converts positions (rad→deg), calls /dsr01/motion/move_joint
    │
    ▼ (MoveJoint service)
Doosan H2017            — Physical robot executes movement
```

**End-to-end mode** (Octo VLA → real robot):
```
MechEye camera          — Captures RGB frames
    │
    ▼
run_octo_live.py        — Runs Octo inference, produces 7-DOF EEF delta actions
    │
    ▼ (TCP port 9005)
tcp_ros_bridge.py       — Publishes to /joint_command
    │
    ▼ (/joint_command)
joint_service_client.py — Calls /dsr01/motion/move_joint
    │
    ▼
Doosan H2017
```

---

## Installation

### 1. System requirements
- Ubuntu 22.04 or 24.04
- ROS2 (Humble for Ubuntu 22.04, Jazzy for Ubuntu 24.04)
- Python 3.11 (required by Isaac Sim — use pyenv or a virtual environment)
- NVIDIA GPU (required for Octo/JAX inference)

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
```bash
pip install isaacsim==4.5.0 --extra-index-url https://pypi.nvidia.com
pip install isaacsim-rl isaacsim-replicator isaacsim-extscache-physics isaacsim-extscache-kit isaacsim-extscache-kit-sdk --extra-index-url https://pypi.nvidia.com
```

> **Note:** Replace `4.5.0` with the Isaac Sim version you want. Check available versions at [pypi.nvidia.com](https://pypi.nvidia.com).

### 5. Set up the Isaac Sim ROS2 workspace
```bash
git clone https://github.com/isaac-sim/IsaacSim-ros_workspaces.git ~/IsaacSim-ros_workspaces
cd ~/IsaacSim-ros_workspaces
# Follow the build instructions in that repo for your ROS2 distro
```

### 6. Install Octo
```bash
git clone https://github.com/octo-models/octo.git ~/Documents/octo
pip install -e ~/Documents/octo
pip install "jax[cuda12_pip]==0.4.20" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

### 7. Install the MechEye camera SDK
```bash
pip install MechEyeAPI
```

### 8. (Real mode only) Install the Doosan ROS2 package
```bash
# Follow the installation guide at:
# https://github.com/doosan-robotics/doosan-robot2
```

### 9. Clone this repository
```bash
git clone https://github.com/DuncanKikkert1/ootf_ros2.git
cd ootf_ros2
```

---

## Repository Structure

```
ootf_ros2/
├── launch/
│   ├── launch.sh               # Top-level launcher (virtual, real, or e2e mode)
│   ├── launch_bridge.sh        # Start the TCP/ROS2 bridge
│   ├── launch_isaacsim.sh      # Start Isaac Sim with the robot
│   ├── launch_joint_client.sh  # Start the Doosan joint service client
│   └── launch_octo.sh          # Start the Octo VLA inference loop
├── octo_vla/
│   ├── run_octo_live.py        # Main Octo inference loop
│   ├── octo_policy.py          # Octo model wrapper
│   ├── mech_eye_camera.py      # MechEye 3D camera interface
│   └── tcp_sender.py           # Sends EEF delta actions over TCP
├── scenes/
│   └── h2017/
│       ├── h2017.urdf          # Doosan H2017 robot description
│       ├── meshes_white/       # Visual mesh files (.dae)
│       └── meshes_collision/   # Collision mesh files (.dae)
└── src/
    ├── tcp_ros_bridge.py       # TCP receiver and ROS2 publisher
    ├── joint_service_client.py # Forwards joint commands to Doosan MoveJoint service
    └── simulation.py           # Isaac Sim simulation entry point
```

---

## Usage

Use the top-level `launch.sh` to start the full pipeline in one command:

```bash
# Virtual mode — TCP bridge + Isaac Sim
bash launch/launch.sh virtual

# Real mode — TCP bridge + Doosan joint service client
bash launch/launch.sh real

# End-to-end mode — Octo VLA + TCP bridge + Doosan robot
bash launch/launch.sh e2e --instruction "pick up the circle"

# End-to-end — prompt for instruction at runtime
bash launch/launch.sh e2e

# End-to-end — goal image conditioned
bash launch/launch.sh e2e --goal-image /path/to/goal.png

# End-to-end — USB camera instead of MechEye
bash launch/launch.sh e2e --usb-camera 0 --instruction "pick up the circle"

# End-to-end — dry run (no camera, random noise frames for testing)
bash launch/launch.sh e2e --dry-run --instruction "pick up the circle"

# End-to-end — custom Octo checkpoint
bash launch/launch.sh e2e --model-path /path/to/checkpoint --dataset-name my_dataset \
                           --instruction "pick up the circle"
```

Press `Ctrl+C` to shut everything down cleanly.

If your Isaac Sim ROS2 workspace is in a non-standard location, set `ISAAC_WS` before running:
```bash
ISAAC_WS=/path/to/workspace bash launch/launch.sh virtual
```

---

## TCP Message Formats

### Bridge (port 9000) — joint commands from `tcp_ros_bridge.py`

```
key_cmd;status;extra_num;gripper;x;y;z;aw;ap;ar
```

| Field       | Type  | Description            |
|-------------|-------|------------------------|
| `key_cmd`   | int   | Command code           |
| `status`    | int   | Status code            |
| `extra_num` | int   | Extra parameter        |
| `gripper`   | float | Gripper value          |
| `x`         | float | Joint 1 position (rad) |
| `y`         | float | Joint 2 position (rad) |
| `z`         | float | Joint 3 position (rad) |
| `aw`        | float | Joint 4 position (rad) |
| `ap`        | float | Joint 5 position (rad) |
| `ar`        | float | Joint 6 position (rad) |

Example: `250;254;0;0;0;0;1.57;0;1.57;0`

### Octo output (port 9005) — EEF delta actions from `tcp_sender.py`

```
dx;dy;dz;drx;dry;drz;gripper\n
```

| Field     | Type  | Description                      |
|-----------|-------|----------------------------------|
| `dx`      | float | EEF delta x (metres)             |
| `dy`      | float | EEF delta y (metres)             |
| `dz`      | float | EEF delta z (metres)             |
| `drx`     | float | EEF delta roll (radians)         |
| `dry`     | float | EEF delta pitch (radians)        |
| `drz`     | float | EEF delta yaw (radians)          |
| `gripper` | float | Gripper (0.0 = open, 1.0 = closed) |

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

---

## Finetuning Octo

To finetune on your own robot demonstrations, use the finetuning script in the Octo repo:

```bash
cd ~/Documents/octo
python scripts/finetune.py \
  --config scripts/configs/finetune_config.py \
  --config.pretrained_path="hf://rail-berkeley/octo-small-1.5" \
  --config.dataset_kwargs.name="your_dataset_name" \
  --config.dataset_kwargs.data_dir="/path/to/your/data" \
  --config.save_dir="/path/to/save/checkpoint" \
  --config.modality="language_conditioned"
```

Then point `run_octo_live.py` at your checkpoint:
```bash
bash launch/launch.sh e2e \
  --model-path /path/to/checkpoint \
  --dataset-name your_dataset_name \
  --instruction "pick up the circle"
```

See `~/Documents/octo/scripts/configs/finetune_config.py` for all available options including finetuning mode (`full`, `head_only`, `head_mlp_only`).
