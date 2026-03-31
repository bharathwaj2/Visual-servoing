# Visual Servoing using AgileX PIPER Arm 

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue.svg)](https://docs.ros.org/en/humble/index.html)

ROS2 packages for **Image-Based (IBVS)** and **Position-Based (PBVS)** visual servoing on the **AgileX PIPER Arm** (7DoF manipulator), supporting both **Eye-in-Hand (EIH)** and **Eye-to-Hand (ETH)** camera configurations. Includes a minimal **Reinforcement Learning (RL)** environment using DDPG for IBVS gain tuning with jerk minimization.

## Features

- **Classical Visual Servoing**:
  - IBVS: 2D image features (ArUco marker corners) + image Jacobian
  - PBVS: 6D pose features + interaction matrix
- **Camera Configurations**:
  - EIH: Intel Realsense D435 on end-effector (`piper_arm_EIH.xacro`)
  - ETH: Intel Realsense D435 in fixed point near base-link (`piper_arm_ETH.xacro`)
- **Simulation & Real Robot**:
  - Gazebo simulation with RealSense plugin
  - MoveIt Servo for real-time control
  - Real robot support (URDF + controllers)
- **RL for IBVS Tuning**:
  - Gymnasium env for DDPG (PyTorch)
  - State: pixel error + cam vel + joints
  - Rewards: step penalty + jerk norm + success bonus
  - Trained models in `src/DDPG/Trained_models/`


## Repository Structure

```
src/
└── piper_arm/
    ├── piper_arm/                 # Core meta package
    ├── piper_arm_description/     # URDFs (sim/real, EIH/ETH)
    │   ├── urdf/piper_arm_EIH.xacro     # EIH
    │   └── urdf/piper_arm_ETH.xacro     # ETH
    |   └── urdf/piper_arm_real
    ├── piper_arm_bringup/         # Gazebo/RViz/base launches
    ├── piper_arm_moveit_config/   # MoveIt config + servo
    ├── realsense_gazebo_plugin/   # RealSense D435 Gazebo sim
    └── visual_servoing/           # VS implementation ⭐
        ├── src/
        │   ├── IBVS.py             # Classical IBVS
        │   ├── PBVS.py             # Classical PBVS
        │   ├── IBVS_RL.py          # RL-tuned IBVS (DDPG)
        │   ├── HEC.py              # Hand-Eye matrix calibration for real time implementation
        ├── src/DDPG/               # RL agent (PyTorch)
        │   ├── ddpg_torch.py
        │   └── networks.py
        └── launch/
            ├── ibvs_fake_env.launch.py    # IBVS sim
            ├── pbvs_real_env.launch.py    # PBVS real
            └── vs.launch.py               # Generic
```

## Prerequisites

- **ROS2 Humble** (Ubuntu 22.04)
- Python 3.10+ (OpenCV, NumPy, SciPy, PyTorch, Gymnasium)
- Colcon build tool

```bash
sudo apt install ros-humble-moveit ros-humble-gazebo-ros-pkgs
pip install torch gymnasium opencv-python scipy tf2-ros
```

## Installation

```bash
cd /home/bharath/visual_servoing
colcon build --packages-select piper_arm visual_servoing
source install/setup.bash
```

## Quick Start: IBVS Simulation (Eye-in-Hand)

1. **Launch Gazebo + IBVS**:
   ```bash
   ros2 launch visual_servoing ibvs_fake_env.launch.py
   ```

2. **RViz visualization** (optional):
   ```bash
   ros2 launch piper_arm_bringup rviz.launch.py
   ```

## Usage Examples

### 1. IBVS Simulation (EIH)
```
ros2 launch visual_servoing ibvs_fake_env.launch.py
```
- Sets initial pose, detects ArUco (6x6_250), converges pixel error using avg image Jacobian (`IBVS.py`).

### 2. PBVS Real Robot (EIH)
```
ros2 launch visual_servoing pbvs_real_env.launch.py
```
- Uses ArUco pose estimation + interaction matrix (`PBVS.py`).
- Can be done in both ETH and EIH setups.

### 3. RL-Tuned IBVS Simulation
```
ros2 launch visual_servoing vs.launch.py  # Set i=3 for IBVS_RL.py
```
- DDPG tunes interaction matrix gains dynamically.
- Train: 100 episodes, success >90% saves models.
- Rewards: `-step - (err/max_steps) + success(T + jerk_norm)`.
- Reward function inspired from the paper, [A Motion Planning Method for Visual Servoing Using Deep Reinforcement Learning in Autonomous Robotic Assembly](https://ieeexplore.ieee.org/abstract/document/10138316)


### 4. Generic Script
```
ros2 launch visual_servoing vs.launch.py  # i=0:PBVS, 1:IBVS, 2:HEC, 3:IBVS_RL
```
- Launch `HEC.py` for computing Hand-Eye calibration matrix in real-time for Eye-in-Hand setup.
- Launch `IBVS.py` for simulation environment Image-based Visual Servoing.
- Launch `PBVS.py` for real-world Position-based Visual Servoing.
- Launch `IBVS_RL.py` for simulation environment IBVS RL training.
- Launch `Generate_valid_poses.py` for recording valid initial arm poses and target features for IBVS RL training(simulation). 
      NOTE: Add a teleoperation package to create new valid poses if required. A pre-existing set of poses are already provided by us.

### 5. Results
  - The simulation and real-world implementation videos can be found in src/Results/
  - Both IBVS and PBVS was demonstrated in Eye-in-Hand setup.

## Control Loop Details

- **Topics**:
  - Sub: `/camera/color/image_raw`, `/camera/aligned_depth_to_color/image_raw`, `/joint_states`
  - Pub: `/servo_node/delta_twist_cmds` (TwistStamped, EE frame)
- **TF**: `end_effector_link` → `camera_color_optical_frame` (hand-eye calibration)
- **ArUco**: DICT_6X6_250, relaxed detection params.
- **Convergence**: Pixel err <5px (IBVS), pose err <2cm/10° (PBVS).

## RL Environment

- **State** (20D): pixel err(8) + cam vel(6) + joints(6)
- **Action** (6D): diagonal gains for `L^+` pseudo-inverse (±0.5)
- **Termination**: goal, timeout(500 steps), limits, no marker
- **Pre-trained**: `src/DDPG/Trained_models/`

## Troubleshooting

- **No marker**: Adjust `minMarkerPerimeterRate` in ArUco params
- **No movement**: Check for proper camera topics subscription
- **Servo fails**: Ensure MoveIt Servo running (`ros2 service call /servo_node/start_servo`)
- **TF errors**: Check hand-eye matrix (`Files/Hand_Eye_matrix_*.npy`)
- **Gazebo camera**: Verify `realsense_gazebo_plugin`


## Acknowledgments

- [PIPER Arm ROS2](https://github.com/agilexrobotics/piper_ros)
- [MoveIt Servo](https://moveit.picknik.ai/humble/doc/examples/moveit_servo/moveit_servo_setup.html)
- [RealSense Gazebo Plugin](https://github.com/intel-ros/realsense_gazebo_plugin)

**Contact**: bharathwaj019@gmail.com

