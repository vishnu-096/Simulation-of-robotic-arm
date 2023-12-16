
# Robot Simulation Challenge

This simulation environment provides a realistic representation of 3 link planar robotic arm operation in a 2D world. It includes advanced features such as sensor simulation, real-time map display, a robust behavior tree, and a shareable logging module, all organized to closely mimic real-world robot operation.

## Table of Contents
- [Getting Started](#getting-started)
- [Key Features](#key-features)
- [Code Organization](#code-organization)
- [Behavior Tree and Actions](#behavior-tree-and-actions)
- [Visualization](#visualization)
- [Logging Module](#logging-module)
- [Usage](#usage)
- [Customization](#customization)
- [Dependencies](#dependencies)
- [Contributing](#contributing)
- [License](#license)

## Getting Started

To view the simulation:

1. Clone the repository.
2. cd into python directory
3. Setup environment $ ./scripts/bootstrap
4. Enter into virtual environment $ python src/challenge.py
2. Install the required dependencies (see [Dependencies](#dependencies)).
3. Run 'python challenge.py'.
4. Use the GUI to interact with the simulation.

## Key Features

### Inverse Kinematics and Control
The analytical inverse kinematics caused problems during unit testing (especially issues with angles and reaching singularities quickly).
Therefore I add differential inverse kinematics based on Jacobian. Also, the calculated target joint states are reached by using a PD controller.

### Sensor Simulation
The simulation includes a realistic sensor model that mimics a robot's sensing capabilities. The sensor parameters, such as radius and field of view (FOV), can be customized via the `params.json` file.

### Real-time Map Display

map is displayed in the GUI, showcasing scanned regions, landmarks, and the robot's planned and executed paths. This visual representation allows users to observe the robot's environment and its interactions in real-time.

### Behavior Tree and Actions
The behavior tree, implemented using the PyTrees library, provides a flexible and modular structure for defining and executing robot behaviors. The 'robot_actions.py' file contains actions such as collision detection and moving to a goal. These actions are easily extendable and reusable, allowing for the creation of complex behaviors.
Now, collision detection, plan path and move-to-goal actions are added sequentially and stepped through in the behavior tree.

### Path planning
Used basic A star path planning based on the grid map generated. Spline interpolation used to smoothen trajectory for easier traversal by robotic arm

### Logging Module
A shareable logging module is integrated into the codebase, providing detailed logs of the simulation process. The logging level can be configured in 'logging_config.py', allowing users to customize the amount of information logged. This modular logging setup enhances code readability, debuggability, and shareability.

## Code Organization

The code is organized in a clear and modular manner, resembling the structure of real-world robotic systems. The main components include:

- 'params.json' : All parameters related robot, sensor and adding obstacles can be specified here and modified seamlessly
- 'challenge.py': The main script to run the simulation.
- 'robot_actions.py': Contains action classes for collision detection and moving to a goal.
- 'visualizer.py': Manages the GUI and visualization of the simulation.
- 'world.py': Defines the world and obstacles.
- 'Robot.py': Implements the robot's kinematics, controller, and behavior tree.

## Visualization

The GUI provided by 'pygame' serves as the primary visualization tool. It displays the world, robot, sensor map, and real-time plots of joint angles and velocities. The visualization is designed for ease of understanding and monitoring.

## Logging Module

The logging module, configured in 'logging_config.py', enables detailed logging of the simulation process. Different log levels (e.g., DEBUG, INFO, ERROR) provide flexibility in the amount of information displayed. Logs are instrumental for debugging and analyzing the robot's behavior during simulation.

## Usage

- Run the simulation using 'python challenge.py' from python/src.
- Use the GUI to interact with the robot, set goals, and start or stop the simulation.
- Monitor real-time plots and sensor maps to gain insights into the robot's behavior.
- Demo Gif provided for reference

## Customization

The 'params.json' file allows users to customize various simulation parameters, such as sensor characteristics, obstacle placement, and robot behavior. This flexibility encourages experimentation and testing under different conditions.

## Dependencies

Ensure the following dependencies are installed:

- 'pygame'
- 'pygame_gui'
- 'matplotlib'
- 'numpy'
- 'scipy'
- 'py_trees'

Install dependencies using:
pip install -r requirements.txt

## IMPROVEMENTS
- Minor changes to use 3D robotic arm. 
- Add noise in joint position measurement and use sensor reading to localize end effector
- Add artificial potential field based obstacle avoidance (IN PROGRESS)
- Better GUI interaction with "Home Robot" button and behavior selection. Right now default behavior is "PLAN_PATH_MOVE_2_GOAL".
- Control velocity and add constrained inverse kinematics solver (IN PROGRESS)
