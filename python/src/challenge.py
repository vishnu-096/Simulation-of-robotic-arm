"""
challenge.py
"""
import time
from typing import List, Tuple, Union
import scipy
import numpy as np
import json
from helper_functions import *
from scipy.optimize import minimize
import pygame_gui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

#Class imports
from Robot import Robot, Mapping, Controller
from Visualizer import Visualizer, World
import py_trees
from py_trees.common import Status
import os

def set_working_directory():
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.realpath(__file__))

    # Set the working directory to the script's directory
    os.chdir(script_dir)

# Call the function to set the working directory
set_working_directory()


params_file ='params.json'

# Function to read obstacles from the configuration file
def read_obstacles_from_config(config_file: str) -> Tuple[str, Tuple[float]]:
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config.get("obstacles", [])

"""
Runner class that runs the entire pipeline from creating world to running robot and visualizing everything on Sim
"""

class Runner:
    def __init__(
        self,
        robot: Robot,
        world: World,
        vis: Visualizer
    ) -> None:
        self.robot = robot
        self.world = world
        self.vis = vis

    def run(self) -> None:
        running = True


        # Check success
        success = False
        collision_detected = False
        while running:
            # Check success
            # success = self.check_success(self.robot, self.world.goal)

            start_sim, running = self.vis.update_display(self.robot, success, collision_detected)

            # if success:
            #     self.robot.home_robot()

            # Step the controller if there is no collission
            if start_sim and not collision_detected:
                success, collision_detected = self.robot.behavior_step()

            # sleep for Robot DT seconds, to force update rate
            time.sleep(self.robot.DT)


    @staticmethod
    def check_success(robot: Robot, goal: Tuple[int, int]) -> bool:
        """
        Check that robot's joint 2 is very close to the goal.
        Don't not use exact comparision, to be robust to floating point calculations.
        """
        return np.allclose(robot.forward(robot.joint_angles, 3), goal, atol=0.5)

    def cleanup(self) -> None:
        self.vis.cleanup()


def generate_random_goal(min_radius: float, max_radius: float) -> Tuple[int, int]:
    """
    Generate a random goal that is reachable by the robot arm
    """
    # Ensure theta is not 0
    theta = (np.random.random() + np.finfo(float).eps) * 2 * np.pi
    # Ensure point is reachable
    r = np.random.uniform(low=min_radius, high=max_radius)

    x = int(r * np.cos(theta))
    y = int(r * np.sin(theta))

    return x, y


def main() -> None:
    screen_height = 2200
    screen_width = 2200

    world_width = 600.0
    world_height = 600.0
    robot_origin = (int(world_width/2), int(world_height/2)) # with respect to world

    goal=(130.0, 20.0) # with respect to robot's origin Can change from GUI
    robot = Robot(params_file)
    world = World(world_width, world_height, robot_origin, goal, robot)
    world.goal = goal

    robot.add_behavior(2)

    # Add Create and add obstacle into the world
    world.add_obstacle("circle",(20.0,100.0,15.0))
    world.add_obstacle("circle",(-20.0,60.0,15.0))
    obstacles = read_obstacles_from_config(params_file)

    # Add obstacles to the world
    for obstacle in obstacles:
        obstacle_type = obstacle["type"]
        dimensions = tuple(obstacle["dimensions"])
        world.add_obstacle(obstacle_type, dimensions)

    # Generate landmarks in the world using obstacles and goals
    true_world_landmarks = world.generate_landmarks()

    vis = Visualizer(world, (screen_width, screen_height))

    runner = Runner(robot, world, vis)

    try:
        runner.run()
    except AssertionError as e:
        print(f'ERROR: {e}, Aborting.')
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    main()
