"""
challenge.py
"""
import time
from typing import List, Tuple, Union
import scipy
import numpy as np
import pygame
from helper_functions import *
from scipy.optimize import minimize
import pygame_gui
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

#Class imports
from Robot import Robot, Mapping, Controller
from Visualizer import Visualizer, World


"""
Runner class that runs the entire pipeline from creating world to running robot and visualizing everything on Sim
"""

class Runner:
    def __init__(
        self,
        robot: Robot,
        controller: Controller,
        world: World,
        vis: Visualizer
    ) -> None:
        self.robot = robot
        self.controller = controller
        self.world = world
        self.vis = vis

    def run(self) -> None:
        running = True

        collision_detected = False

        # Check success
        success = False
        collission_detected = False
        while running:
            # Check success
            # success = self.check_success(self.robot, self.world.goal)
            
            start_sim, running = self.vis.update_display(self.robot, success, collision_detected)


            # Step the controller if there is no collission
            if start_sim:
                collision_detected = self.world.check_collisions(self.robot)

                if not collision_detected :
                    # self.robot = self.controller.step(self.robot)
                    success = self.robot.behavior_step()

                # Check collisions



            # Update the display

            # sleep for Robot DT seconds, to force update rate
            time.sleep(self.robot.DT)

            # time.sleep(5)

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

    world_width = 600
    world_height = 600
    robot_origin = (int(world_width/2), int(world_height/2)) # with respect to world
    num_goals=1
    theta=0
    for iter in range(num_goals):

        # goal = generate_random_goal(Robot.min_reachable_radius(), Robot.max_reachable_radius())
        # goal = (math.cos(theta)*Robot.min_reachable_radius(), math.sin(theta)*Robot.min_reachable_radius())
        goal=(25.0,50.0) # with respect to robot's origin
        robot = Robot()
        robot.add_behavior(1, goal)
        # controller = Controller(goal)
        world = World(world_width, world_height, robot_origin, goal, robot)

        # Add Create and add obstacle into the world
        world.add_obstacle("circle",(30.0,60.0,15.0))
        world.add_obstacle("circle",(100.0,60.0,25.0))
        # Generate landmarks in the world using obstacles and goals
        true_world_landmarks = world.generate_landmarks()

        vis = Visualizer(world, (screen_width, screen_height))

        runner = Runner(robot, controller, world, vis)

        try:
            runner.run()
        except AssertionError as e:
            print(f'ERROR: {e}, Aborting.')
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
