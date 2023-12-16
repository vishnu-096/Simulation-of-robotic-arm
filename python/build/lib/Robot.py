import time
from typing import List, Tuple, Union
import scipy
import numpy as np
import math

from helper_functions import calculate_distance


MOVE_2_GOAL = 1

"""
Simulation of a sensor with 180-degree field of view and defined sensor radius.
This is used to populate points in the grid map of the robot.
"""
class Sensor:
    def __init__(self) -> None:
        # Sensor properties
        self.sensor_radius = 20
        self.landmarks = None  # Landmarks with respect to the robot's frame. Only used as a true reference during perception
        self._sensor_position: List[float] = [0., 0.]
        self.sensor_orientation = 0
        self.sensor_fov = np.pi/2
        self.sensed_frontier = set()  # Unique frontiers x and y positions in the robot's frame detected by the sensor

    @property
    def sensor_position(self) -> List[float]:
        """Getter for sensor position."""
        return self._sensor_position

    @sensor_position.setter
    def sensor_position(self, values: List[float]) -> None:
        """
        Setter for sensor position.

        Args:
            values (List[float]): List containing x and y coordinates of the sensor.
        """
        # Additional logic, validation, or constraints can be added here
        self._sensor_position = values
        if values[0] == 0:
            self.sensor_orientation = math.pi / 2  # sensor orientation is also set in the same position setter
        self.sensor_orientation = math.atan2(values[1], values[0])

    def perceive_environment(self):
        """
        The sensor is assumed to sense in a sector of sensor_fov angle area with the given sensory radius.
        Populates sensed_frontier with points in the FOV.
        """
        for landmark in self.landmarks:
            x, y, *rest = landmark
            if not rest:
                continue
            else:
                r = rest[0]

            angle_s = self.sensor_orientation
            
            # Transformation matrix to sensor frame
            T = np.array([
                [np.cos(angle_s), -np.sin(angle_s), self.sensor_position[0]],
                [np.sin(angle_s), np.cos(angle_s), self.sensor_position[1]],
                [0, 0, 1]
            ])
            
            T_inv = np.linalg.inv(T)

            angles = np.linspace(0, 2 * np.pi, 360)
            starting_angle = -self.sensor_fov / 2
            ending_angle = self.sensor_fov / 2

            for theta in angles:
                point = np.array([x + r * math.cos(theta),
                                  y + r * math.sin(theta),
                                  1])
                #land mark point expressed based on sensor coordinate frame
                transformed_point = np.dot(T_inv, point)
                new_angle = math.atan2(transformed_point[1], transformed_point[0])

                dist = calculate_distance(point[0], point[1], self.sensor_position[0], self.sensor_position[1])

                # Logic for checking whether landmark point falls inside the sensing sector of the sensor
                if dist < self.sensor_radius and starting_angle < new_angle < ending_angle:
                    sensed_point = (int(point[0]), int(point[1]))
                    self.sensed_frontier.add(sensed_point)


"""
Local grid map class for the robot. Generated from the sensor-based perception.
"""
class Mapping:
    def __init__(self, sensor) -> None:
        # Map properties
        self.map_width = 400
        self.map_height = 400
        # x becomes rows, and y is columns. Convert such that the robot is in the center
        self.grid2D = np.zeros((self.map_height, self.map_width))
        self.sensor_scans = set()
        self.landmark_scans = set()
        self.new_landmarks_grid = []
        self.new_scan_grid = []
        self.sensor = sensor

    def update_map(self):
        """
        Updates the grid map (grid2D) with information from the sensor's perception.
        Populates new_landmarks_grid and new_scan_grid with grid coordinates.
        """
        print("frontiers of landmarks : x")
        self.new_landmarks_grid = []
        self.new_scan_grid = []

        for point in self.sensor.sensed_frontier:
            x, y = point
            x_grid = -int(y - self.map_height / 2)
            y_grid = int(x + self.map_width / 2)
            self.new_landmarks_grid.append([x_grid, y_grid])
            self.grid2D[x_grid, y_grid] = 2
            self.landmark_scans.add((x_grid, y_grid))

        # the sector of angular region based on sensor fov
        angles = np.linspace(self.sensor.sensor_orientation - self.sensor.sensor_fov / 2,
                             self.sensor.sensor_orientation + self.sensor.sensor_fov / 2, 180)
        x_s, y_s = self.sensor.sensor_position


        for r in range(0, self.sensor.sensor_radius, 1):
            for angle in angles:
                x_grid = -int(y_s + r * math.sin(angle) - self.map_height / 2)
                y_grid = int(x_s + r * math.cos(angle) + self.map_width / 2)

                if self.grid2D[x_grid, y_grid] != 2:
                    self.grid2D[x_grid, y_grid] = 1
                self.new_scan_grid.append([x_grid, y_grid])
                self.sensor_scans.add((x_grid, y_grid))

        sensor_grid_x = int(self.map_height // 2 - self.sensor.sensor_position[1])
        sensor_grid_y = int(self.sensor.sensor_position[0] + self.map_width // 2)
        self.grid2D[sensor_grid_x, sensor_grid_y] = 3


"""
Robot class for the simulation.
"""
class Robot:
    JOINT_LIMITS = [-6.28, 6.28]
    MAX_VELOCITY = 15
    MAX_ACCELERATION = 50
    DT = 0.033

    link_1: float = 75.  # pixels
    link_2: float = 50.  # pixels
    link_3: float = 25.  # pixels

    link_lengths = [link_1, link_2, link_3]
    analytical_ik = True

    def __init__(self, controller) -> None:
        # Internal variables

        self.all_angles: List[List[float]] = [[], [], []]

        self.joint_angles = [0., 0., 0.]
        self._joint_angles: List[float] = [0., 0., 0.]
        self.eof_pos = self.forward(self.joint_angles, 3)
        self.link_lengths = [self.link_1, self.link_2, self.link_3]
        self.map = None
        self.local_goal = self.eof_pos

        # Run Robot Init Procedure
        self.init_all_modules()

    @property
    def joint_angles(self) -> List[float]:
        """Getter for joint angles."""
        return self._joint_angles


    @joint_angles.setter
    def joint_angles(self, values: List[float]) -> None:
        """
        Setter for joint angles.

        Args:
            values (List[float]): List containing joint angles for the robot.
        """
        # Additional logic, validation, or constraints can be added here
        # For example, checking if the length of the provided list is 3
        assert len(values) == 3, "Joint angles must be a list of length 3"

        for iter, angle in enumerate(values):
            self.all_angles[iter].append(angle)
        for iter, angle in enumerate(values):
            assert self.check_angle_limits(angle), \
                f'Joint 0 value {angle} exceeds joint limits'
            assert self.max_velocity(self.all_angles[iter]) < self.MAX_VELOCITY, \
                f'Joint 0 Velocity {self.max_velocity(self.all_angles[iter])} exceeds velocity limit'
            assert self.max_acceleration(self.all_angles[iter]) < self.MAX_ACCELERATION, \
                f'Joint 0 Accel {self.max_acceleration(self.all_angles[iter])} exceeds acceleration limit'

        # Set the internal _joint_angles attribute
        self._joint_angles = values


    def init_all_modules(self):
        """Initialize all modules of the robot."""
        # Perception Module
        robot_sensor = Sensor()
        self.map = Mapping(robot_sensor)  # Local mapping module of the robot
        self.controller = Controller()

    def update_true_landmarks(self, lds: List[List[Union[float, int]]]):
        """
        Update true landmarks in the robot's perception module.

        Args:
            lds (List[List[Union[float, int]]]): List of landmarks with x, y, and radius.
        """
        self.map.sensor.landmarks = lds

    def percieve_landmarks(self):
        """Perceive landmarks using the robot's sensor."""
        self.map.sensor.perceive_environment()
        self.map.update_map()
        print("Sensor Orientation", np.rad2deg(self.map.sensor.sensor_orientation))
        print("Sensor Position", self.map.sensor.sensor_position)



    @classmethod
    def forward(cls, angles: List[float], joint_num: int) -> Tuple[float, float]:
        """
        Compute the x, y position of the end of the links from the joint angles. Forward Kinemetics

        Args:
            angles (List[float]): List of joint angles.
            joint_num (int): Number of joints to consider.

        Returns:
            Tuple[float, float]: x, y position of the end of the links.
        """
        x = 0
        y = 0
        theta = 0
        for iter in range(joint_num):
            theta += angles[iter]
            x += cls.link_lengths[iter] * np.cos(theta)
            y += cls.link_lengths[iter] * np.sin(theta)
        return x, y

    @classmethod
    def inverse(cls, x: float, y: float, psi: float) -> Tuple[float, float, float]:
        """
        Compute the joint angles from the position of the end of the links.

        Args:
            x (float): x position of eof.
            y (float): y position of eof.
            psi (float): orientation angle value of eof (radians).

        Returns:
            Tuple[float, float]: Joint angles.
        """
        if cls.analytical_ik:
            x2 = x - cls.link_3 * math.cos(psi)
            y2 = y - cls.link_3 * math.sin(psi)

            theta_1 = np.arccos((x2 ** 2 + y2 ** 2 - cls.link_1 ** 2 - cls.link_2 ** 2) /
                                (2 * cls.link_1 * cls.link_2))
            theta_0 = np.arctan2(y2, x2) - \
                np.arctan((cls.link_2 * np.sin(theta_1)) / (cls.link_1 + cls.link_2 * np.cos(theta_1)))
            theta_2 = psi - theta_0 - theta_1
            return theta_0, theta_1, theta_2
        else:
            pass
            # Implement numerical IK if needed



    def add_behavior(self, behavior, arg):
        
        self.behavior = behavior
        if self.behavior == MOVE_2_GOAL:
            self.local_goal = arg
            eof_orientation = math.atan2(self.local_goal[1], self.local_goal[0])
            print("eof Orientation ", eof_orientation)
            self.target_state = np.array(Robot.inverse(self.local_goal[0], self.local_goal[1],
                        eof_orientation))
            # self.state =  np.array(self.joint_angles)

    def behavior_terminate(self)->bool:
        
        if self.behavior == MOVE_2_GOAL:
            return    np.allclose(self.forward(self.joint_angles, 3), self.local_goal, atol=0.5)
        
    def behavior_step(self):
        if not behavior_terminate:
            
            robot.eof_pos = robot.forward(robot.joint_angles, 3)
            robot.map.sensor.sensor_position = robot.eof_pos
            robot.percieve_landmarks()
            self.joint_angles = self.controller.step(self.joint_angles, self.target_state)
            # self.state = np.array(self.joint_angles)
            
            # time.sleep(1)            

        else:
            print("Behavior has been completed")
            
        return behavior_terminate

    @classmethod
    def check_angle_limits(cls, theta: float) -> bool:
        """
        Check if the given joint angle is within limits.

        Args:
            theta (float): Joint angle.

        Returns:
            bool: True if within limits, False otherwise.
        """
        return cls.JOINT_LIMITS[0] < theta < cls.JOINT_LIMITS[1]

    @classmethod
    def max_velocity(cls, all_theta: List[float]) -> float:
        """
        Calculate the maximum velocity from a list of joint angles.

        Args:
            all_theta (List[float]): List of joint angles.

        Returns:
            float: Maximum velocity.
        """
        return float(max(abs(np.diff(all_theta) / cls.DT), default=0.))

    @classmethod
    def max_acceleration(cls, all_theta: List[float]) -> float:
        """
        Calculate the maximum acceleration from a list of joint angles.

        Args:
            all_theta (List[float]): List of joint angles.

        Returns:
            float: Maximum acceleration.
        """
        return float(max(abs(np.diff(np.diff(all_theta)) / cls.DT / cls.DT), default=0.))

    @classmethod
    def min_reachable_radius(cls) -> float:
        """
        Calculate the minimum reachable radius.

        Returns:
            float: Minimum reachable radius.
        """
        return max(cls.link_1 - cls.link_2, 0)

    @classmethod
    def max_reachable_radius(cls) -> float:
        """
        Calculate the maximum reachable radius.

        Returns:
            float: Maximum reachable radius.
        """
        return cls.link_1 + cls.link_2

    def is_goal_reached(self, goal: Tuple[int, int]) -> bool:
        return self.eof_pos == goal

class Controller:
    def __init__(self) -> None:
        """
        Initialize the Controller with a goal.

        Args:
            goal (Tuple[int, int]): Goal position (x, y).
        """
        self.Kp = 0.1
        self.dt = 0.1

    
    def step(self, state, desired_state) -> List[]:
        """
        Perform a step in the control loop.

        Args:
            robot (Robot): The robot to control.

        Returns:
            Robot: Updated robot.
        """
        # Simple P controller
        theta_error = desired_state - np.array(state)
        return (state+theta_error*self.Kp).tolist()
        # robot.joint_angles = (np.array(robot.joint_angles) + theta_error / 10).tolist()
        # robot.eof_pos = robot.forward(robot.joint_angles, 3)
        # robot.map.sensor.sensor_position = robot.eof_pos
        # robot.percieve_landmarks()
        # time.sleep(1)
        # return robot

