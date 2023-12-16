import time
import json
from typing import List, Tuple, Union
from scipy.interpolate import CubicSpline
import numpy as np
import math
import py_trees
import logging
from py_trees.common import Status

from helper_functions import calculate_distance, distance_to_line_segment, Jacobian, forward_temp, astar_search
from robot_actions import MoveToGoalAction, CollisionDetectionAction
import logging_config
import logging

logging_config.setup_logging()
# Get a logger instance
logger = logging.getLogger(__name__)

MOVE_2_GOAL = 1
PLAN_PATH_MOVE_2_GOAL = 2


# Set the logging level for PyTrees
py_trees.logging.level = logging.CRITICAL  # Set it to the desired level, e.g., logging.INFO, logging.DEBUG, etc.

"""
Simulation of a sensor with desired field of view and sensor radius which can be set
in the params.json file.
This is used to populate points in the grid map of the robot.

"""
class Sensor:
    def __init__(self) -> None:
        # Sensor properties
        self.sensor_radius = 50
        self.landmarks = None  # Landmarks with respect to the robot's frame. Only used as a true reference during perception
        self._sensor_position: List[float] = [0., 0.]
        self.sensor_orientation = 0
        self.sensor_fov = 2*np.pi
        self.sensed_frontier = set()  # Unique frontiers x and y positions in the robot's frame detected by the sensor

    def initialize_from_config(self, config):
        self.sensor_radius = config.get('sensor', {}).get('sensor_radius', 50)
        self.sensor_fov = config.get('sensor', {}).get('sensor_fov', 2 * 3.14159)
        self.sensor_position = config.get('sensor', {}).get('sensor_position', [0.0, 0.0])


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


    """
    Perceive the environment and detect landmarks and detect landmarks using the
    simulated sensor. Adds the points where obstacles where detected to landmark

    using the sensor.

    Args:self:  self: The instance of The instance of the class.


    Returns:None

    """
    def perceive_environment(self):

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

                dist = calculate_distance(point, self.sensor_position)

                # Logic for checking whether landmark point falls inside the sensing sector of the sensor
                if dist < self.sensor_radius and starting_angle < new_angle < ending_angle:
                    sensed_point = (int(point[0]), int(point[1]))
                    self.sensed_frontier.add(sensed_point)


"""
Local grid map class for the robot. Generated from the sensor-based perception.
Map is updated in each perception step executed. Displayed on the GUI

Owned by Robot Class
"""
class Mapping:
    def __init__(self, sensor) -> None:
        # Map properties
        self.map_width = 600
        self.map_height = 600
        # x becomes rows, and y is columns. Convert such that the robot is in the center
        self.grid2D = np.zeros((self.map_height, self.map_width))
        self.landmark_scans = set()  # set of landmark scans (x,y) in robot frame
        self.new_landmarks_grid = [] #list of [x,y] which are based on grid coordinates
        self.new_scan_grid = []
        self.sensor = sensor
        
    def update_map(self):
        self.new_landmarks_grid = []
        self.new_scan_grid = []

        for point in self.sensor.sensed_frontier:
            x, y = point
            x_grid = -int(y - self.map_height / 2)
            y_grid = int(x + self.map_width / 2)
            self.new_landmarks_grid.append([x_grid, y_grid])
            self.grid2D[x_grid, y_grid] = 2
            self.landmark_scans.add((x, y))

        # the sector of angular region based on sensor fov
        angles = np.linspace(self.sensor.sensor_orientation - self.sensor.sensor_fov / 2,
                             self.sensor.sensor_orientation + self.sensor.sensor_fov / 2, 180)
        x_s, y_s = self.sensor.sensor_position


        for r in range(self.sensor.sensor_radius):
            for angle in angles:
                x_grid = -int(y_s + r * math.sin(angle) - self.map_height / 2)
                y_grid = int(x_s + r * math.cos(angle) + self.map_width / 2)

                if self.grid2D[x_grid, y_grid] != 2:
                    self.grid2D[x_grid, y_grid] = 1
                self.new_scan_grid.append([x_grid, y_grid])

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
    DT = 0.1

    link_1: float = 75.  # pixels
    link_2: float = 50.  # pixels
    link_3: float = 25.  # pixels

    link_lengths = [link_1, link_2, link_3]
    analytical_ik = True
    is_3d = False
    
    def __init__(self, params_file) -> None:

        #Parameter config file path
        self.params_file = params_file
        # Internal variables
        self.all_angles: List[List[float]] = [[], [], []]
        self.all_joint_velocities: List[List[float]] = []
        self.eof_pos = [0.0, 0.0]


        self._joint_angles: List[float] = [0.0, 0.0, 0.0]
        self.joint_angles: List[float] = [0.0, 0.0, 0.0]
        self._joint_velocity: List[float]=[0.0, 0.0, 0.0]


        self.link_lengths = [self.link_1, self.link_2, self.link_3]
        self.map = None

        self.behavior_done =False
        self.behavior_option = None

        self.global_goal = []
        self.local_goal = self.eof_pos
        self.local_goal_iter = 0
        self.step_iterations = 0
        self.reached_goal = False
        self.planned_path = None
        self.interpolated_path = None
        self.not_collided = True
        self.threshold = 1.5  # Collission threshold

        # Run Robot Init Procedure
        self.init_all_modules()
        self.blackboard = None
    
    @classmethod
    def load_config_class_variables(cls):
        with open(cls.params_file, 'r') as file:
            config = json.load(file)

        cls.link_lengths = config.get('link_lengths')
        cls.analytical_ik = config.get('analytical_ik', True)
        cls.is_3d = config.get('is_3d', False)
        cls.JOINT_LIMITS = config.get('robot', {}).get('JOINT_LIMITS', [-6.28, 6.28])
        cls.MAX_VELOCITY = config.get('robot', {}).get('MAX_VELOCITY', 15)
        cls.MAX_ACCELERATION = config.get('robot', {}).get('MAX_ACCELERATION', 50)
        cls.DT = config.get('robot', {}).get('DT', 0.1)

    def load_config_instance_variables(self):
        with open(self.params_file, 'r') as file:
            config = json.load(file)        
        self.joint_angles = config.get('joint_angles')
        self.eof_pos = self.forward(self.joint_angles, 3)
                        
    @property
    def joint_angles(self) -> List[float]:
        """Getter for joint angles."""
        return self._joint_angles

    @property
    def joint_velocity(self) -> List[float]:
        """Getter for joint velocity."""
        return self._joint_velocity


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
            assert self.check_angle_limits(angle), \
                f'Joint 0 value {angle} exceeds joint limits'
            assert self.max_velocity(self.all_angles[iter]) < self.MAX_VELOCITY, \
                f'Joint 0 Velocity {self.max_velocity(self.all_angles[iter])} exceeds velocity limit'
            assert self.max_acceleration(self.all_angles[iter]) < self.MAX_ACCELERATION, \
                f'Joint 0 Accel {self.max_acceleration(self.all_angles[iter])} exceeds acceleration limit'
            self.all_angles[iter].append(angle)

        #Setting eof position
        self.eof_pos = self.forward(self.joint_angles, 3)

        #finding velocity
        self._joint_velocity = ((np.array(values) - np.array(self._joint_angles))/self.DT).tolist()

        # Adding velocity to all joint velocity TODO remove old velocity after sometime to reduce memory consumption
        self.all_joint_velocities.append(self._joint_velocity)

        # Set the internal _joint_angles attribute
        self._joint_angles = values


    def init_all_modules(self):
        """Initialize all modules of the robot."""
        #Initialize remaining variables of robot instance
        self.load_config_instance_variables()
        # Perception Module        
        robot_sensor = Sensor()
        with open(self.params_file, 'r') as file:
            config = json.load(file)         
        robot_sensor.initialize_from_config(config = config)
        self.map = Mapping(robot_sensor)  # Local mapping module of the robot
        self.controller = Controller()

    """
    Home the robot to its initial position.

    Args:
        self: The instance of the class.

    Returns:
        None
    """
    def home_robot(self):
        is_homed = False
        collided = False
        
        home_pos = self.forward([0.0, 0.0 , 0.0], 3)
        self.add_behavior(1, home_pos)
        self.local_goal = home_pos
        self.global_goal =  home_pos

        #return to home position  
              
        while not (is_homed or collided):
            is_homed, collided = self.behavior_step()    

    """
    Update true landmarks in the robot's perception module.

    Args:
        lds (List[List[Union[float, int]]]): List of landmarks with x, y, and radius.
    """
    def update_true_landmarks(self, lds: List[List[Union[float, int]]]):
        self.map.sensor.landmarks = lds

    def percieve_landmarks(self):
        """Perceive landmarks using the robot's sensor."""
        self.map.sensor.perceive_environment()
        self.map.update_map()

    """
    Update the global goal of the robot.
    Check if the goal is reachable by the robotic arm

    Args:
        self: The instance of the class.
        goal: The new global goal for the robot.

    Returns:
        None
    """
    def update_goal(self, goal):
        self.global_goal = goal
        self.behavior_done = False
        
        #Check whether goal is reachable
        max_reachable_point = self.forward([0.0, 0.0, 0.0], 3)
        max_reachable_radius = np.linalg.norm(np.array(max_reachable_point))
        if np.linalg.norm(np.array(goal)) > max_reachable_radius:
            logger.debug("GOAL is out of reach of robotic arm !!")
            return False    
        else:
            logger.info("New Goal has been Set")

        return True
    
    def collission_detection(self):

        # Check collisions with land mark points. Landmarks considered as obstacles
        min_dist = 1000000
        for ld in self.map.landmark_scans:
            point = np.array([ld[0], ld[1]])

            # Check if the point is within threshold distance from all three links
            for joint_iter in range(len(self.joint_angles)):
    
                line_start = np.array([0, 0]) if joint_iter == 0 else np.array(self.forward(self.joint_angles, joint_iter))
                line_end = np.array(self.forward(self.joint_angles, joint_iter+1))
                distance = distance_to_line_segment(point, line_start, line_end)
                
                if distance < self.threshold:
                    logger.info(" Link has collided :")
                    logger.debug("Distance to obstacle : ",distance)
                    self.not_collided = False
                    return
            # also check for distance from joints
                
                distance_to_joint = calculate_distance(point, line_end)

                if distance_to_joint < self.threshold:
                    logger.info(" Joint has collided :")
                    logger.debug("Distance to obstacle : ",distance_to_joint)
                    self.not_collided = False
                    return

        return 
    
    """
    Plan the path to the goal.

    Args:
        self: The instance of the class.

    Returns:
        None
    """
    def plan_path_to_goal(self):

        start_point = (int(self.eof_pos[0]), int(self.eof_pos[1]))
        logger.debug("Start Point of the Path: ",start_point)
        self.planned_path = astar_search(self.map.grid2D, start_point, tuple(self.global_goal))

        path_np = np.array(self.planned_path)
        x_points, y_points = path_np[:, 0], path_np[:, 1]
        
        # Interpolate the path
        num_points_to_include = 5

# Downsample the waypoints using uniform sampling
        
        downsampled_indices = np.round(np.linspace(0, len(path_np) - 1, num_points_to_include)).astype(int)
        downsampled_x = x_points[downsampled_indices]
        downsampled_y = y_points[downsampled_indices]

        # Create a parameterization based on the cumulative distance along the path
        t = np.cumsum(np.sqrt(np.diff(downsampled_x)**2 + np.diff(downsampled_y)**2))
        t = np.insert(t, 0, 0)  # Insert a starting point at t=0

        # Create cubic splines for x and y coordinates
        spline_x = CubicSpline(t, downsampled_x)
        spline_y = CubicSpline(t, downsampled_y)

        smooth_trajectory_num_of_points = 10
        # Generate a finer parameterization for smoother trajectory
        t_fine = np.linspace(0, t[-1], smooth_trajectory_num_of_points)

        # Interpolate x and y coordinates using cubic splines
        x_interpolated = spline_x(t_fine)
        y_interpolated = spline_y(t_fine)
        
        self.interpolated_path = np.column_stack((x_interpolated, y_interpolated))

        self.local_goal =[ self.interpolated_path[0,0], self.interpolated_path[0,1] ]

    """
    Compute the x, y position of the end of the links from the joint angles. Forward Kinemetics

    Args:
        angles (List[float]): List of joint angles.
        joint_num (int): Number of joints to consider.

    Returns:
        Tuple[float, float]: x, y position of the end of the links.
    """
    @classmethod
    def forward(cls, angles: List[float], joint_num: int) -> Tuple[float, float]:

        x = 0
        y = 0
        theta = 0
        for iter in range(joint_num):
            theta += angles[iter]
            x += cls.link_lengths[iter] * np.cos(theta)
            y += cls.link_lengths[iter] * np.sin(theta)
        return x, y

    """
    Compute the joint angles from the position of the end of the links.

    Args:
        x (float): x position of eof.
        y (float): y position of eof.
        psi (float): orientation angle value of eof (radians).

    Returns:
        Tuple[float, float]: Joint angles.
    """
    @classmethod
    def analytical_inverse(cls, x: float, y: float, psi: float) -> Tuple[float, float, float]:
        if cls.analytical_ik:
            x2 = x - cls.link_3 * math.cos(psi)
            y2 = y - cls.link_3 * math.sin(psi)

            theta_1 = math.acos((x2 ** 2 + y2 ** 2 - cls.link_1 ** 2 - cls.link_2 ** 2) /
                                (2 * cls.link_1 * cls.link_2))
            theta_0 = np.arctan2(y2, x2) - \
                np.arctan((cls.link_2 * np.sin(theta_1)) / (cls.link_1 + cls.link_2 * np.cos(theta_1)))
            theta_2 = psi - theta_0 - theta_1
            return theta_0, theta_1, theta_2
        else:
            pass
            # Implement numerical IK if needed

    """
    Add behavior to the robot. Right now behaviors are MOVE_2_GOAL and
    PLAN_PATH_MOVE_2_GOAL 
    This call will initialize the behavior tree for the robot

    Args:
        self: The instance of the class.
        behavior: The behavior to be added.
        arg: The argument associated with the behavior.

    Returns:
        None
    """
    def add_behavior(self, behavior):
        
        self.behavior_option = behavior

        if behavior == MOVE_2_GOAL:

            self.local_goal = self.global_goal
            self._move_to_goal_behavior_add(plan_path = False)
            #initializing eof orientation
            eof_orientation = math.atan2(self.local_goal[1], self.local_goal[0])
            self.target_state = np.array(self.analytical_inverse(self.local_goal[0], self.local_goal[1],
                        eof_orientation))

        if behavior == PLAN_PATH_MOVE_2_GOAL:

            # self.local_goal =[ self.interpolated_path[0,0], self.interpolated_path[0,1] ]
            self._move_to_goal_behavior_add(plan_path = True)


    def reset_behavior(self):
        #Resetting path and interpolated path of robot after goal is achieved TODO through a reset function
        if self.behavior_option == PLAN_PATH_MOVE_2_GOAL:
            self.planned_path =None
            self.interpolated_path = []
            self.local_goal_iter = 0
            
        self.add_behavior(self.behavior_option)    
                
    def _move_to_goal_behavior_add(self, plan_path: bool):
            
        # Build the tree
        move_to_goal_action = MoveToGoalAction(name="MoveToGoal",robot=self, path_plan=plan_path)
        collision_detection_action = CollisionDetectionAction(name="CollissionDetection",robot=self)
        root_node = py_trees.composites.Sequence(name="Root", memory=True)
        root_node.add_children([collision_detection_action, move_to_goal_action])

        # Create the behavior tree
        self.behavior_tree = py_trees.trees.BehaviourTree(root_node)
       
    def perception_module_step(self):
        self.map.sensor.sensor_position = self.eof_pos
        self.map.sensor.sensor_orientation = sum(self.joint_angles)
        logger.debug("Sensor position:",self.map.sensor.sensor_position)
        self.percieve_landmarks()
        self.collission_detection()

    def behavior_step(self):
        
        #tick the sensor and perception and update map

        self. perception_module_step()
        if not self.behavior_done:    
            self.behavior_tree.tick()

        if self.behavior_tree.root.status == Status.FAILURE:

            logger.info("The behavior execution has failed")
    
        if self.behavior_tree.root.status == Status.SUCCESS:
            logger.info("Reached Goal!")
            self.behavior_done = True
            self.reset_behavior()

        return self.behavior_done, not self.not_collided

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


"""
Controller class owned by the robot.
Functions to calculate differential inverse kinematics and also control the joint position using PD controller 
"""
class Controller:
    def __init__(self) -> None:

        self.Kp = 1.5
        self.Kd = 0.1
        self.dt = 0.1
        self.prev_e = 0.0

    """
    Perform a step in the differential kinematics PID control loop.

    Args:
        joint_state: The current state (joint angles) of the system.
        link_l: The link lengths of the system.
        des_pos: The desired end effector position.
        des_state_dot: The desired end effector velocity.

    Returns:
        list: The new state of the system. (joint angles)
    """
    def step_differential_kin_pid(self, joint_state, link_l, des_pos, des_state_dot):
        dist_update = 0.5

        Kp=np.eye(3)*dist_update

        q = np.array(joint_state, dtype=float)
        e = np.zeros(q.shape)

        cur_pos = forward_temp(q, link_l)
        e = np.array(des_pos) - np.array(cur_pos)
        e_norm = e/np.linalg.norm(e)

        J = Jacobian("3_link_planar", q, link_l)
        dq_dt=np.dot(np.linalg.pinv(J),(np.dot(Kp,e_norm)))
        
        q=q+np.transpose(dq_dt)
         

        return q.tolist()
               
    def step(self, state, desired_state):
        """
        Perform a step in the control loop.

        Args:
            robot (Robot): The robot to control.

        Returns:
            Robot: Updated robot.
        """
        # Simple PD controller
        theta_error = desired_state - np.array(state)
        
        e_dot = (theta_error - self.prev_e)/self.dt
        self.prev_e = theta_error
        
        cur_state = state + theta_error*self.Kp + e_dot*self.Kd
        return cur_state.tolist()


