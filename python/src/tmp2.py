import time
from typing import List, Tuple, Union
import scipy
from scipy.interpolate import CubicSpline
import numpy as np
import math
import py_trees
import logging
from py_trees.common import Status
from py_trees.decorators import SuccessIsRunning
from py_trees.composites import Sequence
from py_trees.blackboard import Blackboard
from rrt import RRT
from astar import astar_search

from helper_functions import calculate_distance, distance_to_line_segment, Jacobian, forward_temp
from robot_actions import MoveToGoalAction, CollisionDetectionAction

MOVE_2_GOAL = 1
PLAN_PATH = 2


# Set the logging level for PyTrees
py_trees.logging.level = logging.INFO  # Set it to the desired level, e.g., logging.INFO, logging.DEBUG, etc.

"""
Simulation of a sensor with 180-degree field of view and defined sensor radius.
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

                dist = calculate_distance(point, self.sensor_position)

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
        self.map_width = 600
        self.map_height = 600
        # x becomes rows, and y is columns. Convert such that the robot is in the center
        self.grid2D = np.zeros((self.map_height, self.map_width))
        # self.sensor_scans = set() # set of sensor scans (x,y) in robot frame
        self.landmark_scans = set()  # set of landmark scans (x,y) in robot frame
        self.new_landmarks_grid = [] #list of [x,y] which are based on grid coordinates
        self.new_scan_grid = []
        self.sensor = sensor
        self.global_goal = []
        
    def update_map(self):
        # print("frontiers of landmarks : x")
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
    
    def __init__(self) -> None:
        # Internal variables

        self.all_angles: List[List[float]] = [[], [], []]
        self.all_joint_velocities: List[List[float]] = [[], [], []]

        self._joint_angles: List[float] = [0., 0., 0.]
        self.joint_angles = [0., 0., 0.]
        self._joint_velocity = List[float]

        self.eof_pos = self.forward(self.joint_angles, 3)
        self.link_lengths = [self.link_1, self.link_2, self.link_3]
        self.map = None

        self.local_goal = self.eof_pos
        self.local_goal_iter = 0
        self.reached_goal = False
        self.planned_path = None
        self.interpolated_path = None
        self.not_collided = True
        self.threshold = 1.5  # Collission threshold

        # Run Robot Init Procedure
        self.init_all_modules()
        self.blackboard = None
    @property
    def joint_angles(self) -> List[float]:
        """Getter for joint angles."""
        return self._joint_angles

    @property
    def joint_velocity(self) -> List[float]:
        """Getter for joint angles."""
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
            self._joint_velocity = (angle-self._joint_angles[iter])/self.DT


        #finding velocity
        # if len(self.join)
        self._joint_velocity = ((np.array(values) - np.array(self._joint_angles))/self.DT).tolist()
        for iter in range(len(values)):
            self.all_joint_velocities[iter].append(self._joint_velocity)
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
        # print("Sensor Orientation", np.rad2deg(self.map.sensor.sensor_orientation))
        # print("Sensor Position", self.map.sensor.sensor_position)


    def collission_detection(self):

        # Check collisions with land mark points. Landmarks considered as obstacles
        min_dist = 1000000
        for ld in self.map.landmark_scans:
            point = np.array([ld[0], ld[1]])
            # print("Landmark : ", ld)
            # Check if the point is within threshold distance from all three links
            for joint_iter in range(len(self.joint_angles)):
    
                line_start = np.array([0, 0]) if joint_iter == 0 else np.array(self.forward(self.joint_angles, joint_iter))
                line_end = np.array(self.forward(self.joint_angles, joint_iter+1))
                distance = distance_to_line_segment(point, line_start, line_end)
                # print(distance)
                
                if distance < self.threshold:
                    print(" Link has collided :")
                    print("Distance to obstacle : ",distance)
                    self.not_collided = False
                    return
            # also check for distance from joints
                
                distance_to_joint = calculate_distance(point, line_end)
                # print(distance_to_joint)    
                if distance_to_joint < self.threshold:
                    print(" Joint has collided :")
                    print("Distance to obstacle : ",distance_to_joint)
                    self.not_collided = False
                    return

        return 

    def plan_path_to_goal(self):
        # rrt = RRT(
        # start=[0, 0],
        # goal = self.local_goal,
        # rand_area=[-200, 200],
        # obstacle_list = [],
        # # play_area=[0, 10, 0, 14]
        # robot_radius=0.8
        # )
        # path = rrt.planning(animation=show_animation)
        # sensor_grid_x = int(self.map.map_height / 2 - self.map.sensor.sensor_position[1])
        # sensor_grid_y = int(self.map.sensor.sensor_position[0] + self.map.map_width/2)
        start_point = (int(self.eof_pos[0]), int(self.eof_pos[1]))
        print(start_point)
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
        print(self.interpolated_path)
        print(self.interpolated_path.shape)
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


# def inv_kinematics_leg(leg_num, des_pos, des_vel, t, dt, q):
#     q_final=[]
#     Kp=np.eye(3)*0.05
#     Kd=np.eye(3)*0.01
#     # q=[0, 0, 0]
#     q=np.array(q,dtype='float')
#     e=np.zeros((3,1))
#     e_dot=np.zeros((3,1))
#     print("Desired Trajectory : ", des_pos)

#     des_pos=np.round(des_pos,3) 
#     des_vel=np.round(des_vel,3)
#     cnt=0
#     # correct_leg_num=leg_num
#     # leg_num=int(leg_num/2)
#     pe=get_transformation_matrix(leg_num,q)
#     pe=pe[0:3,3]

#     for iter in range(len(t)):
#         pd=des_pos[iter,0:3]
#         xd_dot=des_vel[iter,0:3]
#         # #print(dt)
        
        
#         for iter2 in range(1000):
#             # pe=np.round(self.get_transform_foot_base(leg_num, [q[0],q[1],q[2]], False),3)
#             # #print("Angle",q)
#             # #print(leg_num)
#             pe=get_transformation_matrix(leg_num,q)
#             # #print(pe)
#             pe=pe[0:3,3]
            
#             # #print("end effector pos",pe)
#             # print(pd.shape)
#             # print(pe.shape)
#             e[:,0]=pd-pe
            
            
#             # J=np.round(self.get_jacobian(leg_num, q),2)
#             J=get_jacobian_mat(leg_num, q)
#             if cnt==0:
#                 # #print("Jacobian : \n",J)
#                 cnt=cnt+1
#             # if iter==0:
#             #     pe_dot=0
#             # else:
#             #     pe_dot=np.round((pe-des_pos[iter-1,0:3])/dt,3)
#             # e_dot[:,0]=xd_dot-pe_dot

#             # #print("ss\n",e.shape)
#             # #print((Kp*e).shape)
#             dq_dt=np.dot(np.round(np.linalg.pinv(J),3),(np.dot(Kp,e)))#+np.dot(Kd,e_dot)))
#             # #print(dq_dt,"Vel")
            
#             # q=q+np.transpose(dq_dt*dt)
#             q=q+np.transpose(dq_dt)

#             q=q[0]
#             if np.linalg.norm(e)<0.005:
#                 # #print('Target position',pd)
#                 # #print("Foot position now :",pe)
#                 # #print("error",e)
#                 np.linalg.norm(e)
#                 break                

#             # break
#         #print("iteration:",iter," des_pos ",pd)
#         #print(iter2)
#         q_final.append(q)
#         #print( "Solved Inverse Kinematics for ",correct_leg_num)

#     return q_final, dq_dt
    # def jacobian_based_inverse(self, desired_state):
        
    def add_behavior(self, behavior, arg):
        
        self.behavior = behavior
        if self.behavior == MOVE_2_GOAL:
            self.local_goal = arg
            self.global_goal = arg
            # Build the tree
            move_to_goal_action = MoveToGoalAction(name="MoveToGoal",robot=self)
            collision_detection_action = CollisionDetectionAction(name="CollissionDetection",robot=self)
            root_node = py_trees.composites.Sequence(name="Root", memory=True)
            root_node.add_children([collision_detection_action, move_to_goal_action])

            # Create the behavior tree
            self.behavior_tree = py_trees.trees.BehaviourTree(root_node)
       

            eof_orientation = math.atan2(self.local_goal[1], self.local_goal[0])
            self.target_state = np.array(self.inverse(self.local_goal[0], self.local_goal[1],
                        eof_orientation))

        if self.behavior == PLAN_PATH:
            self.global_goal = arg
            self.plan_path_to_goal()

            self.local_goal =[ self.interpolated_path[0,0], self.interpolated_path[0,1] ]
            # Build the tree
            move_to_goal_action = MoveToGoalAction(name="MoveToGoal",robot=self)
            collision_detection_action = CollisionDetectionAction(name="CollissionDetection",robot=self)
            root_node = py_trees.composites.Sequence(name="Root", memory=True)
            root_node.add_children([collision_detection_action, move_to_goal_action])

            # Create the behavior tree
            self.behavior_tree = py_trees.trees.BehaviourTree(root_node)
       

            eof_orientation = math.atan2(self.local_goal[1], self.local_goal[0])
            self.target_state = np.array(self.inverse(self.local_goal[0], self.local_goal[1],
                        eof_orientation))
            # eof_orientation = math.atan2(self.local_goal[1], self.local_goal[0])
            # print("eof Orientation ", eof_orientation)
            # self.target_state = np.array(self.inverse(self.local_goal[0], self.local_goal[1],
            #             eof_orientation))
            # self.state =  np.array(self.joint_angles)

    def behavior_terminate(self) -> bool:
        
        if self.behavior == MOVE_2_GOAL:
            return np.allclose(
                self.forward(self.joint_angles, 3), self.local_goal, atol=0.5
            ) 
    
    
    def behavior_step(self):
        # if not self.behavior_terminate() and self.not_collided:
            
        #     self.eof_pos = self.forward(self.joint_angles, 3)
        #     self.map.sensor.sensor_position = self.eof_pos
        #     self.percieve_landmarks()
            
        #     self.collission_detection()
        #     if self.not_collided:
        #         self.joint_angles = self.controller.step(self.joint_angles, self.target_state)
        #     else:
        #         #Setting joint velocity to zero
        #         self.joint_angles =self.joint_angles
                
        #     # self.state = np.array(self.joint_angles)
            
        #     time.sleep(1)            

        # else:
        # print("Behavior has been completed")
        self.eof_pos = self.forward(self.joint_angles, 3)
        self.map.sensor.sensor_position = self.eof_pos
        print(self.map.sensor.sensor_position)
        self.percieve_landmarks()
        self.collission_detection()

        self.behavior_tree.tick()
        # move_to_goal_status = status_dictionary[self.move_to_goal_action]
        behavior_return = False
        print("Behavior status ",self.behavior_tree.root.status)
        if self.behavior_tree.root.status == Status.FAILURE:
            print("The behavior execution has failed")
        if self.behavior_tree.root.status == Status.SUCCESS:
            print("Reached Goal!")
            behavior_return = True
            # print(move_to_goal_status)
        # collision_detection_status = self.blackboard.get("CollisionDetectionAction")

        # print(f"CollisionDetectionAction Status: {collision_detection_status}")
        # # Check the status of the MoveToGoalAction using the blackboard
        # move_to_goal_status = self.blackboard.get("MoveToGoalAction")
        # print(f"MoveToGoalAction Status: {move_to_goal_status}")
        # if move_to_goal_status == Status.SUCCESS:
        #     print("Goal reached!")
        #     behavior_return = True
        # elif move_to_goal_status == Status.FAILURE:
            
        #     print("Behavior tree failed to reach the goal.")
        # else:
            # print("Behavior tree execution in progress.")
        return behavior_return, not self.not_collided

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
        self.Kp = 0.01
        self.Kd = 0.001
        self.dt = 0.1
        self.prev_e = 0.0

    def step_differential_kin_pid(self, state, link_l, des_x, des_state_dot):
        Kp=np.eye(2)*0.005
        Kd=np.eye(2)*0.001
        MAX_ITER = 1000
        for _ in range(MAX_ITER):        
            q = np.array(state, dtype=float)
            e = np.zeros(q.shape)
            e_dot = np.zeros(q.shape)
            
            cur_x = forward_temp(q, link_l)
            e = np.array(des_x) - np.array(cur_x)
            e_dot =np.array([1.0,1.0])

            J = Jacobian("3_link_planar", q, link_l)
            dq_dt=np.dot(np.round(np.linalg.pinv(J),3),(np.dot(Kp,e)+np.dot(Kd,e_dot)))
            # #print(dq_dt,"Vel")
            
            # q=q+np.transpose(dq_dt*dt)
            q=q+np.transpose(dq_dt*self.dt)
            if np.linalg.norm(e)<0.005:

                print("error",e)
                break            
        print("New q ", q)

        return q.tolist()
               
    def step(self, state, desired_state):
        """
        Perform a step in the control loop.

        Args:
            robot (Robot): The robot to control.

        Returns:
            Robot: Updated robot.
        """
        # Simple P controller
        theta_error = desired_state - np.array(state)
        
        e_dot = (theta_error - self.prev_e)/self.dt
        self.prev_e = theta_error
        
        cur_state = state + theta_error*self.Kp #+ e_dot*self.Kd
        return cur_state.tolist()





import py_trees
from py_trees.common import Status
import numpy as np
from helper_functions import calculate_distance

class CollisionDetectionAction(py_trees.behaviour.Behaviour):
    def __init__(self, name="CollisionDetectionAction", robot=None):
        super(CollisionDetectionAction, self).__init__(name)
        self.robot = robot
        self.status = Status.RUNNING

    def update(self):
        # Check for collisions using robot's collision_detection method
        self.robot.collission_detection()

        # Update the status based on the collision detection result
        if self.robot.not_collided:
            self.status = Status.SUCCESS
        else:
            self.status = Status.FAILURE

        return self.status
class MoveToGoalAction(py_trees.behaviour.Behaviour):
    def __init__(self, name="MoveToGoalAction",  robot = None):
        super(MoveToGoalAction, self).__init__(name)
        self.robot = robot
        self.status = Status.RUNNING
        py_trees.logging.level = py_trees.logging.Level.DEBUG

    def update(self):

        self.feedback_message = f"Executing MoveToGoalAction, Robot position: {self.robot.eof_pos}"
        self.logger.debug(self.feedback_message)

        #Check if we have reached local goal. If reached update the local_goal
        # Execute controller step if not collided
        dist = calculate_distance(self.robot.forward(self.robot.joint_angles, 3), self.robot.local_goal)
        if dist<2.0:
            if self.robot.interpolated_path is None or self.robot.local_goal_iter == self.robot.interpolated_path.shape[0]:
                self.feedback_message = "Goal reached!"
                self.logger.info(self.feedback_message)
                print("Inside action",Status.SUCCESS)
                self.status= Status.SUCCESS
            else:
                print("Moving to next local goal !")
                self.robot.local_goal_iter += 1            
                self.robot.local_goal = self.robot.interpolated_path[self.robot.local_goal_iter,:]
                    
        if self.robot.not_collided:
            des_state= [self.robot.local_goal[0], self.robot.local_goal[1]]
            # self.robot.joint_angles = self.robot.controller.step_differential_kin_pid(self.robot.joint_angles, self.robot.link_lengths,
                                                                        #    des_state, [1, 1, 1])
            self.robot.joint_angles = self.robot.controller.step(self.robot.joint_angles, self.robot.target_state)
        else:
            #Setting joint velocity to zero
            self.robot.joint_angles =self.robot.joint_angles
            self.status = Status.FAILURE
        # Return SUCCESS when the robot reaches the goal (you may need to adjust this condition)
        # x, y = self.robot.forward(self.robot.joint_angles, 3)
        print(dist)
        # if  dist<1.5:

        print("running")
        print(self.status==Status.RUNNING)
        return self.status
        




