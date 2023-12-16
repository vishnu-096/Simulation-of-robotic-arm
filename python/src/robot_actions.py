import py_trees
from py_trees.common import Status
from helper_functions import calculate_distance
import logging_config
import logging

# Get a logger instance
logger = logging.getLogger(__name__)
py_trees.logging.level = logging_config.logging_level

GOAL_THRESHOLD = 2.0

"""
    A action class for collision detection.

    Parameters:
    - name (str): Name of the behavior.
    - robot: The robot instance.
"""
class CollisionDetectionAction(py_trees.behaviour.Behaviour):
    def __init__(self, name="CollisionDetectionAction", robot=None):
        super(CollisionDetectionAction, self).__init__(name)
        self.robot = robot
        self.status = Status.RUNNING

    def update(self):
        # Check for collisions using robot's collision_detection method
        self.robot.collission_detection()

        # Update the status based on the collision detection result
        self.status = Status.SUCCESS if self.robot.not_collided else Status.FAILURE
        return self.status

"""
A Action class for moving to a goal.

Parameters:
- name (str): Name of the action.
- robot: The robot instance.
"""
class MoveToGoalAction(py_trees.behaviour.Behaviour):
    def __init__(self, name="MoveToGoalAction",  robot = None, path_plan = False):
        super(MoveToGoalAction, self).__init__(name)
        self.robot = robot
        self.status = Status.RUNNING
        self.path_planning = path_plan

    """
    Update the MoveToGoalAction behavior.
    
    Plans path if not planned.
    Find best joint angles based on inverse jacobian differential kinematics
    Step towards the best joint positions using PD controller while following the planned path

    Returns:
    - Status: SUCCESS if goal reached, FAILURE if collision and RUNNING if neither.
    """
    def update(self):

        if self.path_planning and self.robot.planned_path is None:
            self.robot.plan_path_to_goal()
        self.feedback_message = f"Executing MoveToGoalAction, Robot position: {self.robot.eof_pos}"
        self.logger.debug(self.feedback_message)

        #Check if we have reached local goal. If reached update the local_goal
        # Execute controller step if not collided
        dist = calculate_distance(self.robot.forward(self.robot.joint_angles, 3), self.robot.local_goal)
        self.robot.step_iterations += 1
        
        if dist< GOAL_THRESHOLD:
            self.robot.step_iterations = 0
            if (self.robot.local_goal_iter == self.robot.interpolated_path.shape[0]):
                self.feedback_message = "Goal reached!"
                self.logger.info(self.feedback_message)
                logger.debug("Inside action",Status.SUCCESS)

                self.status= Status.SUCCESS
                

            else:
                logger.debug("Moving to next local goal !")
                self.robot.local_goal = self.robot.interpolated_path[self.robot.local_goal_iter,:]
                self.robot.local_goal_iter += 1            
        
        # If robot takes too many steps to reach the goal, accept failure
        if  self.robot.step_iterations>1000:
            logger.info("too slow in reaching local goal!")
            self.status = Status.FAILURE
            
        if self.robot.not_collided:
            des_state= [self.robot.local_goal[0], self.robot.local_goal[1], 0]
            target_state = self.robot.controller.step_differential_kin_pid(self.robot.joint_angles, self.robot.link_lengths,
                                                                           des_state, [1, 1, 1])
            self.robot.target_state = target_state
            self.robot.joint_angles = self.robot.controller.step(self.robot.joint_angles, self.robot.target_state)
        else:
            #Setting joint velocity to zero
            self.robot.joint_angles =self.robot.joint_angles
            self.status = Status.FAILURE

        return self.status


        
