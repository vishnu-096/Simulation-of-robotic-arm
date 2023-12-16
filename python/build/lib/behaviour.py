import py_trees
from py_trees.common import Status
from py_trees.decorators import SuccessIsRunning
from py_trees.composites import Sequence
from py_trees.blackboard import Blackboard
from typing import Tuple
from Robot import Robot, Controller

class MoveToGoalAction(py_trees.behaviour.Behaviour):
    def __init__(self, name="MoveToGoalAction", goal: Tuple[int, int] = (0, 0)):
        super(MoveToGoalAction, self).__init__(name)
        self.controller = Controller(goal)
        self.robot = Robot()

    def update(self):
        self.feedback_message = f"Executing MoveToGoalAction, Robot position: {self.robot.eof_pos}"
        self.logger.debug(self.feedback_message)

        self.robot = self.controller.step(self.robot)

        # Return SUCCESS when the robot reaches the goal (you may need to adjust this condition)
        if self.robot.eof_pos == self.controller.goal:
            self.feedback_message = "Goal reached!"
            self.logger.info(self.feedback_message)
            return Status.SUCCESS
        else:
            return Status.RUNNING

robot= Robot()
# Create nodes
move_to_goal_action = MoveToGoalAction(name="MoveToGoalAction", goal=(100, 100), )

# Build the tree

move_to_goal_sequence = Sequence(name="MoveToGoalSequence", memory = True)
move_to_goal_sequence.add_children([move_to_goal_action])

# Instantiate the behavior tree
behavior_tree = py_trees.trees.BehaviourTree(move_to_goal_sequence)

# Main execution loop
for _ in range(20):
    # Execute the behavior tree
    status = behavior_tree.tick()

    # Log the feedback message
    feedback_message = move_to_goal_action.feedback_message
    print(f"Feedback: {feedback_message}")

    # Check the result
    if status == Status.SUCCESS:
        print("Goal reached!")
        break
    elif status == Status.FAILURE:
        print("Behavior tree failed to reach the goal.")
        break
    else:
        print("Behavior tree execution in progress.")
