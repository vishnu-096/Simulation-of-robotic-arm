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


clock = pygame.time.Clock()

def forward_kinematics(theta):
    l1, l2, l3 = 75, 50, 25  # Lengths of the three links
    theta1, theta2, theta3 = theta

    x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2) + l3 * np.cos(theta1 + theta2 + theta3)
    y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2) + l3 * np.sin(theta1 + theta2 + theta3)

    return np.array([x, y])

# Define the inverse kinematics objective function
def inverse_kinematics_objective(theta, target):
    return np.linalg.norm(forward_kinematics(theta) - target)

class Robot:
    JOINT_LIMITS = [-6.28, 6.28]
    MAX_VELOCITY = 15
    MAX_ACCELERATION = 50
    DT = 0.033

    link_1: float = 75.  # pixels
    link_2: float = 50.  # pixels
    link_3: float = 25.  # pixels    
    _theta_0: float      # radians
    _theta_1: float      # radians
    _theta_2: float      # radians
    link_lengths=[link_1, link_2, link_3]

    def __init__(self) -> None:
        # internal variables
        self.all_theta_0: List[float] = []
        self.all_theta_1: List[float] = []
        self.all_theta_2: List[float] = []

        self.joint_angles=[0., 0., 0.]

        self.theta_0 = 0.
        self.theta_1 = 0.
        self.theta_2 = 0.
        self.joint_angles=[0., 0., 0.]
        self.link_lengths=[self.link_1, self.link_2, self.link_3]

    # Getters/Setters
    @property
    def theta_0(self) -> float:
        return self._theta_0

    @theta_0.setter
    def theta_0(self, value: float) -> None:
        self.all_theta_0.append(value)
        self._theta_0 = value
        self.joint_angles[0] =value
        # Check limits
        assert self.check_angle_limits(value), \
            f'Joint 0 value {value} exceeds joint limits'
        assert self.max_velocity(self.all_theta_0) < self.MAX_VELOCITY, \
            f'Joint 0 Velocity {self.max_velocity(self.all_theta_0)} exceeds velocity limit'
        assert self.max_acceleration(self.all_theta_0) < self.MAX_ACCELERATION, \
            f'Joint 0 Accel {self.max_acceleration(self.all_theta_0)} exceeds acceleration limit'

    @property
    def theta_1(self) -> float:
        return self._theta_1

    @theta_1.setter
    def theta_1(self, value: float) -> None:
        self.all_theta_1.append(value)
        self._theta_1 = value
        self.joint_angles[1]=value

        assert self.check_angle_limits(value), \
            f'Joint 1 value {value} exceeds joint limits'
        assert self.max_velocity(self.all_theta_1) < self.MAX_VELOCITY, \
            f'Joint 1 Velocity {self.max_velocity(self.all_theta_1)} exceeds velocity limit'
        assert self.max_acceleration(self.all_theta_1) < self.MAX_ACCELERATION, \
            f'Joint 1 Accel {self.max_acceleration(self.all_theta_1)} exceeds acceleration limit'

    @property
    def theta_2(self) -> float:
        return self._theta_2

    @theta_2.setter
    def theta_2(self, value: float) -> None:
        self.all_theta_2.append(value)
        self._theta_2 = value
        self.joint_angles[2]=value

        assert self.check_angle_limits(value), \
            f'Joint 2 value {value} exceeds joint limits'
        assert self.max_velocity(self.all_theta_2) < self.MAX_VELOCITY, \
            f'Joint 2 Velocity {self.max_velocity(self.all_theta_2)} exceeds velocity limit'
        assert self.max_acceleration(self.all_theta_1) < self.MAX_ACCELERATION, \
            f'Joint 2 Accel {self.max_acceleration(self.all_theta_2)} exceeds acceleration limit'

    # Kinematics
    def joint_1_pos(self) -> Tuple[float, float]:
        """
        Compute the x, y position of joint 1
        """
        return self.forward(self.theta_0, self.theta_1, self.theta_2, 1)

    def joint_2_pos(self) -> Tuple[float, float]:
        """
        Compute the x, y position of joint 2
        """
        return self.forward(self.theta_0, self.theta_1, self.theta_2, 2)

    def joint_3_pos(self) -> Tuple[float, float]:
        """
        Compute the x, y position of joint 3
        """
        return self.forward(self.theta_0, self.theta_1, self.theta_2, 3)
    
    @classmethod
    def forward(cls, theta_0: float, theta_1: float, theta_2: float, joint_num: int) -> Tuple[float, float]:
        """
        Compute the x, y position of the end of the links from the joint angles
        """
        x=0
        y=0
        theta=[theta_0, theta_0+theta_1, theta_0+theta_1+theta_2]
        for iter in range(joint_num):
            x += cls.link_lengths[iter]*np.cos(theta[iter])
            y += cls.link_lengths[iter]*np.sin(theta[iter])

        return x, y

    @classmethod
    def inverse(cls, x: float, y: float, psi: float) -> Tuple[float, float]:
        """
        Compute the joint angles from the position of the end of the links
        """

        x2 = x - cls.link_3*math.cos(psi)
        y2 = y - cls.link_3*math.sin(psi)

        theta_1 = np.arccos((x2 ** 2 + y2 ** 2 - cls.link_1 ** 2 - cls.link_2 ** 2)
                            / (2 * cls.link_1 * cls.link_2))
        theta_0 = np.arctan2(y2, x2) - \
            np.arctan((cls.link_2 * np.sin(theta_1)) /
                      (cls.link_1 + cls.link_2 * np.cos(theta_1)))
        theta_2 = psi - theta_0 - theta_1

#         initial_guess = [0, 0, 0]

#     # Desired end-effector position
#         target_position = np.array([x, y])

# # Use a numerical optimizer to find joint angles
#         result = minimize(inverse_kinematics_objective, initial_guess, args=(target_position,))
#         joint_angles = result.x

        return theta_0, theta_1, theta_2

    @classmethod
    def check_angle_limits(cls, theta: float) -> bool:
        return cls.JOINT_LIMITS[0] < theta < cls.JOINT_LIMITS[1]

    @classmethod
    def max_velocity(cls, all_theta: List[float]) -> float:
        return float(max(abs(np.diff(all_theta) / cls.DT), default=0.))

    @classmethod
    def max_acceleration(cls, all_theta: List[float]) -> float:
        return float(max(abs(np.diff(np.diff(all_theta)) / cls.DT / cls.DT), default=0.))

    @classmethod
    def min_reachable_radius(cls) -> float:
        return max(cls.link_1 - cls.link_2, 0)

    @classmethod
    def max_reachable_radius(cls) -> float:
        return cls.link_1 + cls.link_2


class Obstacle:
    def __init__(self, obstacle_type: str, dimensions: Union[Tuple[float], Tuple[float, float]]) -> None:
        self.obstacle_type = obstacle_type
        self.dimensions = dimensions

class World:
    def __init__(
        self,
        width: int,
        height: int,
        robot: Robot,
        robot_origin: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> None:
        self.width = width
        self.height = height
        self.robot_origin = robot_origin
        self.goal = goal
        self.obstacles: List[Obstacle] = []
        self.robot = Robot
    def convert_to_display(
            self, point: Tuple[Union[int, float], Union[int, float]]) -> Tuple[int, int]:
        """
        Convert a point from the robot coordinate system to the display coordinate system
        """
        robot_x, robot_y = point
        offset_x, offset_y = self.robot_origin

        return int(offset_x + robot_x), int(offset_y - robot_y)

    def add_obstacle(
            self, obstacle_type: str, dimensions: Union[Tuple[float], Tuple[float, float]]) -> None:
        """
        Add obstacle to the world. Dimensions and type of obstacles need to provided
        type: "circle" or "rectangle"
        dimensions: as tuple
        
        """
        obstacle = Obstacle(obstacle_type, dimensions)
        self.obstacles.append(obstacle)

    def check_collisions(self, robot):
        
        joint1_pos = self.convert_to_display(robot.joint_1_pos())
        joint2_pos = self.convert_to_display(robot.joint_2_pos())

        #check collisions with circle obstacles
        for obstacle in self.obstacles:
            if obstacle.obstacle_type == "circle":
                if is_line_circle_intersection(joint1_pos, joint2_pos, obstacle.dimensions):
                    return True

        return False
    
class ConfigSpaceVisualizer:
    def __init__(self, robot: Robot, world: World) -> None:
        pygame.init()
        self.robot = robot
        self.world = world
        self.screen = pygame.display.set_mode((world.width, world.height))
        pygame.display.set_caption('Configuration Space Visualizer')
        self.manager = pygame_gui.UIManager((world.width, world.height))

    def display_config_space(self) -> None:
        # Add code to display the configuration space here
        pass

    def update_display(self) -> bool:
        time_delta = clock.tick(60) / 1000.0
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        self.screen.fill(Visualizer.WHITE)
        self.display_config_space()
        pygame.display.flip()

        return True

    def cleanup(self) -> None:
        pygame.quit()
        
            
class Visualizer:
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    RED: Tuple[int, int, int] = (255, 0, 0)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    clock = pygame.time.Clock()
    time_delta = clock.tick(60) / 1000.0
    is_running = True
    main_screen_width = 1500
    main_screen_height = 1500 
    plot_rect = pygame.Rect(100, 0, 300, 200)
    def __init__(self, world: World) -> None:
        """
        Note: while the Robot and World have the origin in the center of the
        visualization, rendering places (0, 0) in the top left corner.
        """
        pygame.init()
        pygame.font.init()
        self.world = world
        self.width = world.width
        self.height = world.height
        self.screen = pygame.display.set_mode((1000,1000))
        pygame.display.set_caption('Gherkin Challenge')
        # self.font = pygame.font.SysFont('freesansbolf.tff', 30)
        self.font = pygame.font.SysFont('freesans', 30)
        self.time=0
        self.manager = pygame_gui.UIManager((world.width, world.height))
        self.hello_button = pygame_gui.elements.UIButton(relative_rect=pygame.Rect((50, 75), (100, 50)),
                                             text='Say Hello',
                                             manager=self.manager)
        self.config_space_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((200, 75), (150, 50)),
            text='Open Config Space',
            manager=self.manager
        )
        self.plot_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((350, 75), (150, 50)),
            text='Plot Graphs',
            manager=self.manager
        )  
              
        self.config_space_visualizer = ConfigSpaceVisualizer(world.robot, world)

        self.fig = plt.figure(figsize=(5,5))
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvas(self.fig)
        # self.ax.set_xlim(0, 100)
        # self.ax.set_ylim(0, 1)
        self.canvas.draw()
        self.renderer = self.canvas.get_renderer()

    def display_world(self) -> None:
        """
        Display the world
        """
        goal = self.world.convert_to_display(self.world.goal)
        pygame.draw.circle(self.screen, self.RED, goal, 6)
        for obstacle in self.world.obstacles:
            self.display_obstacle(obstacle)
            # print("obstacle : ", obstacle.dimensions)


    def display_obstacle(self, obstacle: Obstacle) -> None:
        """
        Display the obstacle
        """
        obstacle_position = obstacle.dimensions[:2]

        if obstacle.obstacle_type == "circle":
            radius = int(obstacle.dimensions[2])
            pygame.draw.circle(self.screen, self.BLACK, obstacle_position, radius)
        elif obstacle.obstacle_type == "rectangle":
            width, height = map(int, obstacle.dimensions[2:])
            pygame.draw.rect(self.screen, self.BLACK, (obstacle_position[0], obstacle_position[1], width, height))
        else:
            raise ValueError("Unsupported obstacle type")
        
    def display_robot(self, robot: Robot) -> None:
        """
        Display the robot
        """
        j0 = self.world.robot_origin
        j1 = self.world.convert_to_display(robot.joint_1_pos())
        j2 = self.world.convert_to_display(robot.joint_2_pos())
        j3 = self.world.convert_to_display(robot.joint_3_pos())

        # Draw joint 0
        pygame.draw.circle(self.screen, self.BLACK, j0, 4)
        # Draw link 1
        pygame.draw.line(self.screen, self.BLACK, j0, j1, 2)
        # Draw joint 1
        pygame.draw.circle(self.screen, self.BLACK, j1, 4)
        # Draw link 2
        pygame.draw.line(self.screen, self.BLACK, j1, j2, 2)
        # Draw joint 2
        pygame.draw.circle(self.screen, self.BLACK, j2, 4)
        # Draw link 3
        pygame.draw.line(self.screen, self.BLACK, j2, j3, 2)
        # Draw joint 3
        pygame.draw.circle(self.screen, self.BLACK, j3, 4)

    def update_display(self, robot: Robot, success: bool, collission_detected: bool) -> bool:
        self.time=self.time+self.time_delta
        for event in pygame.event.get():
            self.manager.process_events(event)

            # Keypress
            if event.type == pygame.KEYDOWN:
                # Escape key
                if event.key == pygame.K_ESCAPE:
                    return False
            
            # Window Close Button Clicked
            if event.type == pygame.QUIT:
                return False
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                print("Heyyy")
                if event.ui_element == self.hello_button:
                    print('Hello World!')
                if event.ui_element == self.config_space_button:
                    print("Pressed!")
                    self.open_config_space()

        self.screen.fill(self.WHITE)

        self.display_world()
        
        self.display_robot(robot)

        if collission_detected:
            text = self.font.render('Collission!', True, self.BLACK)
            self.screen.blit(text, (10, 1))
        if success:
            text = self.font.render('Success!', True, self.BLACK)
            self.screen.blit(text, (1, 1))
        window_len =15
        if len(robot.all_theta_0)<window_len:
            window_len = len(robot.all_theta_0)

        time_window=np.linspace(max(self.time-self.time_delta*window_len, 0), self.time, window_len)
        print(time_window)
        self.ax.plot(time_window, robot.all_theta_0[-window_len:], '-ob', label='Theta 0')
        self.ax.plot(time_window, robot.all_theta_1[-window_len:], '-og', label='Theta 1')
        self.ax.plot(time_window, robot.all_theta_2[-window_len:], '-or', label='Theta 2')
        # self.fig.savefig('full_figure.png')
        self.ax.legend()

        self.canvas.draw()

        # Render Matplotlib plot onto Pygame surface
        plot_surface = pygame.image.fromstring(self.canvas.tostring_rgb(), self.canvas.get_width_height(), "RGB")
        self.screen.blit(plot_surface, self.plot_rect)        
        self.ax.cla()

        self.manager.update(self.time_delta)

        self.manager.draw_ui(self.screen)
        pygame.display.flip()


        return True

    def open_config_space(self) -> None:
        config_space_running = True
        print("Opening Config screen!")
        while config_space_running:
            config_space_running = self.config_space_visualizer.update_display()

    def cleanup(self) -> None:
        pygame.quit()


class Controller:
    def __init__(self, goal: Tuple[int, int]) -> None:
        self.goal = goal
        eof_orrientation=math.atan2(goal[1],goal[0])
        print("eof Orientation ", eof_orrientation)
        self.goal_theta_0, self.goal_theta_1, self.goal_theta_2 = Robot.inverse(self.goal[0], self.goal[1], eof_orrientation)
        print("Solution :", self.goal_theta_0, self.goal_theta_1, self.goal_theta_2)
        print("Goal :", self.goal)

    def step(self, robot: Robot) -> Robot:
        """
        Simple P controller
        """
        theta_0_error = self.goal_theta_0 - robot.theta_0
        theta_1_error = self.goal_theta_1 - robot.theta_1
        theta_2_error = self.goal_theta_2 - robot.theta_2

        robot.theta_0 += theta_0_error / 10
        robot.theta_1 += theta_1_error / 10
        robot.theta_2 += theta_2_error / 10

        return robot


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
        while running:
            
            # Step the controller if there is no collission
            if not collision_detected:
                self.robot = self.controller.step(self.robot)

            # Check collisions
            collision_detected = self.world.check_collisions(self.robot)

            # Check success
            success = self.check_success(self.robot, self.world.goal)

            # Update the display
            running = self.vis.update_display(self.robot, success, collision_detected)

            # sleep for Robot DT seconds, to force update rate
            time.sleep(self.robot.DT)

    @staticmethod
    def check_success(robot: Robot, goal: Tuple[int, int]) -> bool:
        """
        Check that robot's joint 2 is very close to the goal.
        Don't not use exact comparision, to be robust to floating point calculations.
        """
        return np.allclose(robot.joint_2_pos(), goal, atol=0.25)

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
    height = 1000
    width = 1000

    
    robot_origin  =(450, 450) #= (int(width / 2), int(height / 2))
    num_goals=1
    theta=0
    for iter in range(num_goals):

        theta=theta+(iter+1)*36
        goal = generate_random_goal(Robot.min_reachable_radius(), Robot.max_reachable_radius())
        # goal = (math.cos(theta)*Robot.min_reachable_radius(), math.sin(theta)*Robot.min_reachable_radius())
        # goal=(25.0,5)
        robot = Robot()
        controller = Controller(goal)
        world = World(width, height, robot, robot_origin, goal)
        # world.add_obstacle("circle",(200.0,200.0,15.0))
        vis = Visualizer(world)

        runner = Runner(robot, controller, world, vis)

        try:
            runner.run()
        except AssertionError as e:
            print(f'ERROR: {e}, Aborting.')
        except KeyboardInterrupt:
            pass


if __name__ == '__main__':
    main()
