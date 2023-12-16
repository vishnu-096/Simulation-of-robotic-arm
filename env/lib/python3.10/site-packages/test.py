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


"""
Simulation of a sensor with 180 degree field of view and defined sensor radius.
This is used to populate points in the grid map of robot.
"""
class Sensor:
    def __init__(self, landmarks) -> None:
        self.sensor_radius = 20
        self.landmarks = landmarks  # Landmarks with respect to robot's frame. Only used as true reference during perception
        self._sensor_position : List[float] = [0., 0.]   # position in robot's frame
        self.sensor_orientation = 0
        self.sensor_fov = np.pi
        self.sensed_frontier = set() #unique frontiers x and y position in robot's frame detected by the sensor


    @property
    def sensor_position(self) -> List[float]:
        return self._sensor_position    
    
    @sensor_position.setter
    def sensor_position(self, values: List[float]) -> None:
        # Additional logic, validation, or constraints can be added here
        # For example, checking if the length of the provided list is 3
        self._sensor_position = values
        if values[0] ==0:
            self.sensor_orientation = math.pi/2
        self.sensor_orientation = math.atan2(values[1], values[0])

    def perceive_environment(self):
        """The sensor is assumed to sense in a semi circular area with given sensory radius
        """

        #First check which all landmarks has a chance of intersecting with sensor's FOV
        for landmark in self.landmarks:
            x,y, *rest = landmark
            # print(x,y,rest)
            if rest==[]:
                continue
            else:
                r=rest[0]
            print("landmark ",x ,y, r)

            #check if the landmark is in the same direction of sensor FOV
            # print("land mark:", x, y)
            angle_s = self.sensor_orientation
            print(self.sensor_orientation)

            T = np.array([
            [np.cos(angle_s), -np.sin(angle_s), self.sensor_position[0]],
            [np.sin(angle_s), np.cos(angle_s), self.sensor_position[1]],
            [0, 0, 1]
            ])
            T_inv = np.linalg.inv(T)

            angles = np.linspace(0,2*np.pi,360)
            starting_angle = -self.sensor_fov/2
            ending_angle = self.sensor_fov/2
            for theta in angles:
                point = np.array([x + r*math.cos(theta),
                                  y + r*math.sin(theta),
                                  1])
                
                transformed_point = np.dot(T_inv, point)
                new_angle = math.atan2(transformed_point[1], transformed_point[0])
                # if transformed_point[0]>0:
                #     if  calculate_distance(point[0], point[1], self.sensor_position[0], self.sensor_position[1]) < self.sensor_radius:
   
            # angles = np.linspace(0,2*np.pi,360)

            # for theta in angles:
            #     point = np.array([x + r*math.cos(theta),
            #                       y + r*math.sin(theta),
            #                       1])
                
                dist = calculate_distance(point[0], point[1], self.sensor_position[0], self.sensor_position[1])

                if   dist< self.sensor_radius and new_angle>starting_angle and new_angle < ending_angle:
                    
                        sensed_point = (int(point[0]), int(point[1]))
                        self.sensed_frontier.add(sensed_point)                        


"""
Local grid map class for the robot. Generated from the sensor based perception. 

"""
class Mapping:
    def __init__(self, sensor) -> None:
        self.map_width = 400 # TODO
        self.map_height = 400
        # x becomes rows and y is columns. Convert such that robot is in the center
        self.grid2D = np.zeros((self.map_height, self.map_width))  
        self.max_rows=0
        self.max_cols=0
        self.rob_position=[0,0]
        self.sensor_scans = set()
        self.landmark_scans = set()
        self.new_landmarks_grid = []
        self.new_scan_grid = [] 
        self
        self.sensor = sensor

    def update_map(self):
        print("frontiers of landmarks : x")
        self.new_landmarks_grid = []
        self.new_scan_grid =[]
        for point in self.sensor.sensed_frontier:
            x,y = point
            x_grid = -int(y-self.map_height/2)
            y_grid = int(x+self.map_width/2)
            self.new_landmarks_grid.append([x_grid, y_grid])
            self.grid2D[ x_grid, y_grid] = 2
            self.landmark_scans.add((x_grid, y_grid))
        angles = np.linspace(self.sensor.sensor_orientation  - self.sensor.sensor_fov/2, self.sensor.sensor_orientation + self.sensor.sensor_fov/2, 180)
        x_s, y_s = self.sensor.sensor_position
        for r in range(0, self.sensor.sensor_radius, 1):        
            
            for angle in angles:
                x_grid= -int(y_s + r*math.sin(angle) - self.map_height/2)
                y_grid = int(x_s + r*math.cos(angle) + self.map_width/2)
                if self.grid2D[x_grid, y_grid] != 2:
                    self.grid2D[x_grid, y_grid] = 1
                self.new_scan_grid.append([x_grid, y_grid])    
                self.sensor_scans.add((x_grid, y_grid))
        sensor_grid_x = int(self.map_height//2 -self.sensor.sensor_position[1]) 
        sensor_grid_y = int(self.sensor.sensor_position[0]+self.map_width//2)
        self.grid2D[sensor_grid_x, sensor_grid_y] = 3
        

        # print("landmarkss ",self.landmark_update)
        # print("sensorSss ",self.sensor_update)

        # self.sensor_update= []
        # self.landmark_update = []

class Robot:
    JOINT_LIMITS = [-6.28, 6.28]
    MAX_VELOCITY = 15
    MAX_ACCELERATION = 50
    DT = 0.033
    
    link_1: float = 75.  # pixels
    link_2: float = 50.  # pixels
    link_3: float = 25.  # pixels    

    link_lengths=[link_1, link_2, link_3]
    analytical_ik = True # True for numerical based IK
    
    def __init__(self) -> None:
        # internal variables
        self.all_theta_0: List[float] = []
        self.all_theta_1: List[float] = []
        self.all_theta_2: List[float] = []
        self.all_angles: List[List[float]] = [[], [], []]


        self.joint_angles=[0., 0., 0.]
        self._joint_angles: List[float] = [0., 0., 0.]
        self.eof_pos = self.forward(self.joint_angles, 3)
        self.link_lengths=[self.link_1, self.link_2, self.link_3]
        self.map = None
    # Getter/Setter for joint_angles
    @property
    def joint_angles(self) -> List[float]:
        return self._joint_angles    
    
    @joint_angles.setter
    def joint_angles(self, values: List[float]) -> None:
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
            assert self.max_acceleration(self.all_theta_0) < self.MAX_ACCELERATION, \
                f'Joint 0 Accel {self.max_acceleration(self.all_angles[iter])} exceeds acceleration limit'

        # Set the internal _joint_angles attribute
        self._joint_angles = values
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
    #Perception
    def percieve_landmarks(self):
        self.map.sensor.perceive_environment()
        self.map.update_map()
        print("Sensor Orientation",np.rad2deg(self.map.sensor.sensor_orientation))
        print("Sensor Position",self.map.sensor.sensor_position)


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
    def forward(cls, angles:list, joint_num: int) -> Tuple[float, float]:
        """
        Compute the x, y position of the end of the links from the joint angles
        """
        x=0
        y=0
        # theta=[theta_0, theta_0+theta_1, theta_0+theta_1+theta_2]
        theta=0
        for iter in range(joint_num):
            theta += angles[iter]
            x += cls.link_lengths[iter]*np.cos(theta)
            y += cls.link_lengths[iter]*np.sin(theta)

        return x, y

    @classmethod
    def inverse(cls, x: float, y: float, psi: float) -> Tuple[float, float]:
        """
        Compute the joint angles from the position of the end of the links
        """
        if cls.analytical_ik:
            x2 = x - cls.link_3*math.cos(psi)
            y2 = y - cls.link_3*math.sin(psi)

            theta_1 = np.arccos((x2 ** 2 + y2 ** 2 - cls.link_1 ** 2 - cls.link_2 ** 2)
                                / (2 * cls.link_1 * cls.link_2))
            theta_0 = np.arctan2(y2, x2) - \
                np.arctan((cls.link_2 * np.sin(theta_1)) /
                        (cls.link_1 + cls.link_2 * np.cos(theta_1)))
            theta_2 = psi - theta_0 - theta_1
            return theta_0, theta_1, theta_2

        else:
            pass
    #         initial_guess = [0, 0.2, 0]

    #     # Desired end-effector position
    #         target_position = np.array([x, y])

    # # Use a numerical optimizer to find joint angles
    #         result = minimize(inverse_kinematics_objective, initial_guess, args=(target_position,))
    #         joint_angles = result.x


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
        robot_origin: Tuple[int, int],
        goal: Tuple[int, int]
    ) -> None:
        self.width = width
        self.height = height
        self.robot_origin = robot_origin
        self.goal = goal
        self.obstacles: List[Obstacle] = []

    def convert_to_display(
            self, point: Tuple[Union[int, float], Union[int, float]]) -> Tuple[int, int]:
        """
        Convert a point from the robot coordinate system to the display coordinate system
        """
        robot_x, robot_y = point
        offset_x, offset_y = self.robot_origin

        # print(robot_y)
        return int(offset_x + robot_x), int(offset_y - robot_y)

    def add_obstacle(
            self, obstacle_type: str, dimensions: Union[Tuple[float], Tuple[float, float]]) -> None:
        """
        Add obstacle to the world. Dimensions and type of obstacles need to provided
        type: "circle" or "rectangle"
        dimensions: as tuple
        
        """
        print("Adding obstacle:", dimensions)
        obstacle = Obstacle(obstacle_type, dimensions)
        self.obstacles.append(obstacle)

    def generate_landmarks(self):
        landmarks =[]
        for obstacle in self.obstacles:
             print(obstacle.dimensions)
             landmarks.append(list(obstacle.dimensions))
        # landmarks.append(self.goal)
        print(landmarks)
        return landmarks
    
    def check_collisions(self, robot):
        
        joint1_pos = self.convert_to_display(robot.forward(robot.joint_angles, 1))
        joint2_pos = self.convert_to_display(robot.forward(robot.joint_angles, 2))

        #check collisions with circle obstacles
        for obstacle in self.obstacles:
            if obstacle.obstacle_type == "circle":
                if is_line_circle_intersection(joint1_pos, joint2_pos, obstacle.dimensions):
                    return True

        return False

class Rectangle:
    def __init__(self, x, y, width, height, color=(0, 0, 0)):
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color

class Grid:
    def __init__(self, rows, cols, rect_width, rect_height):
        self.rows = rows
        self.cols = cols
        self.rect_width = rect_width
        self.rect_height = rect_height

        # Initialize the grid with None values
        self.grid = [[None] * cols for _ in range(rows)]

    def add_rectangle(self, row, col, color=(0, 0, 0)):
        # Create a Rectangle instance and add it to the grid
        x = col * (self.rect_width )
        y = row * (self.rect_height )
        self.grid[row][col] = Rectangle(x, y, self.rect_width, self.rect_height, color)

    def draw_cell(self, row, col, color, surface, overwrite):
        x = col * (self.rect_width )
        y = row * (self.rect_height )
        if self.grid[row][col] is None or overwrite:
            self.grid[row][col] = Rectangle(x, y, self.rect_width, self.rect_height, color)
            pygame.draw.rect(surface, self.grid[row][col].color, self.grid[row][col].rect)

    def draw_grid(self, surface):
        # Draw rectangles on the surface based on the grid
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col] is not None:
                    pygame.draw.rect(surface, self.grid[row][col].color, self.grid[row][col].rect)


class Visualizer:
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    RED: Tuple[int, int, int] = (255, 0, 0)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    YELLOW: Tuple[int, int, int] = (0, 255, 0)
    BLUE: Tuple[int, int, int] = (0, 0, 255)

    plot_rect = pygame.Rect(600, 0, 300, 200)
    ui_button_rect : Tuple[int, int, int, int] =  (0, 0, 1000, 30)
    map_start : Tuple[int, int] = (100,700)
    map_width = 400
    map_height = 400




    clock = pygame.time.Clock()
    time_delta = clock.tick(60) / 1000.0

    def __init__(self, world: World) -> None:
        """
        Note: while the Robot and World have the origin in the center of the
        visualization, rendering places (0, 0) in the top left corner.
        """
        pygame.init()
        pygame.font.init()
        self.world = world
        self.screen = pygame.display.set_mode((world.width, world.height))
        pygame.display.set_caption('Gherkin Challenge')
        self.font = pygame.font.SysFont('freesans', 30)

        self.grid_visual = Grid(self.map_height, self.map_width, 1, 1)
        self.time=0
        self.manager = pygame_gui.UIManager((world.width, world.height))
        
        # Button that starts the plot : ToDO selection of plots
        self.plot_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((self.ui_button_rect[0]+5, self.ui_button_rect[0]+50), (50, self.ui_button_rect[3])),
            text='Plot Graphs',
            manager=self.manager
        )
        self.axes={}   #dictionary where keys would be tags like 'pos' or 'vel' and value wwould be axis object to plot them  
        self.initialize_figures() 
        self.plot_flag = False # Default plotting is turned off


        self.static_surface = pygame.Surface((self.map_width, self.map_height))
        self.static_surface = self.static_surface.convert()
        self.static_surface.fill((0, 0, 0))      

        # Create a rectangle
        rect_width, rect_height = 50, 50
        rect_color = (255, 0, 0)  # Red color
        self.rectangle = pygame.Rect(50, 50, rect_width, rect_height)

        # Draw the rectangle on the static surface
        pygame.draw.rect(self.static_surface, rect_color, self.rectangle)


    def initialize_figures(self)->None:
        self.fig = plt.figure()
        ax = self.fig.add_subplot(211)
        self.axes['pos'] =ax
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.renderer = self.canvas.get_renderer()

    def plot_graphs(self, data_tag, robot):
        window_len=15
        if data_tag == 'pos':
            if len(robot.all_angles[0])<window_len:
                window_len =len(robot.all_angles[0])
            time_window=np.linspace(max(self.time-self.time_delta*window_len, 0), self.time, window_len)
            ax=self.axes[data_tag]
            ax.plot(time_window, robot.all_angles[0][-window_len:], '-ob', label='Theta 0')
            ax.plot(time_window, robot.all_angles[1][-window_len:], '-og', label='Theta 1')
            ax.plot(time_window, robot.all_angles[2][-window_len:], '-or', label='Theta 2')
            ax.legend()

            self.canvas.draw()

            # Render Matplotlib plot onto Pygame surface
            plot_surface = pygame.image.fromstring(self.canvas.tostring_rgb(), self.canvas.get_width_height(), "RGB")
            self.screen.blit(plot_surface, self.plot_rect)        
            ax.cla()

    def display_world(self) -> None:
        """
        Display the world
        """
        goal = self.world.convert_to_display(self.world.goal)
        pygame.draw.circle(self.screen, self.RED, goal, 6)
        for obstacle in self.world.obstacles:
            self.display_obstacle(obstacle)


    def display_obstacle(self, obstacle: Obstacle) -> None:
        """
        Display the obstacle
        """
        obstacle_position = self.world.convert_to_display(obstacle.dimensions[:2])

        if obstacle.obstacle_type == "circle":
            radius = int(obstacle.dimensions[2])
            pygame.draw.circle(self.screen, self.BLACK, obstacle_position, radius)
        elif obstacle.obstacle_type == "rectangle":
            width, height = map(int, obstacle.dimensions[2:])
            pygame.draw.rect(self.screen, self.BLACK, (obstacle_position[0], obstacle_position[1], width, height))
        else:
            raise ValueError("Unsupported obstacle type")
    
    def display_robot_map(self, map):#ld_x:List, ld_y:List, eof:Tuple(int, int))->None:
        
        # print(map.grid2D) #Set the size of the grid block
        blockSize =  2

        start_x = self.map_start[0]
        start_y = self.map_start[1]
        end_x = start_x+self.map_width-1
        end_y =  start_y+self.map_height-1

        print("black")
        print(end_y)
        print(self.map_height)
        cnt=0
        x_coord =[]
        y_coord =[]


        for cell in map.new_scan_grid:
            print(cell)
            color = self.YELLOW
            overwrite = False
            self.grid_visual.draw_cell(cell[0], cell[1], color, self.static_surface, overwrite)

        #update display grid with new scan regions
        for cell in map.new_landmarks_grid:
            print(cell)
            color = self.BLUE
            overwrite = True
            self.grid_visual.draw_cell(cell[0], cell[1], color, self.static_surface, overwrite)


    def display_robot(self, robot: Robot) -> None:
        """
        Display the robot
        """
        j0 = self.world.robot_origin
        j1 = self.world.convert_to_display(robot.forward(robot.joint_angles, 1))
        j2 = self.world.convert_to_display(robot.forward(robot.joint_angles, 2))
        j3 = self.world.convert_to_display(robot.forward(robot.joint_angles, 3))

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

                if event.ui_element == self.plot_button:
                    self.plot_flag = not self.plot_flag


        self.screen.fill(self.WHITE)

        self.display_world()
        
        self.display_robot(robot)

        # local_robot_grid = robot.map.sensor.
        # print("ehjeeeeihwkehajkhn")
        # print(robot.map.sensor_update)
        self.display_robot_map(robot.map)

        if collission_detected:
            text = self.font.render('Collission!', True, self.BLACK)
            self.screen.blit(text, (10, 1))
        if success:
            text = self.font.render('Success!', True, self.BLACK)
            self.screen.blit(text, (1, 1))
        
        if self.plot_flag:
            self.plot_graphs('pos', robot)
        self.manager.update(self.time_delta)

        self.manager.draw_ui(self.screen)

        self.screen.blit(self.static_surface, self.map_start)

        pygame.display.flip()

        return True

    def cleanup(self) -> None:
        pygame.quit()


class Controller:
    def __init__(self, goal: Tuple[int, int]) -> None:
        self.goal = goal
        eof_orrientation=math.atan2(goal[1],goal[0])
        print("eof Orientatin ", eof_orrientation)
        self.goal_theta_0, self.goal_theta_1, self.goal_theta_2 = Robot.inverse(self.goal[0], self.goal[1], eof_orrientation)
        print("Solution :", self.goal_theta_0, self.goal_theta_1, self.goal_theta_2)
        print("Goal :", self.goal)

    def step(self, robot: Robot) -> Robot:
        """
        Simple P controller
        """
        target_angles = np.array([self.goal_theta_0, self.goal_theta_1, self.goal_theta_2])
        theta_error = target_angles - np.array(robot.joint_angles) 

        robot.joint_angles = (np.array(robot.joint_angles) + theta_error/10).tolist()
        robot.eof_pos = robot.forward(robot.joint_angles, 3)
        robot.map.sensor.sensor_position = robot.eof_pos
        robot.percieve_landmarks()
        time.sleep(1)
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
    height = 1000
    width = 1000

    
    robot_origin = (int(width / 2), int(height / 2))
    num_goals=1
    theta=0
    for iter in range(num_goals):

        # goal = generate_random_goal(Robot.min_reachable_radius(), Robot.max_reachable_radius())
        # goal = (math.cos(theta)*Robot.min_reachable_radius(), math.sin(theta)*Robot.min_reachable_radius())
        goal=(25.0,50.0)
        robot = Robot()
        controller = Controller(goal)
        world = World(width, height, robot_origin, goal)
        world.add_obstacle("circle",(30.0,60.0,15.0))
        world.add_obstacle("circle",(100.0,60.0,25.0))
        true_world_landmarks = world.generate_landmarks()
        print(true_world_landmarks)
        print(world.robot_origin)
        # for ld_iter in range(len(true_world_landmarks)):
        #     true_world_landmarks[ld_iter][0]-= world.robot_origin[0]
        #     true_world_landmarks[ld_iter][1] -= world.robot_origin[1]

        robot_sensor =Sensor(true_world_landmarks)
        robot_local_map = Mapping(robot_sensor)
        robot.map = robot_local_map
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
