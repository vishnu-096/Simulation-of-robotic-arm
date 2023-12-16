import time
from typing import List, Tuple, Union
import numpy as np
import pygame
import pygame_gui
from helper_functions import *
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from Robot import Robot
import logging_config
import logging

logging_config.setup_logging()
# Get a logger instance
logger = logging.getLogger(__name__)

class Obstacle:
    def __init__(self, obstacle_type: str, dimensions: Union[Tuple[float], Tuple[float, float]]) -> None:
        """Initialize an obstacle with its type and dimensions.

        Args:
            obstacle_type (str): Type of the obstacle, "circle" or "rectangle".
            dimensions (Union[Tuple[float], Tuple[float, float]]): Dimensions of the obstacle.
        """
        self.obstacle_type = obstacle_type
        self.dimensions = dimensions

class World:
    def __init__(
        self,
        width: int,
        height: int,
        robot_origin: Tuple[int, int],
        goal: Tuple[int, int],
        robot
    ) -> None:
        """Initialize the world with its dimensions, robot origin, goal, and robot.

        Args:
            width (int): Width of the world.
            height (int): Height of the world.
            robot_origin (Tuple[int, int]): Initial position of the robot.
            goal (Tuple[int, int]): Goal position for the robot.
            robot: The robot object in the world.
        """
        self.width = width
        self.height = height
        self.robot_origin = robot_origin
        self._goal = goal
        self.obstacles: List[Obstacle] = []
        self.robot = robot

    @property
    def goal(self) -> Tuple[int, int]:
        return self._goal

    @goal.setter
    def goal(self, new_goal: Tuple[int, int]) -> None:
        self._goal = new_goal
        if not self.robot.update_goal(new_goal):  # Call the robot's update_goal method
            logger.error("Error with setting goal. Out of limit")
        
    def convert_to_display(
            self, point: Tuple[Union[int, float], Union[int, float]]) -> Tuple[int, int]:
        """Convert a point from the robot coordinate system to the display coordinate system.

        Args:
            point (Tuple[Union[int, float], Union[int, float]]): Point in the robot coordinate system.

        Returns:
            Tuple[int, int]: Point in the display coordinate system.
        """
        robot_x, robot_y = point
        offset_x, offset_y = self.robot_origin
        return int(offset_x + robot_x), int(offset_y - robot_y)

    def add_obstacle(
            self, obstacle_type: str, dimensions: Union[Tuple[float], Tuple[float, float]]) -> None:
        """Add obstacle to the world.

        Args:
            obstacle_type (str): Type of the obstacle, "circle" or "rectangle".
            dimensions (Union[Tuple[float], Tuple[float, float]]): Dimensions of the obstacle.
        """
        logger.debug("Adding obstacle:", dimensions)
        obstacle = Obstacle(obstacle_type, dimensions)
        self.obstacles.append(obstacle)

    def generate_landmarks(self):
        """Generate landmarks based on the obstacles in the world.

        Returns:
            List[List[Union[float, int]]]: List of landmarks.
        """
        landmarks = [list(obstacle.dimensions) for obstacle in self.obstacles]
        logger.debug(landmarks)

        self.robot.update_true_landmarks(landmarks)
        return landmarks


class Visualizer:
    # Class attributes
    BLACK: Tuple[int, int, int] = (0, 0, 0)
    RED: Tuple[int, int, int] = (255, 0, 0)
    WHITE: Tuple[int, int, int] = (255, 255, 255)
    GREEN: Tuple[int, int, int] = (0, 255, 0)
    BLUE: Tuple[int, int, int] = (0, 0, 255)
    GRAY: Tuple[int, int, int] = (200, 200, 200)
    YELLOW: Tuple[int, int, int] = (0, 255, 255)
    VIOLET: Tuple[int, int, int] = (128, 0, 128)

    # Constants for button and label positions
    UI_BUTTON_WIDTH = 1000
    UI_BUTTON_HEIGHT = 40
    UI_BUTTON_MARGIN_X = 100  # Horizontal margin between buttons
    UI_BUTTON_MARGIN_Y = 60  # Vertical margin between buttons
    GOAL_INPUT_BOX_WIDTH = 100
    GOAL_INPUT_BOX_MARGIN_X = 700

    color_inactive = pygame.Color('lightskyblue3')
    color_active = pygame.Color('dodgerblue2')    
    
    plot_rect = pygame.Rect(800, 200, 300, 200)
    ui_button_rect : Tuple[int, int, int, int] =  (0, 0, 1000, 40)

    # Map dimensions on the screen
    map_start: Tuple[int, int] = (100, 800)
    map_width = 600
    map_height = 600

    #Robot World dimenstions on the screen
    world_start: Tuple[int, int] = (100, 100)
    clock = pygame.time.Clock()
    time_delta = clock.tick(60) / 1000.0

    def __init__(self, world: World, screen_size : Tuple[int, int]) -> None:
        """Initialize the visualizer with the world.

        Args:
            world: The world object to visualize.
        """
        pygame.init()
        pygame.font.init()
        self.world = world
        self.screen = pygame.display.set_mode(screen_size)
        pygame.display.set_caption('Gherkin Challenge')
        self.font = pygame.font.SysFont('freesans', 25)

        self.grid_visual = Grid(self.map_height, self.map_width, 1, 1)
        self.time = 0
        self.manager = pygame_gui.UIManager(screen_size)
        
        # BUTTONS
            
        self.start_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((self.UI_BUTTON_MARGIN_X, self.UI_BUTTON_MARGIN_Y), (180, self.ui_button_rect[3])),
            text='Start Simulation',
            manager=self.manager
        )
        
        # Button that starts the plot : ToDo selection of plots        
        self.plot_button = pygame_gui.elements.UIButton(
            relative_rect=pygame.Rect((self.UI_BUTTON_MARGIN_X*3, self.UI_BUTTON_MARGIN_Y), (150, self.ui_button_rect[3])),
            text='Plot Graphs',
            manager=self.manager
        )

        #Create input box for setting goal position
        # Input box
        self.input_box = pygame_gui.elements.UITextEntryLine(
            relative_rect=pygame.Rect((self.GOAL_INPUT_BOX_MARGIN_X, self.UI_BUTTON_MARGIN_Y), (100, 30)),
            manager=self.manager
        )

        # Create a label for goal input
        label = pygame_gui.elements.UILabel(
            relative_rect=pygame.Rect((self.GOAL_INPUT_BOX_MARGIN_X-200, self.UI_BUTTON_MARGIN_Y), (150, 30)),
            text='Goal Position "x,y" :',
            manager=self.manager
        )   
        
        self.axes = {}   # Dictionary where keys would be tags like 'pos' or 'vel' and value would be axis object to plot them  
        self.initialize_figures() # Initializes surface and other config to plot the graphs using matplotlib

        # Create a surface for robot map grid
        self.static_surface = pygame.Surface((self.map_width, self.map_height))
        self.static_surface = self.static_surface.convert()
        self.static_surface.fill(self.WHITE)      


        # Create a surface for robot world
        self.world_surface_top = pygame.Surface((self.world.width, self.world.height))
        self.world_surface_top = self.static_surface.convert()
        self.world_surface_top.fill(self.WHITE)

        
        # Flags
        self.start_sim = False
        self.load_sim = False
        self.plot_flag = False  # Default plotting is turned off

        
    def initialize_figures(self) -> None:
        """Initialize Matplotlib figures and axes."""

        self.fig, self.axes = plt.subplots(2, 1, figsize=(8,8))
        self.canvas = FigureCanvas(self.fig)
        self.canvas.draw()
        self.renderer = self.canvas.get_renderer()

    """Plot graphs based on robot data.

    Args:
        data_tag: Tag for the data to be plotted.
        robot: The robot object.
    """
    def plot_graphs(self, data_tag, robot):

        #plotting a window length of 15. Last 15 time samples
        window_len = 15
    
        if len(robot.all_angles[0]) < window_len:
            window_len = len(robot.all_angles[0])
            
        time_window = np.linspace(max(self.time - self.time_delta * window_len, 0), self.time, window_len)
        if data_tag == 'pos':
            ax = self.axes[0]
            ax.plot(time_window, robot.all_angles[0][-window_len:], '-ob', label='Theta 0')
            ax.plot(time_window, robot.all_angles[1][-window_len:], '-og', label='Theta 1')
            ax.plot(time_window, robot.all_angles[2][-window_len:], '-or', label='Theta 2')
            ax.set_ylim(-1.6, 1.6)
            ax.set_xlabel('Time')
        
        if data_tag == 'vel':
            ax = self.axes[1]
            vel_data = np.array(robot.all_joint_velocities[-window_len:])
            ax.plot(time_window, vel_data[:, 0], '-ob', label='Theta 0_dot')
            ax.plot(time_window, vel_data[:, 1], '-og', label='Theta 1_dot')
            ax.plot(time_window, vel_data[:, 2], '-or', label='Theta 2_dot')
            ax.set_ylim(-2, 2)
            ax.set_xlabel('Time')
        
        ax.legend()

        self.canvas.draw()

        # Render Matplotlib plot onto Pygame surface


    def display_world(self) -> None:
        """Display the world, including the goal and obstacles."""
        surface = self.world_surface_top

        goal = self.world.convert_to_display(self.world.goal)
        pygame.draw.circle(surface, self.RED, goal, 6)
        for obstacle in self.world.obstacles:
            self.display_obstacle(obstacle)

    def display_obstacle(self, obstacle: Obstacle) -> None:
        """Display an obstacle on the screen.

        Args:
            obstacle: The obstacle object.
        """
        obstacle_position = self.world.convert_to_display(obstacle.dimensions[:2])

        if obstacle.obstacle_type == "circle":
            radius = int(obstacle.dimensions[2])
            pygame.draw.circle(self.world_surface_top, self.BLACK, obstacle_position, radius)
        elif obstacle.obstacle_type == "rectangle":
            width, height = map(int, obstacle.dimensions[2:])
            pygame.draw.rect(self.world_surface_top, self.BLACK, (obstacle_position[0], obstacle_position[1], width, height))
        else:
            raise ValueError("Unsupported obstacle type")

       
    def display_robot_map(self, map, path):    # ld_x:List, ld_y:List, eof:Tuple(int, int))->None:
        """Display the robot's map on the screen.

        Args:
            map: The robot's map object.
        """
        # Update display grid with new scan regions
        for cell in map.new_scan_grid:
            color = self.GREEN
            overwrite = False
            self.grid_visual.draw_cell(cell[0], cell[1], color, self.static_surface, overwrite)

        # Update display grid with new obstacles
        for cell in map.new_landmarks_grid:
            color = self.BLUE
            overwrite = True
            self.grid_visual.draw_cell(cell[0], cell[1], color, self.static_surface, overwrite)

        #Update the sensor position on display grid
        sensor_grid_x = int(self.map_height/2 - map.sensor.sensor_position[1])
        sensor_grid_y = int(map.sensor.sensor_position[0] + self.map_width/2)
        color = self.VIOLET
        overwrite = True

        # Display sensor position as a bigger grid giving blocksize
        self.grid_visual.draw_cell(sensor_grid_x-1
                                   , sensor_grid_y, color, self.static_surface, overwrite, block_width=3)


        if path:
            overwrite = True
            for x,y in path:
                grid_x = int(self.map_height/2 - y)
                grid_y = int(x + self.map_width/2)
                color = self.BLACK
                self.grid_visual.draw_cell(grid_x
                                   ,grid_y, color, self.static_surface, overwrite)                
    """Display the robot on the screen.

    Args:
        robot: The robot object.
    """
    def display_robot(self, robot: Robot) -> None:
        
        start_joint = self.world.robot_origin
        if not robot.is_3d:
            surface = self.world_surface_top

            num_joints = len(robot.joint_angles)
            for iter in range(num_joints):
                end_joint = self.world.convert_to_display(robot.forward(robot.joint_angles, iter+1))
                pygame.draw.circle(surface, self.BLACK, start_joint, 6)
                pygame.draw.line(surface, self.BLACK, start_joint, end_joint, 4)
                start_joint = end_joint
            pygame.draw.circle(surface, self.BLACK, start_joint, 6)
        

    def display_robot_path(self, robot: Robot) -> None:
        if robot.planned_path is None:
            return
          
        surface = self.world_surface_top
        
        path = robot.interpolated_path
        for iter in range(path.shape[0] - 1):
            point1 = self.world.convert_to_display( (path[iter, 0], path[iter, 1]) )
            point2 = self.world.convert_to_display( (path[iter+1, 0], path[iter+1, 1]) )
             
            pygame.draw.aaline(surface, self.VIOLET, point1, point2, 3)

    def update_display(self, robot: Robot, success: bool, collision_detected: bool) -> bool:
        """Update the display with the latest information.

        Args:
            robot: The robot object.
            success: True if the robot has reached the goal, False otherwise.
            collision_detected: True if collision is detected, False otherwise.

        Returns:
            bool: True to if simulation is on, False if simulation is off.
            bool: True to continue updating, False to exit the program.

        """
        self.time = self.time + self.time_delta

        active = False
        for event in pygame.event.get():
            self.manager.process_events(event)

            # Keypress
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return self.start_sim, False
         
            # Window Close Button Clicked
            if event.type == pygame.QUIT:
                return self.start_sim, False
            elif event.type == pygame.USEREVENT:

                if event.user_type == pygame_gui.UI_TEXT_ENTRY_FINISHED:
                    #get the text and extract x, y of goal
                    x_g, y_g = map(float, event.text.split(','))
                    #Set goal in the world which would set robot's goal
                    self.world.goal = [x_g, y_g]
                            
            if event.type == pygame_gui.UI_BUTTON_PRESSED:
                if event.ui_element == self.plot_button:
                    self.plot_flag = not self.plot_flag

                if event.ui_element == self.start_button:
                    self.start_sim = not self.start_sim
            
            
        self.screen.fill(self.BLACK)

        self.display_world()
        
        self.display_robot(robot)
        
        self.display_robot_path(robot)

        self.display_robot_map(robot.map, robot.planned_path)

        if collision_detected:
            text = self.font.render('Collision!', True, self.RED)
            self.screen.blit(text, (10, 1))
        if success:
            text = self.font.render('Success!', True, self.GREEN)
            self.screen.blit(text, (1, 1))
        
        if self.plot_flag:
            self.plot_graphs('pos', robot)
            self.plot_graphs('vel', robot)
            plot_surface = pygame.image.fromstring(self.canvas.tostring_rgb(), self.canvas.get_width_height(), "RGB")
            self.screen.blit(plot_surface, self.plot_rect)        
            for ax in self.axes:
                ax.cla()
            
        self.manager.update(self.time_delta)

        self.manager.draw_ui(self.screen)

        self.screen.blit(self.static_surface, self.map_start)                   
        
        self.screen.blit(self.world_surface_top, self.world_start)

        # clear surface for dynamic plotting
        self.world_surface_top.fill(self.WHITE)


        pygame.display.flip()

        return self.start_sim, True

    def cleanup(self) -> None:
        """Cleanup resources."""

        pygame.quit()

class Rectangle:
    def __init__(self, x, y, width, height, color=(0, 0, 0)):
        """Initialize a rectangle.

        Args:
            x, y, width, height: Position and dimensions of the rectangle.
            color: Color of the rectangle.
        """
        self.rect = pygame.Rect(x, y, width, height)
        self.color = color

class Grid:
    def __init__(self, rows, cols, rect_width, rect_height):
        """Initialize a grid.

        Args:
            rows, cols: Number of rows and columns in the grid.
            rect_width, rect_height: Width and height of each grid cell.
        """
        self.rows = rows
        self.cols = cols
        self.rect_width = rect_width
        self.rect_height = rect_height

        # Initialize the grid with None values
        self.grid = [[None] * cols for _ in range(rows)]

    def add_rectangle(self, row, col, color=(0, 0, 0)):
        """Add a rectangle to the grid.

        Args:
            row, col: Row and column where the rectangle is added.
            color: Color of the rectangle.
        """
        # Create a Rectangle instance and add it to the grid
        x = col * (self.rect_width)
        y = row * (self.rect_height)
        self.grid[row][col] = Rectangle(x, y, self.rect_width, self.rect_height, color)

    def draw_cell(self, row, col, color, surface, overwrite, block_width=1):
        """Draw a cell on the surface.

        Args:
            row, col: Row and column of the cell to draw.
            color: Color of the cell.
            surface: Surface to draw on.
            overwrite: Flag to determine whether to overwrite existing content.
        """

        if block_width>1:
            delta_grid = block_width//2
        else:
            delta_grid =0
            
        for rel_row in range(row-delta_grid, row + delta_grid+1):
            for rel_col in range(col-delta_grid, col+delta_grid+1):
                x = rel_col * self.rect_width
                y = rel_row * self.rect_height

                if self.grid[rel_row][rel_col] is None or overwrite:
                    self.grid[rel_row][rel_col] = Rectangle(x, y, self.rect_width, self.rect_height, color)
                    pygame.draw.rect(surface, self.grid[rel_row][rel_col].color, self.grid[rel_row][rel_col].rect)

    def draw_grid(self, surface):
        """Draw the entire grid on the surface.

        Args:
            surface: Surface to draw on.
        """
        # Draw rectangles on the surface based on the grid
        for row in range(self.rows):
            for col in range(self.cols):
                if self.grid[row][col] is not None:
                    pygame.draw.rect(surface, self.grid[row][col].color, self.grid[row][col].rect)



