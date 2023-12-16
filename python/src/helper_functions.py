import numpy as np
import math
from typing import Tuple, List
from math import cos, sin
import heapq
import matplotlib.pyplot as plt
import numpy as np

import numpy as np

def astar_search(grid, start, goal):
    rows, cols = len(grid), len(grid[0])
    directions = [(1, 0),(-1, 0), (0, 1), (0, -1), (-1, 1), (-1, -1), (1, -1), (1, 1)]  # Possible movements (down, up, right, left and diagonals)

    def is_valid(x, y):
        return 0 <= x < rows and 0 <= y < cols and grid[x][y] != 2

    def heuristic(node):
        return abs(node[0] - goal[0]) + abs(node[1] - goal[1])

    open_set = [(0, start)]  # Priority queue to store nodes to be explored
    came_from = {}  # Dictionary to store the parent of each node
    cost_so_far = {start: 0}  # Dictionary to store the cost to reach each node

    while open_set:
        current_cost, current_node = heapq.heappop(open_set)

        if current_node == goal:
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            return path[::-1]

        for dx, dy in directions:
            neighbor = current_node[0] + dx, current_node[1] + dy

            if is_valid(*neighbor):
                new_cost = cost_so_far[current_node] + 1  # Assuming each step has a cost of 1

                if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                    cost_so_far[neighbor] = new_cost
                    priority = new_cost + heuristic(neighbor)
                    heapq.heappush(open_set, (priority, neighbor))
                    came_from[neighbor] = current_node

    return None  # No path found





def is_line_circle_intersection(point1: Tuple[int, int], point2: Tuple[int, int], circle_dimensions: Tuple[int, int, int]) -> bool:
    center_x, center_y, radius = circle_dimensions

    # Vector from point1 to point2
    d_x = point2[0] - point1[0]
    d_y = point2[1] - point1[1]

    # Vector from point1 to the circle center
    f_x = point1[0] - center_x
    f_y = point1[1] - center_y

    a = d_x**2 + d_y**2
    b = 2 * (f_x * d_x + f_y * d_y)
    c = f_x**2 + f_y**2 - radius**2

    discriminant = b**2 - 4 * a * c

    if discriminant < 0:
        return False  # No intersection

    t1 = (-b + math.sqrt(discriminant)) / (2 * a)
    t2 = (-b - math.sqrt(discriminant)) / (2 * a)

    return 0 <= t1 <= 1 or 0 <= t2 <= 1


def is_line_rectangle_intersection(self, point1: Tuple[int, int], point2: Tuple[int, int], rectangle_dimensions: Tuple[int, int, int, int]) -> bool:
    """
    Check if a line intersects with a rectangle
    """
    rect_x, rect_y, width, height = map(int, rectangle_dimensions[2:])

    # Check if either endpoint is inside the rectangle
    if rect_x <= point1[0] <= rect_x + width and rect_y <= point1[1] <= rect_y + height:
        return True
    if rect_x <= point2[0] <= rect_x + width and rect_y <= point2[1] <= rect_y + height:
        return True

    # Check for line-segment intersection with each side of the rectangle
    return (
        self.is_line_segment_intersection(point1, point2, (rect_x, rect_y), (rect_x + width, rect_y)) or
        self.is_line_segment_intersection(point1, point2, (rect_x + width, rect_y), (rect_x + width, rect_y + height)) or
        self.is_line_segment_intersection(point1, point2, (rect_x + width, rect_y + height), (rect_x, rect_y + height)) or
        self.is_line_segment_intersection(point1, point2, (rect_x, rect_y + height), (rect_x, rect_y))
    )

def Jacobian(arm_type, joint_angles, link_lengths):
    
    if arm_type == "3_link_planar":
        q1 =joint_angles[0]
        q2 = joint_angles[1]
        q3 = joint_angles[2]
        l1 = link_lengths[0]
        l2 = link_lengths[1]
        l3 = link_lengths[2]
        
        J = np.array([
            [-l1*sin(q1) - l2*sin(q1 + q2) - l3*sin(q1 + q2 + q3),  -l2*sin(q1 + q2) - l3*sin(q1 + q2 + q3),  -l3*sin(q1 + q2 + q3)],
            [l1*cos(q1) + l2*cos(q1 + q2) + l3*cos(q1 + q2 + q3),  l2*cos(q1 + q2) + l3*cos(q1 + q2 + q3),  l3*cos(q1 + q2 + q3)],
            [0, 0, 0]
        
        ])
        
    return J

def forward_temp(angles, link_lengths):
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
    for iter in range(len(angles)):
        theta += angles[iter]
        x += link_lengths[iter] * np.cos(theta)
        y += link_lengths[iter] * np.sin(theta)
    return x, y, 0
def calculate_distance(point1, point2 ):
    return math.sqrt(sum((x - y)**2 for x, y in zip(point1, point2)))

def distance_to_line_segment(point, line_start, line_end):
    # Function to calculate the distance from a point to a line segment
    v = line_end - line_start
    w = point - line_start
    c1 = np.dot(w, v)
    c2 = np.dot(v, v)
    
    if c1 <= 0:
        return np.linalg.norm(point - line_start)
    
    if c2 <= c1:
        return np.linalg.norm(point - line_end)
    
    b = c1 / c2
    pb = line_start + b * v
    return np.linalg.norm(point - pb)

