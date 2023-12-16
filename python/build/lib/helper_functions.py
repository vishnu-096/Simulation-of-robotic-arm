import numpy as np
import math
from typing import Tuple, List

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

def calculate_distance(x1, y1, x2, y2 ):
    return math.sqrt((x2 - x1)**2 + (y2 - y1)**2) 