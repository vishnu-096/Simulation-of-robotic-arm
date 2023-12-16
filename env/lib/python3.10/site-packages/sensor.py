import pygame
import sys
import random

# Initialize Pygame
pygame.init()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Pygame Circles Example")

# Set up colors
black = (0, 0, 0)
white = (255, 255, 255)

# Create a rectangular surface
rectangular_surface = pygame.Surface((400, 100))
rectangular_surface.fill(black)

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill(black)
    rectangular_surface.fill(black)

    # Draw circles on the rectangular surface
    for _ in range(1):
        circle_radius = random.randint(10, 50)
        circle_position = (random.randint(0, width - 2 * circle_radius),
                           random.randint(0, height - 2 * circle_radius))
        pygame.draw.circle(rectangular_surface, (255,0,0), circle_position, circle_radius)

    # Blit the rectangular surface onto the main screen
    screen.blit(rectangular_surface, (0, 0))

    # Update the display
    pygame.display.flip()


# Quit Pygame
pygame.quit()
sys.exit()