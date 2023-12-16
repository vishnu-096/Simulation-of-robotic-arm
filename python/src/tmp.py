import pygame
import sys
import numpy as np

# Initialize Pygame
pygame.init()

# Set up display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Pygame Curvy Line")

# Define your x and y coordinates (replace with your data)
x_coordinates = np.linspace(50, width - 50, 20)
y_coordinates = np.sin(x_coordinates / (width - 100) * 4 * np.pi) * (height / 4) + height / 2

# Create clock object to control frame rate
clock = pygame.time.Clock()

# Main loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Clear the screen
    screen.fill((255, 255, 255))

    # Draw the curvy line using anti-aliased line
    for i in range(len(x_coordinates) - 1):
        pygame.draw.aaline(screen, (0, 0, 255), (x_coordinates[i], y_coordinates[i]), (x_coordinates[i+1], y_coordinates[i+1]))

    # Update the display
    pygame.display.flip()

    # Cap the frame rate
    clock.tick(30)

# Quit Pygame
pygame.quit()
sys.exit()
