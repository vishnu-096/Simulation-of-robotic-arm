import pygame
import pygame_gui

pygame.init()

# Set up Pygame screen
screen_width, screen_height = 800, 600
screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption('Real-time Text Entry Example')

# Create Pygame GUI manager
manager = pygame_gui.UIManager((screen_width, screen_height))

# Create UITextEntryLine
text_entry_line = pygame_gui.elements.UITextEntryLine(
    relative_rect=pygame.Rect((50, 50), (200, 30)),
    manager=manager
)

# Create UILabel to display real-time text
real_time_label = pygame_gui.elements.UILabel(
    relative_rect=pygame.Rect((50, 100), (200, 30)),
    text='',
    manager=manager
)

# Main loop
clock = pygame.time.Clock()
is_running = True

while is_running:
    time_delta = clock.tick(60) / 1000.0

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            is_running = False
        manager.process_events(event)

    manager.update(time_delta)

    # Update real-time label text
    real_time_label.set_text(f'Real-time Text: {text_entry_line.get_text()}')

    screen.fill((255, 255, 255))
    manager.draw_ui(screen)

    pygame.display.flip()

pygame.quit()
