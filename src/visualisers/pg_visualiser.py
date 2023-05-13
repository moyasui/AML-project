import pygame
import pandas as pd
import numpy as np

# Set up Pygame
pygame.init()
clock = pygame.time.Clock()

# Set up the display
width, height = 800, 600
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption('3D Function Animation')

# Read CSV file
dataset = pd.read_csv('src/csvs/lorenz.csv')
dataset.columns = dataset.columns.str.replace(' ', '')
data = dataset[dataset['particle'] == 0]
# Assuming the CSV file has columns named 'x', 'y', 'z', and 't'
x_values = data['x']
y_values = data['y']
z_values = data['z']
t_values = data['t']

# Set up animation parameters
num_frames = len(data)  # Number of frames in the animation
t_min = t_values.min()  # Minimum time value
t_max = t_values.max()  # Maximum time value

# List to store all the points
points = []

# Scaling factor for points
scale_factor = 10

# Animation loop
for frame in range(num_frames):
    # Clear the screen
    screen.fill((0, 0, 0))

    # Get the x, y, z, and t values for the current frame
    x = x_values[frame]
    y = y_values[frame]
    z = z_values[frame]
    t = t_values[frame]

    # Normalize the time value
    normalized_t = (t - t_min) / (t_max - t_min)

    # Compute the position based on x and y values
    pos_x = int(x * scale_factor) + width // 2
    pos_y = int(y * scale_factor) + height // 2

    # Compute the size based on the z value
    size = int(z * scale_factor)

    # Compute the color based on the normalized time value
    color_r = 200 - (10 * int(normalized_t * 255)) % 200
    color_g = 255 - (10 * int(normalized_t * 255)) % 255
    color_b = (10 * int(normalized_t * 255)) % 255

    # Add the current point to the list
    points.append((pos_x, pos_y, color_r, color_g, color_b))

    # Add the current point to the list
    points.append((pos_x, pos_y, color_r, color_g, color_b))

    # Draw all the points on the screen
    for point in points:
        pygame.draw.circle(screen, (point[2], point[3], point[4]), (point[0], point[1]), 1)

    # Update the display
    pygame.display.flip()

    # Limit the frame rate
    clock.tick(60)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            quit()