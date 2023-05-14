import pygame
import pandas as pd
import numpy as np

def py_visualiser(filename=None, seq_pos=None, indx=8,):
# Set up Pygame
    pygame.init()
    clock = pygame.time.Clock()
    if seq_pos is None:
        has_pred = False

    # Set up the display
    width, height = 800, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('3D Function Animation')

    # Read CSV file 'src/csvs/lorenz.csv'
    if filename is None:
        filename = "src/csvs/lorenz.csv"
    dataset = pd.read_csv(filename)
    dataset.columns = dataset.columns.str.replace(' ', '')
    data = dataset[dataset['particle'] == indx]
    
    t_test = data['t']
    # x_test = data['x']
    # y_test = data['y']
    # z_test = data['z']

    

    xy_t = data[['x','y']]
    

    # Set up animation parameters
    t_min = t_test.min()  # Minimum time value
    t_max = t_test.max()  # Maximum time value 

    # Scaling factor for points
    scale_factor = 10

    test_points = np.array([])
    scaled_xy_t = xy_t * scale_factor + width // 2
    
    if has_pred:
        xy_pred = seq_pos[:, :1]
        pred_points = np.array([])
        scaled_xy_pred = xy_pred * scale_factor + width // 2

    # Animation loop
    for frame in range(len(data)):
        # Clear the screen
        screen.fill((0, 0, 0))

        # Get the x, y, z, and t values for the current frame
        t = t_test.iloc[frame]

        # Normalize the time value
        normalized_t = (t - t_min) / (t_max - t_min)

        colour_tests = []
        colour_preds = []
        # Compute the color based on the normalized time value
        colour_test = 160, 100, int(normalized_t * 255) % 255
        colour_pred = 0, 100, int(normalized_t * 255) % 255 

        # Add the current point to the list
        # test_points.append((pos_x, pos_y, color_test))

        test_point = np.array(scaled_xy_t.iloc[frame])
        print(test_point)
        test_points = np.append(test_points, test_point)

        if has_pred:
            pred_point = np.array([scaled_xy_pred[frame], colour_pred]) 
            pred_points = np.append(pred_points, pred_point)

            
        
        # Draw all the points on the screen
        for point in test_points:
            # print(point[1])
            pygame.draw.circle(screen, colour_pred, (point[0],point[1]), 1)
            # pygame.draw.circle(screen, colour_pred, pred_points[0], 1)

        # Update the display
        pygame.display.flip()

        # Limit the frame rate
        clock.tick(60)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

py_visualiser()