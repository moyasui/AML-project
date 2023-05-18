import pygame
import pandas as pd
import numpy as np

def py_visualiser(test_steps, len_seq=2, dataset=None, filename=None, seq_pos=None, indx=8,):
    # TODO: 
# Set up Pygame
    pygame.init()
    clock = pygame.time.Clock()

    has_pred = True
    if seq_pos is None:
        has_pred = False

    # Set up the display
    width, height = 800, 800
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption('Lorenz Attractor Numerical vs RNN Animation')

    # Read CSV file 'src/csvs/lorenz.csv'
    if filename:
        filename = "src/csvs/lorenz.csv"
        dataset = pd.read_csv(filename)
        dataset.columns = dataset.columns.str.replace(' ', '')
        data = dataset[dataset['particle'] == indx]
    elif dataset is not None:
        data = dataset[dataset['particle'] == indx]
    else:
        raise Exception("No values!")


    t_test = data['t']
    # x_test = data['x']
    # y_test = data['y']
    # z_test = data['z']

    

    xy_t = data[['x','y']]
    

    # Set up animation parameters
    t_min = t_test.min()  # Minimum time value
    t_max = t_test.max()  # Maximum time value 

    # Scaling factor for points
    scale_factor = 400

    test_points = []
    scaled_xy_t = xy_t * scale_factor + width // 2
    
    if has_pred:
        xy_pred = seq_pos[:, :2]
        pred_points = []
        scaled_xy_pred = xy_pred * scale_factor + width // 2

    # Animation loop
    for frame in range(test_steps):
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

        test_point = np.array(scaled_xy_t.iloc[frame+len_seq]-np.array([300,300]))
        
        test_points.append(test_point)

        if has_pred:
            pred_point = np.array(scaled_xy_pred[frame]-np.array([300,300]))
            pred_points.append(pred_point)

            
        # Draw all the points on the screen
        for i in range(len(test_points)):
            pygame.draw.circle(screen, colour_test, test_points[i], 2)
            if has_pred:
                pygame.draw.circle(screen, colour_pred, pred_points[i], 2)

        # Update the display
        pygame.display.flip()

        # Limit the frame rate
        clock.tick(60)

        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()


# py_visualiser()