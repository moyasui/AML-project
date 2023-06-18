# visualisation of csvs

# TODO: write a function called: pg_visualise()
# TODO: read csvs as a data frame, depending on the input: cols_of_interest, visualise the columns of interest

import pandas as pd
import numpy as np
import pygame
from sklearn.preprocessing import MinMaxScaler
import pygame.freetype
import sys

def read_csv(filename, particle=None, cols=None):
    # Load data from csv file
    df = pd.read_csv(filename)

    # Remove spaces from column names
    df.rename(columns=lambda x: x.replace(' ', ''), inplace=True)

    # If cols is not specified, use all columns
    if cols is None:
        cols = df.columns

    # Select rows where particle column equals given particle
    if particle is not None:
        df = df[df['particle'] == particle]

    # Select columns of interest
    df = df[cols]

    # Scale and normalize the data
    scaler = MinMaxScaler()
    print(df.shape)
    scaled_data = scaler.fit_transform(df)

    # Convert to numpy arrays
    x, y = scaled_data[:, 0], scaled_data[:, 1]

    return x, y, scaler


def visualiser(filename, particle, cols=None, predicted_x=None, predicted_y=None, steps=100):
    # Call read_csv and get the x, y and scaler
    x, y, scaler = read_csv(filename, particle, cols)

    # Define the predicted points
    predicted = None
    if predicted_x is not None and predicted_y is not None:
        # Rescale the predicted values using the scaler
        predicted = np.array([predicted_x, predicted_y])
        predicted = scaler.transform(predicted.reshape(-1, 2))  # change here

    # Initiate the pygame
    pygame.init()

    # Get the screen size
    infoObject = pygame.display.Info()
    screen = pygame.display.set_mode((infoObject.current_w, infoObject.current_h))

    # Define a button
    button = pygame.Rect(0, 0, 60, 30)

    # Draw circles for the points
    old_points_xy = []
    old_points_predicted = []

    # Calculate the scale factor to keep points within 90% of the screen
    scale_factor = 0.7

    # Loop for the visualisation
    running = True
    for step in range(steps):
        pygame.time.delay(10)  # Add a delay of 0.01 seconds
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if button.collidepoint(event.pos):
                    running = False

        # Make background black
        screen.fill((0, 0, 0))

        radius = 3
        # Draw older points in increasingly lighter colors
        for i, point_xy in enumerate(old_points_xy):
            color = (i / len(old_points_xy) * 255)
            pygame.draw.circle(screen, (0, color, color), point_xy, radius)
            if predicted is not None:
                pygame.draw.circle(screen, (150, color, color), old_points_predicted[i], radius)

        # Draw new points in white and add to old points list
        point_xy = (int((x[step]*scale_factor + (1 - scale_factor) / 2) * infoObject.current_w), int((y[step]*scale_factor + (1 - scale_factor) / 2) * infoObject.current_h))
        old_points_xy.append(point_xy)
        pygame.draw.circle(screen, (0, 255, 255), point_xy, radius)
        if predicted is not None:
            point_predicted = (int((predicted[step, 0]*scale_factor + (1 - scale_factor) / 2) * infoObject.current_w), int((predicted[step, 1]*scale_factor + (1 - scale_factor) / 2) * infoObject.current_h))
            old_points_predicted.append(point_predicted)
            pygame.draw.circle(screen, (255, 255, 255), point_predicted, radius)

        # Draw the button in white
        pygame.draw.rect(screen, [255, 255, 255], button)

        pygame.display.flip()

    # Quit pygame
    pygame.quit()
    sys.exit()


# test
if __name__ == "__main__":

    # No prediction
    filename = "csvs/lorenz.csv"
    indx = 0
    steps = 800
    test = np.zeros(steps)
    cols = ['x', 'y']
    visualiser(filename, indx, cols, test, test, steps)

    