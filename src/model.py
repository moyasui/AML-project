import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input

from tensorflow.keras.layers import Dense, LSTM, Dropout, GRU, Bidirectional, SimpleRNN
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.random import set_seed


import tensorflow as tf


rng = np.random.default_rng(1205)

data_folder = "src/csvs/"

# ------------ DEBUGING FUNC -----------------
IS_DEBUG = True
def debug_print(stuff):
    if IS_DEBUG:
        print(stuff)

# ------------ DEBUGING FUNC END -----------------



def visualise_data(data):
    plt.figure(figsize=(20, 10))

    plt.plot(train['x'], label='train_x', color='red')
    plt.plot(test['x'], label='test_x', linestyle='--', color='red')
    plt.plot(train['y'], label='train_y', color='green')
    plt.plot(test['y'], label='test_y', linestyle='--', color='green')
    plt.plot(train['z'], label='train_z', color='blue')
    plt.plot(test['z'], label='test_z', linestyle='--', color='blue')

    plt.legend()
    plt.show()

# visualise_data(sequences)

# format data
# TODO: ?????
def format_data(data, length_of_sequence = 2):  
    """
        Inputs:
            data(a numpy array): the data that will be the inputs to the recurrent neural
                network
            length_of_sequence (an int): the number of elements in one iteration of the
                sequence patter.  For a function approximator use length_of_sequence = 2 because 
                the input is the current value and the previous value.
        Returns:
            rnn_input (a 3D numpy array): the input data for the recurrent neural network.  Its
                dimensions are (length of data - length of sequence, length of sequence, 
                dimension of data)
            rnn_output (a numpy array): the training data for the neural network
        Formats data to be used in a recurrent neural network.
    """

    X, Y = [], []

    # in the case of many sequences
    data.reshape(-1,)
    for i in range(len(data)-length_of_sequence):
        # Get the next length_of_sequence elements
        a = data[i:i+length_of_sequence]
        # Get the element that immediately follows that
        b = data[i+length_of_sequence]

        X.append(a)
        Y.append(b)
    rnn_input = np.array(X)
    rnn_output = np.array(Y)

    debug_print([rnn_input.shape, rnn_output.shape])
    return rnn_input, rnn_output


# the model
def build_rnn(input_shape, n_hidden):
    model = keras.models.Sequential([
        keras.layers.LSTM(n_hidden, input_shape=input_shape), # TODO: too many nodes
        keras.layers.Dense(3)
    ])
    # TODO: graphical representaiton
    # model.summary()
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model


# traning and the parameters
def train_rnn(model, rnn_input, rnn_train , batch_size, epochs, n_hidden): 

    
    debug_print(rnn_input.shape)
    hist = model.fit(rnn_input, rnn_train, epochs=epochs, batch_size=batch_size)
    print(hist)
    return hist, model



# testing and visualisation
# TODO: can't test like that, separate visualisation (using pg)
def test_rnn (y, model):

    """
        Inputs:
            x1 (a list or numpy array): The complete x component of the data set
            y (a list or numpy array): The complete y component of the data set
            plot_min (an int or float): the smallest x value used in the training data
            plot_max (an int or float): the largest x valye used in the training data
        Returns:
            None.
        Uses a trained recurrent neural network model to predict future points in the 
        series.  Computes the MSE of the predicted data set from the true data set, saves
        the predicted data set to a csv file, and plots the predicted and true data sets w
        while also displaying the data range used for training.
    """
    # Add the training data as the first dim points in the predicted data array as these
    # are known values.
    train = int(0.8 * len(y))
    y_pred = y[:train].tolist()
    # Generate the first input to the trained recurrent neural network using the last two 
    # points of the training data.  Based on how the network was trained this means that it
    # will predict the first point in the data set after the training data.  All of the 
    # brackets are necessary for Tensorflow.
    next_input = np.array([[y[train-2], y[train-1]]])
    print(next_input)
    # Save the very last point in the training data set.  This will be used later.
    last = [y[train-1]]

    # Iterate until the complete data set is created.
    for i in range (train, len(y)):
        # Predict the next point in the data set using the previous two points.

        next = model.predict(next_input, verbose=0)

        # convert to list 
        next = next.tolist()
        # Append just the number of the predicted data set
        y_pred.append(next[0])
        #print("y_pred[0]", y_pred[0])
        # Create the input that will be used to predict the next data point in the data set.
        next_input = [last, next]
        next_input = np.array(next_input, dtype=object)
        next_input = np.reshape(next_input, (1, 2, 3))
        next_input = np.asarray(next_input).astype('float32')
        last = next


    # Print the mean squared error between the known data set and the predicted data set.
    mse = np.square(np.subtract(y[train:], y_pred[train:])).mean()
    print('pct MSE: ', mse/np.mean(y))
    # Save the predicted data set as a csv file for later use
    name = 'src/csvs/Predicted'+str(train)+'.csv'
    np.savetxt(name, y_pred, delimiter=',')
    # Plot the known data set and the predicted data set.  The red box represents the region that was used
    # for the training data.

    # break y_pred into actual x, y, z coordinates
    x_plot = y[:,0]
    y_plot = y[:,1]
    z_plot = y[:,2]

    # break y_pred into actual x, y, z coordinates
    x_pred_plot = [i[0] for i in y_pred[train:]]
    y_pred_plot = [i[1] for i in y_pred[train:]]
    z_pred_plot = [i[2] for i in y_pred[train:]]


    fig = plt.figure(figsize=(18, 10))
    ax = fig.add_subplot(projection='3d')

    # lineplot instead of scatterplot
    ax.plot(x_plot, y_plot, z_plot, c='b', label='Actual')
    ax.plot(x_pred_plot, y_pred_plot, z_pred_plot, c='r', label='Predicted', linestyle='dashed')
    ax.plot(x_plot[train:], y_plot[train:], z_plot[train:], c='g', label='Testing Data')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title('Actual vs. Predicted')
    plt.show()
    
    return y_pred, y



# TODO: not done

# def new_test(model, sequences, test_size=2):

#     pos_preds = np.zeros(test_size, dtype=object)

#     for i in range(test_size):
#         sequence = sequences[-i]
#         sequence_preds = []

#         input = sequence.iloc[:2].drop(['t'])

#         for j in range(len(sequence)):

#             if previous_output is not None:
#                 single_step_input = previous_output  # Use previous output as input
            
#             single_step_input = np.reshape(single_step_input, (1,2,3))
#             print(single_step_input)
#             prediction = model.predict(single_step_input)
#             sequence_preds.append(prediction)
#             previous_output = prediction

#         pos_preds[i] = sequence_preds



