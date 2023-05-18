import numpy as np
import utils as ut
import RNN as rnn
from tensorflow import keras
import plot_utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# import pygame as pg
from visualisers.pg_visualiser import py_visualiser

train_size = 0.8
rng = np.random.default_rng(2048)
n_epochs = 1
batch_size = None
spacial_dim = 3
n_hidden = 32
test_indx = 1
test_steps = 10

def rnn_alt(my_model, train_inputs, train_targets, n_epochs, spacial_dim, ic, len_seq, is_saving_model=False, save_name=None):
    
    my_model.build(optimizer='adam', loss='mean_squared_error')
    my_model.fit(train_inputs, train_targets, n_epochs)
    my_model.summary()
    if is_saving_model:
        if save_name is None:
            my_model.my_save(f"trained_models/{my_model.name}.h5")
        else:
            my_model.my_save(save_name)
    
    
    pred_seq = my_model.test(ic,n_steps=test_steps)
    pred_seq = pred_seq.reshape(-1, spacial_dim)

    return pred_seq

def load_model(model_name):
    model_lstm = rnn.Lstm()
    model_lstm.my_load(model_name)

    return model_lstm

def lorenz_pred(len_seq):
    # ## Constants

    # len_seq = 2
    raw_data, inputs, targets = ut.prep_data('lorenz', len_seq)

    train_test, sequenced_train_test = ut.train_test_split(
        inputs, targets, train_size, len_seq, spacial_dim
    )

    train_inputs, test_inputs, train_targets, test_targets = train_test
    (
        sequenced_train_inputs,
        sequenced_test_inputs,
        sequenced_train_targets,
        sequenced_test_targets,
    ) = sequenced_train_test

    model_vinilla = rnn.Simple_rnn(n_hidden=32, n_layers=1, input_shape=(len_seq, spacial_dim))
    model_lstm = rnn.Lstm(n_hidden=32, n_layers=1, input_shape=(len_seq, spacial_dim)) 
    
    # predicts
    ic = sequenced_test_inputs[test_indx]

    pred_lstm = rnn_alt(model_lstm, train_inputs, train_targets, n_epochs, 
                        spacial_dim, ic, len_seq, is_saving_model=True)
    pred_vanilla = rnn_alt(model_vinilla, train_inputs, train_targets,
                            n_epochs, spacial_dim, ic, len_seq)

    return sequenced_test_targets, pred_lstm, pred_vanilla

def plot_look_back(sequenced_test_targets, pred_lstm, pred_vanilla):
    # Extract x, y, z coordinates from test data
    test_x = sequenced_test_targets[test_indx][:test_steps, 0]
    test_y = sequenced_test_targets[test_indx][:test_steps, 1]
    test_z = sequenced_test_targets[test_indx][:test_steps, 2]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the test and predicted sequence
    
    ax.plot(pred_lstm[:,0], pred_lstm[:, 1], pred_lstm[:, 2], label='LSTM Predicted Sequence')
    ax.plot(pred_vanilla[:, 0], pred_vanilla[:, 1], pred_vanilla[:, 2], label='Vanilla Predicted Sequence')
    ax.plot(test_x, test_y, test_z, label='Test Data', linestyle=":")

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f'Predicted Sequence vs Test Data (look back = {len_seq})')

    # Add a legend
    ax.legend()

    # save plot
    plt.savefig('./Analysis/figs/lorenz_rnn' + str(len_seq) + '.pdf')

    # Show the plot
    # plt.show()
    plt.close()

    # pg vis

    # py_visualiser(test_steps, len_seq, dataset=raw_data, seq_pos=pred_lstm, indx=8+test_indx)

# Testing for parameters

for len_seq in (3,):
    sequenced_test_targets, pred_lstm, pred_vanilla = lorenz_pred(len_seq)
    plot_look_back(sequenced_test_targets, pred_lstm, pred_vanilla)