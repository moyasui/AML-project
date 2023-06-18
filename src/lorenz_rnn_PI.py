import numpy as np
import utils as ut
import RNN as rnn
from tensorflow import keras
import loss_functions as lf

from tensorflow.keras import optimizers
from tensorflow.keras.losses import MeanSquaredError

import plot_utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# import pygame as pg
from visualisers.pg_visualiser import visualiser

train_size = 0.8
n_epochs = 10
batch_size = None
test_steps = None

spacial_dim = 3
n_hidden = 32
test_indx = 1


rng = np.random.default_rng(2044)


def rnn_alt(
    my_model,
    optimizer,
    train_inputs,
    train_targets,
    n_epochs,
    spacial_dim,
    ic,
    is_saving_model=False,
    save_name=None,
    is_PI=False
):

    if is_PI:
        pi_loss = lf.loss(momentum_conservation=True, momentum_weight=0.5)
        my_model.build(optimizer=optimizer, loss=pi_loss.custom_loss)
        history = my_model.fit(train_inputs, train_targets, n_epochs)
        my_model.summary()

    else:
        my_model.build(optimizer=optimizer, loss='mean_squared_error')
        history = my_model.fit(train_inputs, train_targets, n_epochs)
        my_model.summary()

    if is_saving_model:
        if save_name is None:
            my_model.my_save(f'trained_models/{my_model.name}.h5')
        else:
            my_model.my_save(save_name)

    pred_seq = my_model.predict(ic, n_steps=test_steps)
    pred_seq = pred_seq.reshape(-1, spacial_dim)

    return pred_seq, history


def load_model(model_name):
    model_lstm = rnn.Lstm()
    model_lstm.my_load(model_name)

    return model_lstm


def lorenz_PI_pred(optimizer, len_seq):

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

    model_lstm_PI = rnn.Lstm(
        n_hidden=32, n_layers=1, input_shape=(len_seq, spacial_dim)
    )
    model_lstm = rnn.Lstm(
        n_hidden=32, n_layers=1, input_shape=(len_seq, spacial_dim)
    )
    
    # predicts
    ic = sequenced_test_inputs[test_indx]

    pred_lstm, history_lstm = rnn_alt(
        model_lstm,
        optimizer,
        train_inputs,
        train_targets,
        n_epochs,
        spacial_dim,
        ic,
        False,
    )

    pred_lstm_PI, history_lstm_PI = rnn_alt(
        model_lstm_PI,
        optimizer,
        train_inputs,
        train_targets,
        n_epochs,
        spacial_dim,
        ic,
        False,
        True
    )

    return sequenced_test_targets, pred_lstm, pred_lstm_PI, history_lstm, history_lstm_PI


def eval(sequenced_test_targets, pred_lstm, pred_PI):

    mse = MeanSquaredError()
    test = sequenced_test_targets[test_indx][:test_steps]
    err_lstm = mse(test, pred_lstm)
    err_PI = mse(test, pred_PI)
    return err_lstm, err_PI


def plot_results(
    sequenced_test_targets,
    pred_lstm,
    pred_vanilla,
    len_seq,
    learning_rate,
):

    # Extract x, y, z coordinates from test data
    test_x = sequenced_test_targets[test_indx][:test_steps, 0]
    test_y = sequenced_test_targets[test_indx][:test_steps, 1]
    test_z = sequenced_test_targets[test_indx][:test_steps, 2]

    # Create a 3D plot
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Plot the predicted sequence
    ax.plot(
        pred_lstm[:, 0],
        pred_lstm[:, 1],
        pred_lstm[:, 2],
        label='LSTM Predicted Sequence',
        linestyle=':'
    )
    ax.plot(
        pred_vanilla[:, 0],
        pred_vanilla[:, 1],
        pred_vanilla[:, 2],
        label='LSTM PI Predicted Sequence',
        linestyle='-.'
    )
    ax.plot(test_x, test_y, test_z, label='Test Data')

    # Set labels and title
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    #ax.set_title(f'Predicted Sequence vs Test Data (look back = {len_seq})')

    # Add a legend
    ax.legend()

    # save plot

    file_info = f'./Analysis/figs/lorenz_LSTMPI_rnn-{len_seq}-{n_epochs}-{learning_rate:.2f}.pdf'

    plt.savefig(file_info)

    # Show the plot
    plt.show()
    # Don't show the plot
    plt.close()


def plot_loss(history_lstm, history_lstm_PI):
    loss_lstm = history_lstm.history['loss']
    val_loss_lstm = history_lstm.history['val_loss']

    loss_lstm_PI = history_lstm_PI.history['loss']
    val_loss_lstm_PI = history_lstm_PI.history['val_loss']

    # make moving avg of all quantities form numpy array
    #loss_lstm = ut.moving_average(loss_lstm, 5)[4:]
    #val_loss_lstm = ut.moving_average(val_loss_lstm, 5)[4:]
    #
    #loss_lstm_PI = ut.moving_average(loss_lstm_PI, 5)[4:]
    #val_loss_lstm_PI = ut.moving_average(val_loss_lstm_PI, 5)[4:]    

    plt.plot(loss_lstm, label='Loss LSTM')
    plt.plot(loss_lstm_PI, label='Loss LSTM PI')
    #plt.plot(val_loss_lstm, label='Val loss LSTM')
    #plt.plot(val_loss_lstm_PI, label='Val loss LSTM PI')

    #plt.ylim([0, 10])
    plt.xlabel('Epoch')
    plt.ylabel('log History MSE')

    #plt.yscale("log")
    plt.legend()
    plt.grid(True)
    file_info = f'./Analysis/figs/loghist_lorenz_LSTMPI_rnn.pdf'
    plt.savefig(file_info)
    plt.show()
    plt.close()




optimizer = (
    optimizers.legacy.Adam()
)   # tf warning says it slows down on m1 and m2
sequenced_test_targets, pred_lstm, pred_lstm_PI, history_lstm, history_lstm_PI = lorenz_PI_pred(
    optimizer, 2
)   # this is correct in comparision to before
plot_results(sequenced_test_targets, pred_lstm, pred_lstm_PI, 2, 0.001)

plot_loss(history_lstm, history_lstm_PI)


print("ERRORS LSTM LSTMPI", eval(sequenced_test_targets, pred_lstm, pred_lstm_PI))



####
# Extract x, y, z coordinates from predicted sequence
pred_x_lstm = pred_lstm[:, 0]
pred_y_lstm = pred_lstm[:, 1]
pred_z_lstm = pred_lstm[:, 2]

pred_x_lstm_PI = pred_lstm_PI[:, 0]
pred_y_lstm_PI= pred_lstm_PI[:, 1]
pred_z_lstm_PI = pred_lstm_PI[:, 2]


# Extract x, y, z coordinates from test data
test_x = sequenced_test_targets[test_indx][:, 0]
test_y = sequenced_test_targets[test_indx][:, 1]
test_z = sequenced_test_targets[test_indx][:, 2]


# make a graphs that plots the coordinates prediction versus actual per epoch in the same plot
#get colors from the pallet pastel in order
colors = sns.color_palette("pastel", 6)

errors_x_LSTM = np.abs(pred_x_lstm - test_x)
errors_y_LSTM = np.abs(pred_y_lstm - test_y)
errors_z_LSTM = np.abs(pred_z_lstm - test_z)
total_lstm = errors_x_LSTM + errors_y_LSTM + errors_z_LSTM

errors_x_LSTM_PI = np.abs(pred_x_lstm_PI - test_x)
errors_y_LSTM_PI = np.abs(pred_y_lstm_PI - test_y)
errors_z_LSTM_PI = np.abs(pred_z_lstm_PI - test_z)
total_lstm_PI = errors_x_LSTM_PI + errors_y_LSTM_PI + errors_z_LSTM_PI

# plot the total errors
plt.plot(total_lstm, label='LSTM', color=colors[0])
plt.plot(total_lstm_PI, label='LSTM PI', color=colors[1])
plt.xlabel('t')
plt.ylabel('$\sum_i |x_i^{(t)} - \hat{x_i}^{(t)}|$')
plt.legend()

plt.savefig('./Analysis/figs/lorenz_rnn_lstm_lstmPI_coordinates_errors.pdf')
plt.show()