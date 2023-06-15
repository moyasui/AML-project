import numpy as np
import utils as ut
import RNN as rnn
from tensorflow import keras

from tensorflow.keras import optimizers
from tensorflow.keras.losses import MeanSquaredError

import plot_utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# import pygame as pg
from visualisers.pg_visualiser import pg_visualiser

train_size = 0.8
n_epochs = 500
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
):

    my_model.build(optimizer=optimizer, loss='mean_squared_error')
    my_model.fit(train_inputs, train_targets, n_epochs)
    my_model.summary()

    if is_saving_model:
        if save_name is None:
            my_model.my_save(f'trained_models/{my_model.name}.h5')
        else:
            my_model.my_save(save_name)

    pred_seq = my_model.predict(ic, n_steps=test_steps)
    pred_seq = pred_seq.reshape(-1, spacial_dim)

    return pred_seq


def load_model(model_name):
    model_lstm = rnn.Lstm()
    model_lstm.my_load(model_name)

    return model_lstm


def lorenz_pred(optimizer, len_seq):

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

    model_vanilla = rnn.Simple_rnn(
        n_hidden=32, n_layers=1, input_shape=(len_seq, spacial_dim)
    )
    model_lstm = rnn.Lstm(
        n_hidden=32, n_layers=1, input_shape=(len_seq, spacial_dim)
    )
    
    # predicts
    ic = sequenced_test_inputs[test_indx]

    pred_vanilla = rnn_alt(
        model_vanilla,
        optimizer,
        train_inputs,
        train_targets,
        n_epochs,
        spacial_dim,
        ic,
        False
    )

    pred_lstm = rnn_alt(
        model_lstm,
        optimizer,
        train_inputs,
        train_targets,
        n_epochs,
        spacial_dim,
        ic,
        False,
    )


    return sequenced_test_targets, pred_lstm, pred_vanilla


def eval(sequenced_test_targets, pred_lstm, pred_vanilla):

    mse = MeanSquaredError()
    test = sequenced_test_targets[test_indx][:test_steps]
    err_lstm = mse(test, pred_lstm)
    err_vanilla = mse(test, pred_vanilla)
    return err_lstm, err_vanilla


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
        label='Vanilla Predicted Sequence',
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

    file_info = f'./Analysis/figs/lorenz_rnn-{len_seq}-{n_epochs}-{learning_rate:.2f}.pdf'

    plt.savefig(file_info)

    # Show the plot
    # plt.show()
    # Don't show the plot
    plt.close()


# pg vis

# py_visualiser(test_steps, len_seq, dataset=raw_data, seq_pos=pred_lstm, indx=8+test_indx)

# Testing hyperparameters

# for len_seq in (2,):
#     sequenced_test_targets, pred_lstm, pred_vanilla = lorenz_pred("adam", len_seq)
#     plot_sim_lstm(sequenced_test_targets, pred_lstm, pred_vanilla)


# all_n_epochs = (20, 150, 500)
# len_seqs = range(2, 7)
# dim_heatmap = (len(all_n_epochs), len(len_seqs))

# errs_lstm = np.zeros(dim_heatmap)
# errs_vanilla = np.zeros(dim_heatmap)

# for i, n_epochs in enumerate(all_n_epochs):
#     for j, len_seq in enumerate(len_seqs):
#         # for k, lr in enumerate(np.logspace(10e-4,10e-1,4)):
#         print(f'------------{n_epochs} epochs------{len_seq} steps---------')
#         optimizer = (
#             optimizers.legacy.Adam()
#         )   # tf warning says it slows down on m1 and m2
#         sequenced_test_targets, pred_lstm, pred_vanilla = lorenz_pred(
#             optimizer, len_seq
#         )
#         errs_lstm[i, j], errs_vanilla[i, j] = eval(
#             sequenced_test_targets, pred_lstm, pred_vanilla
#         )


# print(errs_lstm, errs_vanilla)

# # make two heatmaps. One for lstm and one for vanilla but display on same image
# sns.heatmap(
#     errs_lstm,
#     annot=True,
#     fmt='.4g',
#     cmap='Blues',
#     cbar_kws={'label': 'Mean Squared Error'},
#     xticklabels=len_seqs,
#     yticklabels=all_n_epochs,
# )
# # axis labels
# plt.xlabel('Sequence Length')
# plt.ylabel('Number of Epochs')

# # save plot
# plt.savefig('./Analysis/figs/smallgridsearch_lenght_epochs_lstm.pdf')
# plt.show()

# sns.heatmap(
#     errs_vanilla,
#     annot=True,
#     fmt='.4g',
#     cmap='Blues',
#     cbar_kws={'label': 'Mean Squared Error'},
#     xticklabels=len_seqs,
#     yticklabels=all_n_epochs,
# )
# plt.xlabel('Sequence Length')
# plt.ylabel('Number of Epochs')
# plt.savefig('./Analysis/figs/smallgridsearch_lenght_epochs_vanilla.pdf')
# plt.show()


# # make two axis to plot the two sns heatmaps in one
# fig, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
# sns.heatmap(
#     errs_lstm,
#     annot=True,
#     fmt='.4g',
#     cmap='Blues',
#     cbar_kws={'label': 'Mean Squared Error'},
#     xticklabels=len_seqs,
#     yticklabels=all_n_epochs,
#     ax=ax1,
# )
# sns.heatmap(
#     errs_vanilla,
#     annot=True,
#     fmt='.4g',
#     cmap='Blues',
#     cbar_kws={'label': 'Mean Squared Error'},
#     xticklabels=len_seqs,
#     yticklabels=all_n_epochs,
#     ax=ax2,
# )

# # axis labels
# ax1.set_xlabel('Sequence Length')
# ax1.set_ylabel('Number of Epochs')
# ax2.set_xlabel('Sequence Length')
# ax2.set_ylabel('Number of Epochs')
# plt.savefig('./Analysis/figs/smallgridsearch_lenght_epochs_vanillalstm.pdf')

# plt.show()


optimizer = (
    optimizers.legacy.Adam()
)   # tf warning says it slows down on m1 and m2
sequenced_test_targets, pred_lstm, pred_vanilla = lorenz_pred(
    optimizer, 2
)   # this is correct in comparision to before
plot_results(sequenced_test_targets, pred_lstm, pred_vanilla, 2, 0.001)


print("ERRORS LSTM VANILLA", eval(sequenced_test_targets, pred_lstm, pred_vanilla))



####
# Extract x, y, z coordinates from predicted sequence
pred_x_lstm = pred_lstm[:, 0]
pred_y_lstm = pred_lstm[:, 1]
pred_z_lstm = pred_lstm[:, 2]

pred_x_vanilla = pred_vanilla[:, 0]
pred_y_vanilla= pred_vanilla[:, 1]
pred_z_vanilla = pred_vanilla[:, 2]

# print average errors for each model, relative to the test data
print("LSTM average error: ", np.mean(np.abs(pred_lstm - sequenced_test_targets[test_indx])))
print("pred_vanilla average error: ", np.mean(np.abs(pred_vanilla - sequenced_test_targets[test_indx])))

# Extract x, y, z coordinates from test data
test_x = sequenced_test_targets[test_indx][:, 0]
test_y = sequenced_test_targets[test_indx][:, 1]
test_z = sequenced_test_targets[test_indx][:, 2]


# make a graphs that plots the coordinates prediction versus actual per epoch in the same plot
#get colors from the pallet pastel in order
colors = sns.color_palette("pastel", 6)

errors_x_vanilla = np.abs(pred_x_vanilla - test_x)
errors_y_vanilla = np.abs(pred_y_vanilla - test_y)
errors_z_vanilla = np.abs(pred_z_vanilla - test_z)
total_vanilla = errors_x_vanilla + errors_y_vanilla + errors_z_vanilla

errors_x_LSTM = np.abs(pred_x_lstm - test_x)
errors_y_LSTM = np.abs(pred_y_lstm - test_y)
errors_z_LSTM = np.abs(pred_z_lstm - test_z)
total_lstm = errors_x_LSTM + errors_y_LSTM + errors_z_LSTM

# plot the total errors
plt.plot(total_vanilla, label='vanilla', color=colors[0])
plt.plot(total_lstm, label='LSTM', color=colors[1])
plt.xlabel('t')
plt.ylabel('$\sum_i |x_i^{(t)} - \hat{x_i}^{(t)}|$')
plt.legend()

plt.savefig('./Analysis/figs/lorenz_rnn_lstm_vanilla_coordinates_errors.pdf')
plt.show()