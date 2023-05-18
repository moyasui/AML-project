import numpy as np
import utils as ut
import RNN as rnn
import loss_functions as lf
import plot_utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

import tensorflow as tf
import random
# import pygame as pg
from visualisers.pg_visualiser import py_visualiser

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)

# ## Constants

train_size = 0.8
n_epochs = 100
batch_size = None
len_seq = 2
spacial_dim = 3
n_hidden = 32
test_indx = 1

raw_data, inputs, targets = ut.prep_data('lorenz', len_seq)
inputs[-1]

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


# train_inputs.shape, train_targets.shape
# print(sequences_train_inputs), test_targets.shape


# ## The rnn
#

model_lstm_PI = rnn.lstm(n_hidden=32, n_layers=1, input_shape=(2, spacial_dim))

pi_loss = lf.loss(momentum_conservation=True, momentum_weight=0.005)

model_lstm_PI.build(optimizer='adam', loss=pi_loss.custom_loss)
model_lstm_PI.fit(train_inputs, train_targets, n_epochs)
model_lstm_PI.summary()



model_lstm = rnn.lstm(n_hidden=32, n_layers=1, input_shape=(2, spacial_dim))

model_lstm.build(optimizer='adam', loss="mean_squared_error")
model_lstm.fit(train_inputs, train_targets, n_epochs)
model_lstm.summary()



# model = load_model('multi_1000_32.h5')

pred_seq_lstm = model_lstm.test(sequenced_test_inputs, test_indx)
pred_seq_lstm_PI = model_lstm_PI.test(sequenced_test_inputs, test_indx)

# model.save('multi.h5')


# ## Visualisation

# separate x,y,z

pred_seq_lstm = pred_seq_lstm.reshape(-1, spacial_dim)
pred_seq_lstm_PI = pred_seq_lstm_PI.reshape(-1, spacial_dim)

# Extract x, y, z coordinates from predicted sequence
pred_x_lstm = pred_seq_lstm[:, 0]
pred_y_lstm = pred_seq_lstm[:, 1]
pred_z_lstm = pred_seq_lstm[:, 2]

pred_x_lstm_PI = pred_seq_lstm_PI[:, 0]
pred_y_lstm_PI= pred_seq_lstm_PI[:, 1]
pred_z_lstm_PI = pred_seq_lstm_PI[:, 2]

# print average errors for each model, relative to the test data
print("LSTM average error: ", np.mean(np.abs(pred_seq_lstm - sequenced_test_targets[test_indx])))
print("LSTM PI average error: ", np.mean(np.abs(pred_seq_lstm_PI - sequenced_test_targets[test_indx])))

# Extract x, y, z coordinates from test data
test_x = sequenced_test_targets[test_indx][:, 0]
test_y = sequenced_test_targets[test_indx][:, 1]
test_z = sequenced_test_targets[test_indx][:, 2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the predicted sequence
ax.plot(pred_x_lstm, pred_y_lstm, pred_z_lstm, label='LSTM Predicted Sequence',  linestyle='-.')
ax.plot(pred_x_lstm_PI, pred_y_lstm_PI, pred_z_lstm_PI, label='LSTM PI Predicted Sequence',  linestyle='--')


# Plot the test data
ax.plot(test_x, test_y, test_z, label='Test Data')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Predicted Sequence vs Test Data')

# Add a legend
ax.legend()

# save plot
plt.savefig('./Analysis/figs/lorenz_rnn.pdf')

# Show the plot
plt.show()

# make a graphs that plots the coordinates prediction versus actual per epoch in the same plot
#get colors from the pallet pastel in order
colors = sns.color_palette("pastel", 6)

errors_x = np.abs(pred_x_lstm - test_x)
errors_y = np.abs(pred_y_lstm - test_y)
errors_z = np.abs(pred_z_lstm - test_z)
total = errors_x + errors_y + errors_z

errors_x_PI = np.abs(pred_x_lstm_PI - test_x)
errors_y_PI = np.abs(pred_y_lstm_PI - test_y)
errors_z_PI = np.abs(pred_z_lstm_PI - test_z)
total_PI = errors_x_PI + errors_y_PI + errors_z_PI

# plot the total errors
plt.plot(total, label='LSTM', color=colors[0])
plt.plot(total_PI, label='LSTM PI', color=colors[1])
plt.xlabel('t')
plt.ylabel('$\sum_i |x_i^{(t)} - \hat{x_i}^{(t)}|$')
plt.legend()

plt.savefig('./Analysis/figs/lorenz_rnn_PI_coordinates_errors.pdf')
plt.show()




# pg vis

#py_visualiser(dataset=raw_data, seq_pos=pred_seq)
