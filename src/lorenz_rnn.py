import numpy as np
import utils as ut
import RNN as rnn

import plot_utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# import pygame as pg
from visualisers.pg_visualiser import py_visualiser


# ## Constants

train_size = 0.8
rng = np.random.default_rng(2048)
n_epochs = 1000
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

model_lstm = rnn.lstm(n_hidden=32, n_layers=1, input_shape=(2, spacial_dim))
model_lstm.build(optimizer='adam', loss='mean_squared_error')
model_lstm.fit(train_inputs, train_targets, n_epochs)
model_lstm.summary()

model_vanilla = rnn.simple_rnn(n_hidden=32, n_layers=1, input_shape=(2, spacial_dim))
model_vanilla.build(optimizer='adam', loss='mean_squared_error')
model_vanilla.fit(train_inputs, train_targets, n_epochs)
model_vanilla.summary()


# model = load_model('multi_1000_32.h5')

pred_seq_lstm = model_lstm.test(sequenced_test_inputs, test_indx)
pred_seq_vanilla = model_vanilla.test(sequenced_test_inputs, test_indx)
# model.save('multi.h5')


# ## Visualisation

# separate x,y,z

pred_seq_lstm = pred_seq_lstm.reshape(-1, spacial_dim)
pred_seq_vanilla = pred_seq_vanilla.reshape(-1, spacial_dim)

# Extract x, y, z coordinates from predicted sequence
pred_x_lstm = pred_seq_lstm[:, 0]
pred_y_lstm = pred_seq_lstm[:, 1]
pred_z_lstm = pred_seq_lstm[:, 2]

pred_x_vanilla = pred_seq_vanilla[:, 0]
pred_y_vanilla = pred_seq_vanilla[:, 1]
pred_z_vanilla = pred_seq_vanilla[:, 2]

# Extract x, y, z coordinates from test data
test_x = sequenced_test_targets[test_indx][:, 0]
test_y = sequenced_test_targets[test_indx][:, 1]
test_z = sequenced_test_targets[test_indx][:, 2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the predicted sequence
ax.plot(pred_x_lstm, pred_y_lstm, pred_z_lstm, label='LSTM Predicted Sequence', linestyle='--')
ax.plot(pred_x_vanilla, pred_y_vanilla, pred_z_vanilla, label='Vanilla Predicted Sequence', linestyle='-.')


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


# pg vis

#py_visualiser(dataset=raw_data, seq_pos=pred_seq)
