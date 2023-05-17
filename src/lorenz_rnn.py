import numpy as np
import utils as ut
import RNN_copy as rnn
from tensorflow import keras
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.losses import MeanSquaredError
import plot_utils

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns

# import pygame as pg
from visualisers.pg_visualiser import py_visualiser


train_size = 0.8
rng = np.random.default_rng(2048)
n_epochs = 1000
batch_size = None
len_seq = 2
spacial_dim = 3
n_hidden = 32
test_indx = 1
test_steps = 10

def rnn_alt(my_model, 
            optimizer, 
            train_inputs, train_targets, 
            n_epochs, 
            spacial_dim, 
            ic, 
            is_saving_model=False, 
            save_name=None):
    
    my_model.build(optimizer=optimizer, loss='mean_squared_error')
    my_model.fit(train_inputs, train_targets, n_epochs)
    my_model.summary()
    if is_saving_model:
        if save_name is None:
            my_model.my_save(f"trained_models/{my_model.name}.h5")
        else:
            my_model.my_save(save_name)
    
    
    pred_seq = my_model.predict(ic,n_steps=test_steps)
    pred_seq = pred_seq.reshape(-1, spacial_dim)

    return pred_seq

def load_model(model_name):
    model_lstm = rnn.Lstm()
    model_lstm.my_load(model_name)

    return model_lstm

def lorenz_pred(optimizer, len_seq):
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

    pred_lstm = rnn_alt(model_lstm, optimizer, train_inputs, train_targets, n_epochs, spacial_dim, ic, False)
    pred_vanilla = rnn_alt(model_vinilla, optimizer, train_inputs, train_targets,n_epochs, spacial_dim, ic)

pred_x_vanilla = pred_seq_vanilla[:, 0]
pred_y_vanilla = pred_seq_vanilla[:, 1]
pred_z_vanilla = pred_seq_vanilla[:, 2]

def eval(sequenced_test_targets, pred_lstm, pred_vanilla):

    mse = MeanSquaredError()
    test = sequenced_test_targets[test_indx][:test_steps]
    err_lstm = mse(test, pred_lstm)
    err_vanilla = mse(test, pred_vanilla)
    return err_lstm, err_vanilla

def eval_n_plot(sequenced_test_targets, pred_lstm, pred_vanilla, len_seq, n_epochs, learning_rate):
    # Extract x, y, z coordinates from test data
    test_x = sequenced_test_targets[test_indx][:test_steps, 0]
    test_y = sequenced_test_targets[test_indx][:test_steps, 1]
    test_z = sequenced_test_targets[test_indx][:test_steps, 2]

# Create a 3D plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Plot the predicted sequence
ax.plot(pred_x_lstm, pred_y_lstm, pred_z_lstm, label='LSTM Predicted Sequence')
ax.plot(pred_x_vanilla, pred_y_vanilla, pred_z_vanilla, label='Vanilla Predicted Sequence')


# Plot the test data
ax.plot(test_x, test_y, test_z, label='Test Data', linestyle='-.')

# Set labels and title
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title('Predicted Sequence vs Test Data')

# Add a legend
ax.legend()

    # save plot
    file_info = f"./Analysis/figs/lorenz_rnn-{len_seq}-{n_epochs}-{learning_rate:.2f}.pdf"
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

errs_lstm = np.zeros((4,4))
errs_vanilla = np.zeros((4,4))

all_n_epochs = (20,150,500,1000)
len_seqs = range(2,6)
for i, n_epochs in enumerate(all_n_epochs):
    for j, len_seq in enumerate(len_seqs):
        # for k, lr in enumerate(np.logspace(10e-4,10e-1,4)):
        optimizer = Adam(learning_rate=10e-2)
        sequenced_test_targets, pred_lstm, pred_vanilla = lorenz_pred(optimizer, len_seq)
        errs_lstm[i,j], errs_vanilla[i,j] = eval(sequenced_test_targets, pred_lstm, pred_vanilla)
            
            
print(errs_lstm, errs_vanilla)


# Set up the figure and axes
fig, (ax1, ax2) = plt.subplots(1, 2)
sns.set(font_scale=1.4)  # Adjust font size

# Plot the confusion matrix

sns.heatmap(errs_lstm, annot=True, cmap='Blues', cbar=True, square=True,
            xticklabels=len_seqs, yticklabels=all_n_epochs, ax=ax1)

sns.heatmap(errs_vanilla, annot=True, cmap='Blues', cbar=True, square=True,
            xticklabels=len_seqs, yticklabels=all_n_epochs ,ax=ax2)
# Add labels, title, and axis ticks

ax1.set_xlabel('n_epochs')
ax2.set_xlabel('n_epochs')
ax1.set_ylabel('len_seq')
ax2.set_ylabel('len_seq')
plt.title('Confusion Matrix')
plt.show()
