### here we will test the loss functions from the loss_functions.py file

# import path to the loss_functions.py file
import sys
sys.path.append('../')

import numpy as np
import pandas as pd

import loss_functions as lf

# define the true and predicted values from one of the lorenz attractors in the dataset
dataset = pd.read_csv('../lorenz.csv', header=0)
dataset.columns = dataset.columns.str.replace(' ', '')
dataset0 = dataset[dataset['particle'] == 0]

pos_true = np.array(dataset0[['x', 'y', 'z']])
pos_pred = np.array(dataset0[['x', 'y', 'z']])

# create the loss object
print("> TEST WITHOUT NOISE ")

loss = lf.loss(pos_true, pos_pred)

# test the mse loss
mse_loss = loss.mse_loss()
print(">> MSE loss: ", mse_loss.numpy())

# test the momentum conservation loss
momentum_loss = loss._momentum_conservation_loss()
print(">> Momentum conservation loss: ", momentum_loss.numpy())

# test the energy conservation loss
energy_loss = loss._energy_conservation_loss()
print(">> Energy conservation loss: ", energy_loss.numpy())

# test the custom loss
custom_loss = loss.custom_loss(momentum_conservation=True, momentum_weight=0.5, energy_conservation=True, energy_weight=0.5)
print(">> Custom loss: ", custom_loss.numpy())


print("> TEST WITH NOISE")

# add noise to the predicted values
pos_pred = pos_pred + np.random.normal(0, 0.1, pos_pred.shape)


loss = lf.loss(pos_true, pos_pred)

# test the mse loss
mse_loss = loss.mse_loss()
print(">> MSE loss: ", mse_loss.numpy())

# test the momentum conservation loss
momentum_loss = loss._momentum_conservation_loss()
print(">> Momentum conservation loss: ", momentum_loss.numpy())

# test the energy conservation loss
energy_loss = loss._energy_conservation_loss()
print(">> Energy conservation loss: ", energy_loss.numpy())

# test the custom loss
custom_loss = loss.custom_loss(momentum_conservation=True, momentum_weight=0.5, energy_conservation=True, energy_weight=0.5)
print(">> Custom loss: ", custom_loss.numpy())




