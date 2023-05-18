# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
import random

seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)
tf.keras.utils.set_random_seed(seed)


class simple_rnn():
    """
    here we implement the simple keras RNN
    """
    def __init__(self, n_hidden, n_layers, input_shape):
        self.model = Sequential()
        self.input_shape = input_shape
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.sequence_length = input_shape[0]
        self.spacial_dim = input_shape[1]
        self.history = None


    def build(self, optimizer, loss, dropout = 0):
        
        for _ in range(self.n_layers):
            self.model.add(SimpleRNN(self.n_hidden, input_shape=self.input_shape))
        #self.model.add(Dropout(dropout))

        self.model.add(Dense(self.input_shape[1])) # output layer, input_shape[1] = spacial_dim
        self.model.compile(optimizer=optimizer, loss=loss)
    

    def fit(self, X_train, y_train, epochs):
        # add warning that batch_size is default to None
        print("Warning: Batch size is default to None")

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=None, verbose=1 )
        

    def test(self, sequenced_test_inputs, seq_indx=0):
        predicted_sequence = []
        current_step = sequenced_test_inputs[seq_indx][0].reshape(-1, self.sequence_length, self.spacial_dim) # Initialize the current 2 step with the input data
        print(current_step.shape)
        for _ in range(len(sequenced_test_inputs[0])):
            predicted_step = self.model.predict(current_step)
            predicted_sequence.append(predicted_step)
            # Update the current step by shifting the window
            current_step = np.concatenate([current_step[:, 1:, :], predicted_step.reshape(1,1, self.spacial_dim)], axis=1)

        predicted_sequence = np.array(predicted_sequence)

        # processed_sequence = write_result(predicted_sequence)
        return predicted_sequence

    def summary(self):
        return self.model.summary()
    
    def get_history(self):
        return self.history



class lstm():
    """
    here we implement the simple keras LSTM
    """
    
    def __init__(self, n_hidden, n_layers, input_shape):
        self.model = Sequential()
        self.input_shape = input_shape
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.sequence_length = input_shape[0]
        self.spacial_dim = input_shape[1]
        self.history = None

    def build(self, optimizer, loss, dropout = 0):
        
        for _ in range(self.n_layers):
            self.model.add(LSTM(self.n_hidden, input_shape=self.input_shape))
        #self.model.add(Dropout(dropout))

        self.model.add(Dense(self.input_shape[1])) # output layer, input_shape[1] = spacial_dim
        self.model.compile(optimizer=optimizer, loss=loss)
    

    def fit(self, X_train, y_train, epochs):
        # add warning that batch_size is default to None
        print("Warning: Batch size is default to None")

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=None, verbose=1, )
        

    def test(self, sequenced_test_inputs, seq_indx=0):
        predicted_sequence = []
        current_step = sequenced_test_inputs[seq_indx][0].reshape(-1, self.sequence_length, self.spacial_dim) # Initialize the current 2 step with the input data
        print(current_step.shape)
        for _ in range(len(sequenced_test_inputs[0])):
            predicted_step = self.model.predict(current_step)
            predicted_sequence.append(predicted_step)
            # Update the current step by shifting the window
            current_step = np.concatenate([current_step[:, 1:, :], predicted_step.reshape(1,1, self.spacial_dim)], axis=1)

        predicted_sequence = np.array(predicted_sequence)

        # processed_sequence = write_result(predicted_sequence)
        return predicted_sequence

    def summary(self):
        return self.model.summary()

    def get_history(self):
        return self.history

    

    
