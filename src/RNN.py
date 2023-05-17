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

rng = np.random.default_rng(2048)



class Simple_rnn():
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
        self.name = f"LSTM with {self.n_layers} layers and {self.n_hidden} hidden nodes \n Input shape: {self.input_shape}"

    def __repr__(self) -> str:
        return self.name

    def build(self, optimizer, loss, dropout = 0):
        
        for _ in range(self.n_layers):
            self.model.add(SimpleRNN(self.n_hidden, input_shape=self.input_shape))
        #self.model.add(Dropout(dropout))

        self.model.add(Dense(self.input_shape[1])) # output layer, input_shape[1] = spacial_dim
        self.model.compile(optimizer=optimizer, loss=loss)
    

    def fit(self, X_train, y_train, epochs):
        # add warning that batch_size is default to None
        print("Warning: Batch size is default to None")

        self.model.fit(X_train, y_train, epochs=epochs, batch_size=None, verbose=1, )
        

    def test(self, ic, n_steps=0):
        predicted_sequence = []
        current_step = ic[0].reshape(-1, self.sequence_length, self.spacial_dim) # Initialize the current 2 step with the input data
        print(current_step.shape)
        if not n_steps:
            n_steps = len(ic)
        for _ in range(n_steps):
            predicted_step = self.model.predict(current_step)
            predicted_sequence.append(predicted_step)
            # Update the current step by shifting the window
            current_step = np.concatenate([current_step[:, 1:, :], predicted_step.reshape(1,1, self.spacial_dim)], axis=1)

        predicted_sequence = np.array(predicted_sequence)

        # processed_sequence = write_result(predicted_sequence)
        return predicted_sequence

    def summary(self):
        return self.model.summary()



class Lstm():
    """
    here we implement the simple keras LSTM
    """
    
    def __init__(self, n_hidden=0, n_layers=0, input_shape=(0,0)):
        self.model = Sequential()
        self.input_shape = input_shape
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.sequence_length = input_shape[0]
        self.spacial_dim = input_shape[1]
        self.name = f"lorenz-lstm-{self.n_layers}-layers-{self.n_hidden}-hidden_nodes-input_shape{self.input_shape}"

    
    def __repr__(self) -> str:
        return self.name

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
        

    def test(self, ic, n_steps=0):
        predicted_sequence = []
        current_step = ic[0].reshape(-1, self.sequence_length, self.spacial_dim) # Initialize the current 2 step with the input data
        print(current_step.shape)
        if not n_steps:
            n_steps = len(ic)
        for _ in range(n_steps):
            predicted_step = self.model.predict(current_step)
            predicted_sequence.append(predicted_step)
            # Update the current step by shifting the window
            current_step = np.concatenate([current_step[:, 1:, :], predicted_step.reshape(1,1, self.spacial_dim)], axis=1)

        predicted_sequence = np.array(predicted_sequence)

        # processed_sequence = write_result(predicted_sequence)
        return predicted_sequence

    def summary(self):
        return self.model.summary()

    def my_save(self, model_name):
        self.model.save(model_name)
    
    def my_load(self, model_name):
        self.model = keras.models.load_model(model_name)
        self.name = model_name
        self.input_shape = self.model.layers[0].input_shape
        self.sequence_length = self.input_shape[1]
        self.spacial_dim = self.input_shape[2]


    
