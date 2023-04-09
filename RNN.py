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

set_seed(42)
np.random.seed(42)


class simple_rnn():
    """
    here we implement the simple keras RNN
    """
    model = Sequential()
    def __init__(self, X_train, y_train, X_test, y_test, epochs, batch_size, neurons, dropout, optimizer, loss, metrics):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.neurons = neurons
        self.dropout = dropout
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def build_model(self):
        self.model.add(SimpleRNN(self.neurons, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(1))
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return self.model

    def fit_model(self):
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=False)
        return self.model


class lstm():
    """
    here we implement the simple keras LSTM
    """
    model = Sequential()
    def __init__(self, X_train, y_train, X_test, y_test, epochs, batch_size, neurons, dropout, optimizer, loss, metrics):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test
        self.epochs = epochs
        self.batch_size = batch_size
        self.neurons = neurons
        self.dropout = dropout
        self.optimizer = optimizer
        self.loss = loss
        self.metrics = metrics

    def build_model(self):
        self.model.add(LSTM(self.neurons, input_shape=(self.X_train.shape[1], self.X_train.shape[2])))
        self.model.add(Dropout(self.dropout))
        self.model.add(Dense(1))
        self.model.compile(optimizer=self.optimizer, loss=self.loss, metrics=self.metrics)
        return self.model

    def fit_model(self):
        self.model.fit(self.X_train, self.y_train, epochs=self.epochs, batch_size=self.batch_size, verbose=1, shuffle=False)
        return self.model

    

    
