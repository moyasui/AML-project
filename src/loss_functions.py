import keras.backend as K

from tensorflow.python.ops import math_ops
import numpy as np
import tensorflow as tf
#tf.config.experimental_run_functions_eagerly(True)

SIGMA = 10 
BETA = 8/3
RHO = 28

# here we will add different loss functions to the model, trying to investigate how adding physics to the model affects the loss function
class loss():
    def __init__(self, momentum_conservation=False, momentum_weight=0.5):

        self.momentum_conservation = momentum_conservation
        self.momentum_weight = momentum_weight

        self.pos_true_x = None
        self.pos_true_y = None
        self.pos_true_z = None
        self.pos_pred_x = None
        self.pos_pred_y = None
        self.pos_pred_z = None

        self.dx_dt = None
        self.dy_dt = None
        self.dz_dt = None

        self.px_true  = None
        self.py_true  = None
        self.pz_true  = None

        self.px_pred  = None
        self.py_pred  = None
        self.pz_pred  = None

    def _mse_loss(self, pos_true, pos_pred):
        """
        simple mse implementation manually
        """
        squared_error = math_ops.squared_difference(pos_true, pos_pred)
        return K.mean(squared_error, axis=-1)
        

    def _momentum_conservation_loss(self): 
        # loss_x = self._mse_loss(self.dx_dt, self.px_pred[1:-1])
        # loss_y = self._mse_loss(self.dy_dt, self.py_pred[1:-1])
        # loss_z = self._mse_loss(self.dz_dt, self.pz_pred[1:-1])

        loss_x = self._mse_loss(self.px_true, self.px_pred)
        loss_y = self._mse_loss(self.px_true, self.py_pred)
        loss_z = self._mse_loss(self.px_true, self.pz_pred)

        return loss_x + loss_y + loss_z
        

    def custom_loss(self, pos_true, pos_pred):
        """
        this is a custom function that will be used in a keras model.
        we are then going to be using keras.backend to do the calculations instead of tensorflow
        """


        self.pos_true_x = K.flatten(pos_true[:, 0])
        self.pos_true_y = K.flatten(pos_true[:, 1])
        self.pos_true_z = K.flatten(pos_true[:, 2])

        self.pos_pred_x = K.flatten(pos_pred[:, 0])
        self.pos_pred_y = K.flatten(pos_pred[:, 1])
        self.pos_pred_z = K.flatten(pos_pred[:, 2])

        mse_loss_x = self._mse_loss(self.pos_true_x, self.pos_pred_x)
        mse_loss_y = self._mse_loss(self.pos_true_y, self.pos_pred_y)
        mse_loss_z = self._mse_loss(self.pos_true_z, self.pos_pred_z)

        mse_loss = mse_loss_x + mse_loss_y + mse_loss_z
        

        self.px_true = K.flatten(SIGMA * (self.pos_true_y - self.pos_true_x))
        self.py_true = K.flatten(self.pos_true_x *(RHO - self.pos_true_z) - self.pos_true_y)
        self.pz_true = K.flatten(self.pos_true_z * (self.pos_true_y - BETA))

        self.px_pred = K.flatten(SIGMA * (self.pos_pred_y - self.pos_pred_x))
        self.py_pred = K.flatten(self.pos_pred_x *(RHO - self.pos_pred_z) - self.pos_pred_y)
        self.pz_pred = K.flatten(self.pos_pred_z * (self.pos_pred_y - BETA))

        # dt = 1
       
        # self.dx_dt = (self.pos_pred_x[2:] -  self.pos_pred_x[:-2]) / 2*dt
        # self.dy_dt = (self.pos_pred_y[2:] - self.pos_pred_y[:-2] )/  2*dt
        # self.dz_dt = (self.pos_pred_z[2:] - self.pos_pred_z[:-2] )/  2*dt
        

        energy_loss = 0
        momentum_loss = 0


        if self.momentum_conservation:
            momentum_loss = self.momentum_weight * self._momentum_conservation_loss()

        return mse_loss + energy_loss + momentum_loss






