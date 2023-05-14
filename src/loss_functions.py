import tensorflow as tf

SIGMA = 10 
BETA = 8/3
RHO = 28

# here we will add different loss functions to the model, trying to investigate how adding physics to the model affects the loss function
class loss():
    def __init__(self, pos_true, pos_pred):
        self.pos_true = pos_true
        self.pos_pred = pos_pred

        self.px_true = -SIGMA * (self.pos_true[:, 0] - self.pos_true[:, 1])
        self.py_true = RHO * self.pos_true[:, 0] - self.pos_true[:, 1] - self.pos_true[:, 0] * self.pos_true[:, 2]
        self.pz_true = -BETA * self.pos_true[:, 2] + self.pos_true[:, 0] * self.pos_true[:, 1]

        self.px_pred = -SIGMA * (self.pos_pred[:, 0] - self.pos_pred[:, 1])
        self.py_pred = RHO * self.pos_pred[:, 0] - self.pos_pred[:, 1] - self.pos_pred[:, 0] * self.pos_pred[:, 2]
        self.pz_pred = -BETA * self.pos_pred[:, 2] + self.pos_pred[:, 0] * self.pos_pred[:, 1]

    def mse_loss(self):
        mse_loss = tf.keras.losses.MeanSquaredError()
        return mse_loss(self.pos_true, self.pos_pred)

    def _momentum_conservation_loss(self): 
        mse_loss = tf.keras.losses.MeanSquaredError()
        return mse_loss(self.px_true, self.px_pred) + mse_loss(self.py_true, self.py_pred) + mse_loss(self.pz_true, self.pz_pred)

    def _energy_conservation_loss(self):
        mse_loss = tf.keras.losses.MeanSquaredError()
        return mse_loss(self.px_true**2 + self.py_true**2 + self.pz_true**2, self.px_pred**2 + self.py_pred**2 + self.pz_pred**2)

    def custom_loss(self, momentum_conservation=False, momentum_weight=0.5, energy_conservation=False, energy_weight=0.5):
        mse_loss = self.mse_loss()
        energy_loss = 0
        momentum_loss = 0

        if energy_conservation:
            energy_loss = energy_weight * self._energy_conservation_loss()

        if momentum_conservation:
            momentum_loss = momentum_weight * self._momentum_conservation_loss()

        return mse_loss + momentum_loss + energy_loss






