import numpy as np


class Activation(object):
    def __init__(self):
        pass

    def forward():
        raise NotImplementedError

    def backward():
        raise NotImplementedError


class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return 1 / (1 + np.exp(-z))

    def backward(self, a):
        z_sig = Sigmoid.forward(a)
        return z_sig * (1 - z_sig)


class Relu(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        return np.maximum(0, z)

    def backward(self, a):
        return 1 * (a > 0)


class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        z_exp = np.exp(z)
        y_hat = z_exp / np.sum(z_exp, axis=1, keepdims=True)
        y_hat = np.where(y_hat == 0, 10**-10, y_hat)
        return y_hat

    def backward(self, a, y):
        return a - y
