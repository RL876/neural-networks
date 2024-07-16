import numpy as np


class Loss:
    def __init__(self):
        pass

    def __call__(self, y_hat, y):
        loss = self._getLoss(y_hat, y)
        acc = self._getAcc(y_hat, y)
        return loss, acc

    def _getAcc(self, y_hat, y):
        m = y.shape[0]
        y_hat_argmax = np.argmax(y_hat, axis=1)
        y_argmax = np.argmax(y, axis=1)
        return (y_hat_argmax == y_argmax).sum() / m

    def _getLoss(self, y_hat, y):
        raise NotImplementedError


class MSE(Loss):
    def __init__(self):
        super().__init__()

    def _getLoss(self, y_hat, y):
        return np.average(np.square(y_hat - y) ** 0.5)


class CategoricalCrossentropy(Loss):
    def __init__(self):
        super().__init__()

    def _getLoss(self, y_hat, y):
        return -np.average(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))
