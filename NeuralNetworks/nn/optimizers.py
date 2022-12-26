import numpy as np


class Optimizer:
    def __init__(self, learning_rate):
        self.lr = learning_rate

    def __call__(self, data, grad):
        raise NotImplementedError


class Default(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def __call__(self, data, grad):
        for key in data.keys():
            data[key] = data[key] - self.lr * grad[key]
        return data


class AdaGrad(Optimizer):
    def __init__(self, learning_rate):
        super().__init__(learning_rate)

    def __call__(self, data, grad):
        for key in data.keys():
            lr = self.lr * grad[key] ** 2
            data[key] = data[key] - lr / np.sqrt(lr + 1e-8) * grad[key]
        return data
