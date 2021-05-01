import numpy as np


class Softmax(object):
    @staticmethod
    def forward(x_in):
        exps = np.exp(x_in-np.max(x_in, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)


class Tanh(object):

    @staticmethod
    def forward(x_in):
        return np.tanh(x_in)

    @staticmethod
    def backward(x_in):
        # dEdX = dEdY * dYdX = dEdY * 1 - (tanh(X))^2
        return 1 - (np.tanh(x_in)) ** 2


class ReLu(object):

    @staticmethod
    def forward(x_in):
        return np.maximum(x_in, 0)

    @staticmethod
    def backward(x_in):
        return x_in > 0
