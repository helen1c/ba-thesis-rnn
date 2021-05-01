import numpy as np


class Softmax(object):

    @staticmethod
    def forward(x_in):
        return np.exp(x_in) / np.sum(np.exp(x_in), axis=0)


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
