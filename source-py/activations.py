import numpy as np


class Softmax(object):
    @staticmethod
    def forward(x_in):
        exps = np.exp(x_in-np.max(x_in, axis=-1, keepdims=True))
        return exps / np.sum(exps, axis=-1, keepdims=True)


class Tanh(object):

    @staticmethod
    def forward(X_in):
        return np.tanh(X_in)

    @staticmethod
    def backward(X_in):
        # dEdX = dEdY * dYdX = dEdY * 1 - (tanh(X))^2
        return 1 - (np.tanh(X_in)) ** 2

    @staticmethod
    def backward_calculated(tanh_x_in):
        return 1 - tanh_x_in**2


class Sigmoid(object):

    @staticmethod
    def forward(x_in):
        return 1. / (1 + np.exp(-x_in))

    @staticmethod
    def backward(x_in):
        fw = Sigmoid().forward(x_in)
        return fw * (1 - fw)

    @staticmethod
    def backward_calculated(sigmoid_x):
        return sigmoid_x * (1 - sigmoid_x)


class ReLu(object):

    @staticmethod
    def forward(x_in):
        return np.maximum(x_in, 0)

    @staticmethod
    def backward(x_in):
        return x_in > 0

    @staticmethod
    def backward_calculated(relu_x):
        return relu_x > 0
