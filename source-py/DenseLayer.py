import numpy as np


class DenseLayer(object):

    def __init__(self, input_dim, output_dim, use_bias=True):
        sq = np.sqrt(1. / input_dim)
        self.use_bias = use_bias
        self.weights = np.random.uniform(-sq, sq, (output_dim, input_dim))
        if use_bias:
            self.bias = np.random.uniform(-sq, sq, output_dim)
        else:
            self.bias = np.zeros(output_dim)

        self.x_in = None

    def forward(self, x_in):
        self.x_in = x_in
        return np.tensordot(self.x_in, self.weights.T, axes=((-1), 0)) + self.bias

    def backward(self, de_dy):
        # de_dw = de_dy * dYdW = de_dy * X
        # dEdb = de_dy * dYdb = de_dy
        # dEdX = de_dy * dYdX = de_dy * W

        axis = tuple(range(len(self.x_in.shape) - 1))
        de_dw = np.tensordot(de_dy, self.x_in, axes=(axis, axis))
        de_db = np.sum(de_dy, axis=axis)
        de_dx = np.tensordot(de_dy, self.weights, axes=(-1, 0))

        return de_dx, de_dw, de_db
