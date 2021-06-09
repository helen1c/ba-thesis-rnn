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
        return np.einsum('b...i,ih->b...h', self.x_in, self.weights.T) + self.bias

    def backward(self, de_dy):
        # de_dw = de_dy * dYdW = de_dy * X
        # dEdb = de_dy * dYdb = de_dy
        # dEdX = de_dy * dYdX = de_dy * W

        #einsum nema mogucnost sumiranja po opcionalnim
        #dimenzijama, ako barem jedan od argumenata nije fiksan
        #zato se koristi tensordot
        axis = tuple(range(len(self.x_in.shape) - 1))
        de_dw = np.tensordot(de_dy, self.x_in, axes=(axis, axis))
        de_db = de_dy.sum(axis=axis)
        de_dx = np.einsum('b...h,hi->b...i', de_dy, self.weights)

        return de_dx, de_dw, de_db

