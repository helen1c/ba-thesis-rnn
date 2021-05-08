import numpy as np

from activations import Sigmoid
from activations import Tanh


class LSTMLayer(object):

    def __init__(self, input_dim, hidden_dim, use_bias=True):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias

        sq = np.sqrt(1. / hidden_dim)
        # input weights (W_in_hi|W_fgt_hi|W_g_hi|W_out_hi)
        self.input_weights = np.random.uniform(-sq, sq, (4, hidden_dim, input_dim))
        # hidden weights (W_in_hh|W_fgt_hh|W_g_hh|W_out_hh)
        self.hidden_weights = np.random.uniform(-sq, sq, (4, hidden_dim, hidden_dim))

        self.tanh = Tanh
        self.sigmoid = Sigmoid

        self.gates = None
        self.H = None
        self.C = None

        if self.use_bias:
            # bias = (in_bias|fgt_bias|g_bias|out_bias)
            self.bias = np.random.uniform(-sq, sq, (4, hidden_dim))
        else:
            self.bias = np.zeros((4, hidden_dim))

    def forward(self, X_in, h_0=None, c_0=None):
        batch_size = X_in.shape[0]
        seq_len = X_in.shape[1]

        self.H = np.zeros((batch_size, seq_len + 1, self.hidden_dim))
        if h_0 is not None:
            self.H[:, 0, :] = h_0

        self.C = np.zeros((batch_size, seq_len + 1, self.hidden_dim))
        if c_0 is not None:
            self.C[:, 0, :] = c_0

        self.gates = np.zeros((4, batch_size, seq_len, self.hidden_dim))

        for i in range(seq_len):
            # input_gate
            self.gates[0, :, i, :] = self.sigmoid.forward(
                np.dot(X_in[:, i, :], self.input_weights[0, :, :].T) + np.dot(self.H[:, i, :], self.hidden_weights[0, :, :].T) + self.bias[0, :])
            # forget gate
            self.gates[1, :, i, :] = self.sigmoid.forward(
                np.dot(X_in[:, i, :], self.input_weights[1, :, :].T) + np.dot(self.H[:, i, :], self.hidden_weights[1, :, :].T) + self.bias[1, :])
            # c~ gate
            self.gates[2, :, i, :] = self.tanh.forward(
                np.dot(X_in[:, i, :], self.input_weights[2, :, :].T) + np.dot(self.H[:, i, :], self.hidden_weights[2, :, :].T) + self.bias[2, :])
            # output gate
            self.gates[3, :, i, :] = self.sigmoid.forward(
                np.dot(X_in[:, i, :], self.input_weights[3, :, :].T) + np.dot(self.H[:, i, :], self.hidden_weights[3, :, :].T) + self.bias[3, :])

            self.C[:, i + 1, :] = self.gates[1, :, i, :] * self.C[:, i, :] + self.gates[0, :, i, :] * self.gates[2, :, i, :]
            self.H[:, i + 1, :] = self.gates[3, :, i, :] * self.tanh.forward(self.C[:, i + 1, :])

        return self.H, self.H[:, seq_len, :], self.C[:, seq_len, :]

    def backward(self, X_in, dEdY):

        batch_size = X_in.shape[0]
        seq_len = X_in.shape[1]

        dEdW_in = np.zeros_like(self.input_weights)
        dEdW_hh = np.zeros_like(self.hidden_weights)
        dEdB_in = np.zeros_like(self.bias)

        H_grad = np.zeros((batch_size, seq_len, self.hidden_dim))
        C_grad = np.zeros((batch_size, seq_len, self.hidden_dim))
        X_grad = np.zeros((batch_size, seq_len, self.input_dim))

        gates_grad = np.zeros((4, batch_size, seq_len, self.hidden_dim))

        for i in range(seq_len - 1, -1, -1):

            if i < seq_len - 1:
                H_grad[:, i, :] = np.matmul(gates_grad[:, :, i + 1, :], self.hidden_weights).sum(axis=0) + dEdY[:, i, :]
                C_grad[:, i, :] = H_grad[:, i, :] * self.gates[3, :, i, :] * self.tanh.backward(self.C[:, i + 1, :]) + C_grad[:, i + 1, :] * self.gates[1, :, i + 1, :]
            else:
                H_grad[:, i, :] = dEdY[:, i, :]
                C_grad[:, i, :] = H_grad[:, i, :] * self.gates[3, :, i, :] * self.tanh.backward(self.C[:, i + 1, :])

            gates_grad[0, :, i, :] = C_grad[:, i, :] * self.gates[2, :, i, :] * self.sigmoid.backward_calculated(self.gates[0, :, i, :])
            gates_grad[1, :, i, :] = C_grad[:, i, :] * self.C[:, i, :] * self.sigmoid.backward_calculated(self.gates[1, :, i, :])
            gates_grad[2, :, i, :] = C_grad[:, i, :] * self.gates[0, :, i, :] * self.tanh.backward_calculated(self.gates[2, :, i, :])
            gates_grad[3, :, i, :] = H_grad[:, i, :] * self.tanh.forward(self.C[:, i + 1, :]) * self.sigmoid.backward_calculated(self.gates[3, :, i, :])

            X_grad[:, i, :] = np.matmul(gates_grad[:, :, i, :], self.input_weights).sum(axis=0)

            dEdW_in[0, :, :] += np.einsum('bi,bo->bio', gates_grad[0, :, i, :], X_in[:, i, :]).sum(axis=0)
            dEdW_in[1, :, :] += np.einsum('bi,bo->bio', gates_grad[1, :, i, :], X_in[:, i, :]).sum(axis=0)
            dEdW_in[2, :, :] += np.einsum('bi,bo->bio', gates_grad[2, :, i, :], X_in[:, i, :]).sum(axis=0)
            dEdW_in[3, :, :] += np.einsum('bi,bo->bio', gates_grad[3, :, i, :], X_in[:, i, :]).sum(axis=0)

            if i < seq_len - 1:
                dEdW_hh[0, :, :] += np.einsum('bi,bo->bio', gates_grad[0, :, i + 1, :], self.H[:, i + 1, :]).sum(axis=0)
                dEdW_hh[1, :, :] += np.einsum('bi,bo->bio', gates_grad[1, :, i + 1, :], self.H[:, i + 1, :]).sum(axis=0)
                dEdW_hh[2, :, :] += np.einsum('bi,bo->bio', gates_grad[2, :, i + 1, :], self.H[:, i + 1, :]).sum(axis=0)
                dEdW_hh[3, :, :] += np.einsum('bi,bo->bio', gates_grad[3, :, i + 1, :], self.H[:, i + 1, :]).sum(axis=0)

            if self.use_bias:
                dEdB_in[0, :] += np.sum(gates_grad[0, :, i, :], axis=0)
                dEdB_in[1, :] += np.sum(gates_grad[1, :, i, :], axis=0)
                dEdB_in[2, :] += np.sum(gates_grad[2, :, i, :], axis=0)
                dEdB_in[3, :] += np.sum(gates_grad[3, :, i, :], axis=0)

        return dEdW_in, dEdW_hh, dEdB_in, X_grad
