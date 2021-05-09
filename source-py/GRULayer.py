from activations import Sigmoid, Tanh
import numpy as np


class GRULayer(object):

    def __init__(self, input_dim, hidden_dim, use_bias=True):

        # r_t = sigmoid(W_r_hi.x_t + W_r_hh.h_(t-1) + b_r)
        # z_t = sigmoid(W_z_hi.x_t + W_z_hh.h_(t-1) + b_z)
        # c_t = tanh(W_n_hi.x_t + W_n_hh.h_(t-1) * r_t + b_c)
        # h_t = (1-z_t) * n_t + z_t * h_(t-1)

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.use_bias = use_bias

        sq = np.sqrt(1. / hidden_dim)
        # input weights [W_r_hi,W_z_hi,W_c_hi]
        self.input_weights = np.random.uniform(-sq, sq, (3, hidden_dim, input_dim))
        # hidden weights [W_r_hi,W_z_hi,W_c_hi]
        self.hidden_weights = np.random.uniform(-sq, sq, (3, hidden_dim, hidden_dim))

        self.tanh = Tanh
        self.sigmoid = Sigmoid

        self.gates = None
        self.H = None
        self.C = None

        if self.use_bias:
            # bias = [r_bias,z_bias,c_bias]
            self.bias = np.random.uniform(-sq, sq, (3, hidden_dim))
        else:
            self.bias = np.zeros((3, hidden_dim))

    def forward(self, X_in, h_0=None):
        batch_size = X_in.shape[0]
        seq_len = X_in.shape[1]

        self.H = np.zeros((batch_size, seq_len + 1, self.hidden_dim))
        if h_0 is not None:
            self.H[:, 0, :] = h_0

        self.gates = np.zeros((3, batch_size, seq_len, self.hidden_dim))

        for i in range(seq_len):
            # reset_gate
            self.gates[0, :, i, :] = self.sigmoid.forward(
                np.dot(X_in[:, i, :], self.input_weights[0, :, :].T) + np.dot(self.H[:, i, :], self.hidden_weights[0, :, :].T) + self.bias[0, :])
            # z_gate
            self.gates[1, :, i, :] = self.sigmoid.forward(
                np.dot(X_in[:, i, :], self.input_weights[1, :, :].T) + np.dot(self.H[:, i, :], self.hidden_weights[1, :, :].T) + self.bias[1, :])
            # update gate
            self.gates[2, :, i, :] = self.tanh.forward(
                np.dot(X_in[:, i, :], self.input_weights[2, :, :].T) + self.gates[0, :, i, :] * np.dot(self.H[:, i, :], self.hidden_weights[2, :, :].T) + self.bias[2, :])

            self.H[:, i + 1, :] = self.gates[1, :, i, :] * self.H[:, i, :] + (1 - self.gates[1, :, i, :]) * self.gates[2, :, i, :]

        return self.H, self.H[:, seq_len, :]

    def backward(self, X_in, dEdY):
        batch_size = X_in.shape[0]
        seq_len = X_in.shape[1]

        dEdW_in = np.zeros_like(self.input_weights)
        dEdW_hh = np.zeros_like(self.hidden_weights)
        dEdB_in = np.zeros_like(self.bias)

        H_grad = np.zeros((batch_size, seq_len, self.hidden_dim))
        X_grad = np.zeros((batch_size, seq_len, self.input_dim))

        gates_grad = np.zeros((3, batch_size, seq_len, self.hidden_dim))

        for i in range(seq_len - 1, -1, -1):

            if i < seq_len - 1:
                fs = np.dot(gates_grad[1, :, i + 1, :], self.hidden_weights[1, :, :])
                ss = self.gates[0, :, i, :] * (np.dot(gates_grad[2, :, i + 1, :], self.hidden_weights[2, :, :]))
                ts = dEdY[:, i, :] * self.gates[1, :, i, :]
                fos = np.dot(gates_grad[0, :, i + 1, :], self.hidden_weights[0, :, :])
                H_grad[:, i, :] = fs + ss + ts + fos
            else:
                H_grad[:, i, :] = dEdY[:, i, :]

            gates_grad[2, :, i, :] = ((1 - self.gates[1, :, i, :]) * H_grad[:, i, :]) * self.tanh.backward_calculated(self.gates[2, :, i, :])
            gates_grad[1, :, i, :] = ((self.H[:, i, :] * H_grad[:, i, :]) + (-1 * self.gates[2, :, i, :] * H_grad[:, i, :])) * self.sigmoid.backward_calculated(self.gates[1, :, i, :])
            gates_grad[0, :, i, :] = (np.dot(gates_grad[2, :, i, :], self.hidden_weights[2, :, :]) * self.H[:, i, :]) * self.sigmoid.backward_calculated(self.gates[0, :, i, :])

            X_grad[:, i, :] = np.dot(gates_grad[2, :, i, :], self.input_weights[2, :, :]) + np.dot(gates_grad[1, :, i, :], self.input_weights[1, :, :]) + np.dot(gates_grad[0, :, i, :],
                                                                                                                                                                 self.input_weights[0, :, :])
            h_t_T = self.H[:, i, :].T

            dEdW_in[0, :, :] += np.dot(gates_grad[0, :, i, :].T, X_in[:, i, :])
            dEdW_in[1, :, :] += np.dot(gates_grad[1, :, i, :].T, X_in[:, i, :])
            dEdW_in[2, :, :] += np.dot(gates_grad[2, :, i, :].T, X_in[:, i, :])

            if i < seq_len - 1:
                dEdW_hh[0, :, :] += np.dot(h_t_T, gates_grad[0, :, i, :])
                dEdW_hh[1, :, :] += np.dot(h_t_T, gates_grad[1, :, i, :])
                dEdW_hh[2, :, :] += np.dot((self.H[:, i, :] * self.gates[0, :, i, :]).T, gates_grad[2, :, i, :])

            if self.use_bias:
                dEdB_in[0, :] += np.sum(gates_grad[0, :, i, :], axis=0)
                dEdB_in[1, :] += np.sum(gates_grad[1, :, i, :], axis=0)
                dEdB_in[2, :] += np.sum(gates_grad[2, :, i, :], axis=0)

        return dEdW_in, dEdW_hh, dEdB_in, X_grad
