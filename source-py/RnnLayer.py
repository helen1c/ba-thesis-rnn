import numpy as np

from activations import Tanh


class RnnLayer(object):

    def __init__(self, input_dim, hidden_dim, seq_len, batch_size, use_bias=True, activation=Tanh):
        sq = np.sqrt(1. / hidden_dim)
        self.use_bias = use_bias
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.activation = activation()
        self.input_weights = np.random.uniform(-sq, sq, (hidden_dim, input_dim))
        self.hidden_weights = np.random.uniform(-sq, sq, (hidden_dim, hidden_dim))

        if self.use_bias:
            self.bias = np.random.uniform(-sq, sq, hidden_dim)
        else:
            self.bias = np.zeros(hidden_dim)

    def forward(self, x_in):
        # treba li dodati provjeru je li X_in stvarno ima sekvencu jednaku seq_len?
        # treba li dodati provjeru je li X_in prva koordinata jednaka batch_size

        # u ovom slucaju sam pretpostavio da je za sve inpute, pocetno stanje 0 u 0. vremenskom trenutku
        H = np.zeros((self.batch_size, self.seq_len + 1, self.hidden_dim))

        for i in range(self.seq_len):
            input_part = np.einsum('ij,jk->ik', x_in[:, i, :], self.input_weights.T)
            hidden_part = np.einsum('ij,jk->ik', H[:, i, :], self.hidden_weights.T)

            H[:, i + 1, :] = self.activation.forward(input_part + hidden_part + self.bias)

        return H, H[:, self.seq_len, :]

    def book_forward(self, x_in):

        H = np.zeros((self.batch_size, self.seq_len + 1, self.hidden_dim))

        for i in range(self.seq_len):
            # ovdje dobivam transponirano iz mog forwarda, ali sam u einsum zamijenio vrijednosti, tako da zapravo dobijem isto
            input_part = np.einsum('ij,jk->ki', self.input_weights, x_in[:, i, :].T)
            hidden_part = np.einsum('ij,jk->ik', self.hidden_weights, H[:, i, :].T)

            H[:, i + 1, :] = self.activation.forward(input_part + hidden_part + self.bias)

        return H, H[:, self.seq_len, :]

    def backward(self, x, h, dEdY):
        dEdW_in = np.zeros_like(self.input_weights)
        dEdW_hh = np.zeros_like(self.hidden_weights)

        dEdB_in = np.zeros_like(self.bias)

        H_grad = np.zeros((self.batch_size, self.seq_len + 1, self.hidden_dim))
        H_grad[:, self.seq_len, :] = dEdY[:, self.seq_len - 1, :]

        for i in range(self.seq_len, 0, -1):

            activation_backward = self.activation.backward(h[:, i, :])
            back_reshaped = activation_backward.reshape(self.batch_size, self.hidden_dim, 1)

            dEdW_in += np.sum(back_reshaped * (np.einsum('bh,bi->bhi', H_grad[:, i, :], x[:, i - 1, :])), axis=0)
            dEdW_hh += np.sum(back_reshaped * (np.einsum('bh,bk->bhk', H_grad[:, i, :], h[:, i - 1, :])), axis=0)

            if self.use_bias:
                dEdB_in += np.sum(self.activation.backward(h[:, i, :]) * H_grad[:, i, :], axis=0)
            else:
                pass
            b = np.dot(H_grad[:, i, :], self.hidden_weights)
            a = b * activation_backward

            if i > 1:
                H_grad[:, i - 1, :] = a + dEdY[:, i - 2, :]
            else:
                H_grad[:, i - 1, :] = np.dot(H_grad[:, i, :], self.hidden_weights) * activation_backward

            #if i > 1:
            #    H_grad[:, i - 1, :] = ((np.einsum('bh,hk->bk', H_grad[:, i, :], self.hidden_weights) * activation_backward) + dEdY[:, i - 2, :])
            #else:
            #    H_grad[:, i - 1, :] = np.einsum('bh,hk->bk', H_grad[:, i, :], self.hidden_weights) * activation_backward

        return dEdW_in, dEdW_hh, dEdB_in

    def backward_checker(self, X, H, dEdY):
        dEdW_in = np.zeros_like(self.input_weights)
        dEdW_hh = np.zeros_like(self.hidden_weights)

        print(f'self.bias={self.bias}')

        dEdB_in = np.zeros_like(self.bias)

        H_grad = np.zeros((self.batch_size, self.seq_len + 1, self.hidden_dim))
        H_grad[:, self.seq_len, :] = dEdY[:, self.seq_len - 1, :]

        for i in range(self.seq_len, 0, -1):

            for k in range(self.batch_size):
                act_grad = np.diag(self.activation.backward(H[k, i, :]))
                h_grad = H_grad[k, i, :].reshape(self.hidden_dim, 1)

                dEdW_in += np.dot(act_grad, np.dot(h_grad, X[k, i - 1, :].reshape(1, self.input_dim)))
                dEdW_hh += np.dot(act_grad, np.dot(h_grad, H[k, i - 1, :].reshape(1, self.hidden_dim)))

            if self.use_bias:
                dEdB_in += np.sum(self.activation.backward(H[:, i, :]) * H_grad[:, i, :], axis=(0))
            else:
                pass

            if i > 1:
                H_grad[:, i - 1, :] = np.einsum('bh,hk->bk', H_grad[:, i, :],
                                                self.hidden_weights) * self.activation.backward(H[:, i, :]) + dEdY[:,
                                                                                                              i - 2, :]
            else:
                H_grad[:, i - 1, :] = np.einsum('bh,hk->bk', H_grad[:, i, :],
                                                self.hidden_weights) * self.activation.backward(H[:, i, :])

        return dEdW_in, dEdW_hh, dEdB_in
