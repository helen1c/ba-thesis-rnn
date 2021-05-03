import numpy as np

from RnnLayer import RnnLayer
from DenseLayer import DenseLayer
from loss_functions import CrossEntropyLoss
from activations import Softmax

# def __init__rnnlayer(self, input_dim, hidden_dim, seq_len, batch_size, use_bias=True, activation=Tanh):
# def __init__dense_layer(self, input_dim, output_dim, use_bias=True):

batch_size = 3  # Number of training samples
# Addition of 2 n-bit numbers can result in a n+1 bit number
sequence_len = 3  # Length of the binary sequence

rnn = RnnLayer(2, 3, sequence_len, batch_size)
dense = DenseLayer(3, 2)
clos = CrossEntropyLoss()

X = np.array([[[0., 1.], [1., 0.], [0., 1.]],
              [[0., 1.], [0., 1.], [0., 1.]],
              [[1., 0.], [1., 0.], [1., 0.]]])

T = np.array([[[0., 1.], [1., 0.], [0., 1.]],
              [[0., 1.], [0., 1.], [0., 1.]],
              [[1., 0.], [1., 0.], [1., 0.]]])

num_iter = 1000
learning_rate = 0.1

for i in range(num_iter):
    H, _ = rnn.forward(X)
    out = dense.forward(H[:, 1:, :])
    loss = clos.forward(T, out)

    print(f'{i + 1}. iteracija- loss: {loss}')

    dEdY = clos.backward(T)

    de_dx, de_dw, de_db_d = dense.backward(dEdY, H[:, 1:, :])
    dEdW_in, dEdW_hh, de_db_r = rnn.backward(X, H, de_dx)

    dense.weights = dense.weights - learning_rate * de_dw
    if dense.use_bias:
        dense.bias = dense.bias - learning_rate * de_db_d
    rnn.input_weights = rnn.input_weights - learning_rate * dEdW_in
    rnn.hidden_weights = rnn.hidden_weights - learning_rate * dEdW_hh
    if rnn.use_bias:
        rnn.bias = rnn.bias - learning_rate * de_db_r

X_t = np.array([[[1., 0.], [1., 0.], [0., 1.]],
                [[1., 0.], [0., 1.], [1., 0.]],
                [[0., 1.], [1., 0.], [0., 1.]]])

H, _ = rnn.forward(X_t)
out = dense.forward(H[:, 1:, :])
print(Softmax.forward(out))

