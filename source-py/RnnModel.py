import numpy as np

from RnnLayer import RnnLayer
from DenseLayer import DenseLayer
from loss_functions import CrossEntropyLoss

# def __init__rnnlayer(self, input_dim, hidden_dim, seq_len, batch_size, use_bias=True, activation=Tanh):
# def __init__dense_layer(self, input_dim, output_dim, use_bias=True):


rnn = RnnLayer(2, 3, 3, 3, use_bias=False)
dense = DenseLayer(3, 2, use_bias=False)
clos = CrossEntropyLoss()

X = np.array([[[0., 1.], [1., 0.], [0., 1.]],
              [[0., 1.], [0., 1.], [0., 1.]],
              [[1., 0.], [1., 0.], [1., 0.]]])

T = np.array([[[0., 1.], [1., 0.], [0., 1.]],
              [[0., 1.], [0., 1.], [0., 1.]],
              [[1., 0.], [1., 0.], [1., 0.]]])

num_iter = 3000
learning_rate = 0.003

for i in range(num_iter):

    H, _ = rnn.forward(X)
    out = dense.forward(H[:, 1:, :])
    loss = clos.forward(T, out)

    print(f'{i + 1}. iteracija- loss: {loss}')

    dEdY = clos.backward(T)
    de_dx, de_dw, _ = dense.backward(dEdY, H[:, 1:, :])
    dEdW_in, dEdW_hh, _ = rnn.backward(X, H, de_dx)

    dense.weights = dense.weights - learning_rate * de_dw
    rnn.input_weights = rnn.input_weights - learning_rate * dEdW_in
    rnn.hidden_weights = rnn.hidden_weights - learning_rate * dEdW_hh
