import numpy as np
from DenseLayer import DenseLayer
from RNNLayer import RnnLayer
from activations import ReLu

class RNNClassifier(object):
    def __init__(self, input_dim, hidden_dim, vocab_size, embeddings, use_bias=True):
        self.embedding_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embeddings = embeddings
        self.use_bias = use_bias

        self.rnn_layer_0 = RnnLayer(input_dim, hidden_dim, use_bias=use_bias, activation=ReLu)
        self.rnn_layer_1 = RnnLayer(hidden_dim, hidden_dim, use_bias=use_bias, activation=ReLu)
        self.dense = DenseLayer(hidden_dim, vocab_size)

        self.inputs = None

        self.H_0 = None
        self.H_1 = None

    def forward(self, inputs):

        self.inputs = self.get_embeddings(inputs)
        self.H_0, _ = self.rnn_layer_0.forward(self.inputs)
        self.H_1, _ = self.rnn_layer_1.forward(self.H_0[:, 1:, :])
        return self.dense.forward(self.H_1[:, 1:, :])

    def backward(self, dEdY):
        dense_x, dense_w, dense_b = self.dense.backward(dEdY)
        rnn_1_wih, rnn_1_whh, rnn_1_b, rnn_1_x = self.rnn_layer_1.backward(self.H_0[:, 1:, :], self.H_1, dense_x)
        rnn_0_wih, rnn_0_whh, rnn_0_b, _ = self.rnn_layer_0.backward(self.inputs, self.H_0, rnn_1_x)

        return [dense_w, rnn_1_wih, rnn_1_whh, rnn_0_wih, rnn_0_whh, dense_b, rnn_0_b, rnn_1_b], [self.dense.weights, self.rnn_layer_1.input_weights, self.rnn_layer_1.hidden_weights,
                                                                       self.rnn_layer_0.input_weights, self.rnn_layer_0.hidden_weights, self.dense.bias, self.rnn_layer_0.bias, self.rnn_layer_1.bias]

    def get_embeddings(self, mini_batch):
        w2vec = np.zeros((mini_batch.shape[0], mini_batch.shape[1], self.embeddings.shape[1]))
        for m in range(mini_batch.shape[0]):
            for j in range(mini_batch.shape[1]):
                w2vec[m, j, :] = self.embeddings[mini_batch[m, j]]

        return w2vec
