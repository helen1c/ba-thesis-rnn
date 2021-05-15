import numpy as np
from DenseLayer import DenseLayer
from LSTMLayer import LSTMLayer



class LSTMClassifier(object):
    def __init__(self, input_dim, hidden_dim, vocab_size, embeddings, use_bias=True):
        self.embedding_dim = input_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.embeddings = embeddings
        self.use_bias = use_bias

        self.lstm_layer_0 = LSTMLayer(input_dim, hidden_dim)
        self.lstm_layer_1 = LSTMLayer(hidden_dim, hidden_dim)
        self.dense = DenseLayer(hidden_dim, vocab_size)

        self.inputs = None

        #self.H_0 = None
        #self.H_1 = None

        self.H_0 = None

    def forward(self, inputs):

        #self.inputs = self.get_embeddings(inputs)
        #self.H_0, _, _ = self.lstm_layer_0.forward(self.inputs)
        #self.H_1, _, _ = self.lstm_layer_1.forward(self.H_0[:, 1:, :])
        #return self.dense.forward(self.H_1[:, 1:, :])

        self.inputs = self.get_embeddings(inputs)
        self.H_0, _, _ = self.lstm_layer_0.forward(self.inputs)
        return self.dense.forward(self.H_0[:, 1:, :])

    def backward(self, dEdY):
        #2 sloja
        #dense_x, dense_w, dense_b = self.dense.backward(dEdY)
        #lstm_1_wih, lstm_1_whh, lstm_1_b, x_grad_1 = self.lstm_layer_1.backward(self.H_0[:, 1:, :], dense_x)
        #lstm_0_wih, lstm_0_whh, lstm_0_b, _ = self.lstm_layer_0.backward(self.inputs, x_grad_1)

        dense_x, dense_w, dense_b = self.dense.backward(dEdY)
        lstm_0_wih, lstm_0_whh, lstm_0_b, _ = self.lstm_layer_0.backward(self.inputs, dense_x)

        #2 sloja
        #return [dense_w, dense_b, lstm_1_wih, lstm_1_whh, lstm_1_b, lstm_0_wih, lstm_0_whh, lstm_0_b], \
        #       [self.dense.weights, self.dense.bias, self.lstm_layer_1.input_weights, self.lstm_layer_1.hidden_weights,
        #        self.lstm_layer_1.bias, self.lstm_layer_0.input_weights, self.lstm_layer_0.hidden_weights,
        #        self.lstm_layer_0.bias]

        return [dense_w, dense_b, lstm_0_wih, lstm_0_whh, lstm_0_b], \
               [self.dense.weights, self.dense.bias, self.lstm_layer_0.input_weights, self.lstm_layer_0.hidden_weights,
                self.lstm_layer_0.bias]

    def get_embeddings(self, mini_batch):
        w2vec = np.zeros((mini_batch.shape[0], mini_batch.shape[1], self.embeddings.shape[1]))
        for m in range(mini_batch.shape[0]):
            for j in range(mini_batch.shape[1]):
                w2vec[m, j, :] = self.embeddings[mini_batch[m, j]]

        return w2vec
