import numpy as np
import torch
from torch import nn

from DenseLayer import DenseLayer
from RnnLayer import RnnLayer
from loss_functions import CrossEntropyLoss

RNN_TYPES = ['RNN', 'LSTM', 'GRU']


class RNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, output_dim,
                 rnn_type='RNN'):
        super().__init__()
        self.output_dim = output_dim

        assert rnn_type in RNN_TYPES, f'Use one of the following: {str(RNN_TYPES)}'
        RnnCell = getattr(nn, rnn_type)
        self.rnn = RnnCell(embedding_dim, hidden_dim, batch_first=True, bias=False)
        self.fc = nn.Linear(hidden_dim, output_dim, bias=False)

    def forward(self, X):
        rnn_out, _ = self.rnn(X)
        fc_out = self.fc(rnn_out)
        return fc_out


batch_size = 32
seq_len = 15
embedding_dim = 100
hidden_dim = 20
vocab_size = 5_000
rnn_type = 'RNN'

X = torch.randn(batch_size, seq_len, embedding_dim)
y = torch.randint(vocab_size, (batch_size * seq_len,))

X_ = X.detach().numpy()
y_ = np.eye(vocab_size)[y.numpy()]
y_ = y_.reshape(batch_size, seq_len, vocab_size)

rnn = RNN(embedding_dim=embedding_dim,
          hidden_dim=hidden_dim,
          output_dim=vocab_size,
          rnn_type='RNN')

cel = nn.CrossEntropyLoss()
out = rnn(X).squeeze().view(batch_size * seq_len, vocab_size)
loss = cel(out, y)

rnn_l = RnnLayer(embedding_dim, hidden_dim, use_bias=False)
dense_l = DenseLayer(hidden_dim, vocab_size, use_bias=False)
cel_ = CrossEntropyLoss()

rnn_l.input_weights = rnn.rnn.weight_ih_l0.detach().numpy()
rnn_l.hidden_weights = rnn.rnn.weight_hh_l0.detach().numpy()
dense_l.weights = rnn.fc.weight.detach().numpy()

rnn_out_, _ = rnn_l.forward(X_)
fc_out = dense_l.forward(rnn_out_)
loss_ = cel_.forward(y_, fc_out[:, 1:, :])
print(np.isclose(loss_, loss.detach().numpy()))
