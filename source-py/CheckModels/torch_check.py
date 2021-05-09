import torch
from torch import nn

import numpy as np

from RnnLayer import RnnLayer
N = 32
emb_dim = 300
seq_len = 28
hidden_dim = 200


x = torch.randn(N, seq_len, emb_dim, requires_grad=True)

x_ = x.detach().numpy()

rnn = nn.RNN(emb_dim, hidden_dim, bias=False, batch_first=True)
rnn_ = RnnLayer(emb_dim, hidden_dim, use_bias=False)
rnn_.input_weights = rnn.weight_ih_l0.detach().numpy()
rnn_.hidden_weights = rnn.weight_hh_l0.detach().numpy()

rnn_out, h_n = rnn(x)
rnn_out_, h_n_ = rnn_.forward(x_)
rnn_out__ = rnn_out_[:, 1:, :] # remove zeros prepended to every hidden output

print('RNN layer forward check: ', np.isclose(rnn_out.detach().numpy(), rnn_out__).all())
print('RNN layer forward check last hidden: ', np.isclose(h_n.detach().numpy(), h_n_).all())
de_dy = torch.randn(N, seq_len, hidden_dim)
de_dy_ = de_dy.numpy()

rnn_out.retain_grad()
rnn_out.backward(de_dy)

print(f'[TORCH] dE/dWih:\n{rnn.weight_ih_l0.grad}\n')
print(f'[TORCH] dE/dWhh:\n{rnn.weight_hh_l0.grad}\n')
#print(f'[TORCH] dE/dy:\n{de_dy}\n')
#print(f'[TORCH] dE/dH:\n{rnn_out.grad}\n')
dEdW_in, dEdW_hh, _ = rnn_.backward(x_, rnn_out_, de_dy_)

print(f'[CUSTOM] dE/dWih:\n{dEdW_in}\n')
print(f'[CUSTOM] dE/dWhh:\n{dEdW_hh}\n')


print('RNN layer gradient check dEdW_in: ', np.isclose(rnn.weight_ih_l0.grad.numpy(), dEdW_in, atol=1e-3).all())
print('RNN layer gradient check dEdW_hh: ', np.isclose(rnn.weight_hh_l0.grad.numpy(), dEdW_hh, atol=1e-3).all())
#print('RNN layer gradient check dEdH: ', np.isclose(rnn_out.grad.numpy(), H_grad[:,1:,:]).all())