import torch
from torch import nn

import numpy as np
from GRULayer import GRULayer

N = 5
emb_dim = 5
seq_len = 4
hidden_dim = 5

# N = 20
# emb_dim = 40
# seq_len = 32
# hidden_dim = 200

x = torch.randn(N, seq_len, emb_dim, requires_grad=True)
x_ = x.detach().numpy()

gru = nn.GRU(emb_dim, hidden_dim, bias=False, batch_first=True)
gru_ = GRULayer(emb_dim, hidden_dim, use_bias=False)
wih = gru.weight_ih_l0.detach().numpy()
whh = gru.weight_hh_l0.detach().numpy()

gru_.input_weights[0, :, :] = wih[0:hidden_dim, :]
gru_.input_weights[1, :, :] = wih[hidden_dim: 2 * hidden_dim, :]
gru_.input_weights[2, :, :] = wih[2 * hidden_dim: 3 * hidden_dim, :]

gru_.hidden_weights[0, :, :] = whh[0:hidden_dim, :]
gru_.hidden_weights[1, :, :] = whh[hidden_dim: 2 * hidden_dim, :]
gru_.hidden_weights[2, :, :] = whh[2 * hidden_dim: 3 * hidden_dim, :]

h_0 = torch.randn(1, N, hidden_dim, requires_grad=True)
h_0_ = h_0.detach().numpy().reshape((N, hidden_dim))

gru_out, h_n = gru(x, h_0)
gru_out_, h_n_ = gru_.forward(x_, h_0_)

# gru_out, h_n = gru(x)
# gru_out_, h_n_ = gru_.forward(x_)

gru_out__ = gru_out_[:, 1:, :]  # remove zeros prepended to every hidden output

print('GRU layer forward check: ', np.isclose(gru_out.detach().numpy(), gru_out__, atol=1e-3).all())
print('GRU layer forward check last hidden: ', np.isclose(h_n.detach().numpy(), h_n_, atol=1e-3).all())

de_dy = torch.randn(N, seq_len, hidden_dim)
de_dy_ = de_dy.numpy()

x.retain_grad()
gru_out.backward(de_dy)

dEdW_in, dEdW_hh, a, X_grad = gru_.backward(x_, de_dy_)

print('LSTM layer gradient check dEdW_in: ', np.isclose(gru.weight_ih_l0.grad.numpy(), dEdW_in.reshape(3 * hidden_dim, emb_dim), atol=1e-3).all())
print('LSTM layer gradient check dEdW_hh: ', np.isclose(gru.weight_hh_l0.grad.numpy(), dEdW_hh.reshape(3 * hidden_dim, hidden_dim), atol=1e-3).all())
print('LSTM layer gradient check dEdX: ', np.isclose(x.grad.numpy(), X_grad, atol=1e-3).all())

#print(f'Njihove tezine: {gru.weight_hh_l0.grad}')
#print(f'Moje tezine: {dEdW_hh.reshape(3 * hidden_dim, emb_dim)}')
