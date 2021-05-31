from LSTMLayer import LSTMLayer
import torch
from torch import nn
from timeit import default_timer as timer

import numpy as np

 #N = 5
#emb_dim = 6
#seq_len = 3
#hidden_dim = 8

N = 32
emb_dim = 100
seq_len = 32
hidden_dim = 200

start = timer()

for i in range(1):
    print(f'Iteration number: {i}')
    x = torch.randn(N, seq_len, emb_dim, requires_grad=True)
    x_ = x.detach().numpy()
    lstm = nn.LSTM(emb_dim, hidden_dim, bias=False, batch_first=True)
    lstm_ = LSTMLayer(emb_dim, hidden_dim, use_bias=False)
    wih = lstm.weight_ih_l0.detach().numpy()
    whh = lstm.weight_hh_l0.detach().numpy()

    lstm_.input_weights[0,:,:] = wih[0:hidden_dim, :]
    lstm_.input_weights[1,:,:] = wih[hidden_dim: 2*hidden_dim, :]
    lstm_.input_weights[2,:,:] = wih[2*hidden_dim: 3*hidden_dim, :]
    lstm_.input_weights[3,:,:] = wih[3*hidden_dim: 4*hidden_dim, :]

    lstm_.hidden_weights[0,:,:] = whh[0:hidden_dim, :]
    lstm_.hidden_weights[1,:,:] = whh[hidden_dim: 2*hidden_dim, :]
    lstm_.hidden_weights[2,:,:] = whh[2*hidden_dim: 3*hidden_dim, :]
    lstm_.hidden_weights[3,:,:] = whh[3*hidden_dim: 4*hidden_dim, :]


    lstm_out, h_n = lstm(x)
    lstm_out_, h_n_, c_n_ = lstm_.forward(x_)
    lstm_out__ = lstm_out_[:, 1:, :] # remove zeros prepended to every hidden output

    print('LSTM layer forward check: ', np.isclose(lstm_out.detach().numpy(), lstm_out__, atol=1e-3).all())
    print('LSTM layer forward check last hidden: ', np.isclose(h_n[0].detach().numpy(), h_n_, atol=1e-3).all())
    print('LSTM layer forward check last c_n: ', np.isclose(h_n[1].detach().numpy(), c_n_, atol=1e-3).all())

    de_dy = torch.randn(N, seq_len, hidden_dim)
    de_dy_ = de_dy.numpy()
    x.retain_grad()
    t_bckw = timer()
    lstm_out.backward(de_dy)
    t_bckw_end = timer()

    print(f'Seconds elapsed in torch backward: {t_bckw_end - t_bckw}')
    m_bckw = timer()
    dEdW_in, dEdW_hh, a, X_grad = lstm_.backward_with_memory(x_, de_dy_)
    m_bckw_end = timer()

    print(f'Seconds elapsed in my backward: {m_bckw_end - m_bckw}')

    m_bckw1 = timer()
    dEdW_in1, dEdW_hh1, a1, X_grad1 = lstm_.backward(x_, de_dy_)
    m_bckw_end1 = timer()

    print(f'Seconds elapsed in my backward memory reduction: {m_bckw_end1 - m_bckw1}')

    print('LSTM layer gradient check dEdW_in: ', np.isclose(lstm.weight_ih_l0.grad.numpy(), dEdW_in.reshape(4*hidden_dim,emb_dim), atol=1e-3).all())
    print('LSTM layer gradient check dEdW_hh: ', np.isclose(lstm.weight_hh_l0.grad.numpy(), dEdW_hh.reshape(4*hidden_dim,hidden_dim), atol=1e-3).all())
    print('LSTM layer gradient check dEdX: ', np.isclose(x.grad.numpy(), X_grad, atol=1e-3).all())


end = timer()

print(f'Seconds elapsed: {end - start}')