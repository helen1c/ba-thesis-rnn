from LSTMLayer import LSTMLayer
import numpy as np

lstm = LSTMLayer(2,1)

lstm.input_weights = np.array([[[0.95, 0.8]],

       [[0.7, 0.45]],

       [[0.45, 0.25]],

       [[0.6, 0.4]]])

lstm.hidden_weights = np.array([[[0.8]],

       [[ 0.1]],

       [[0.15]],

       [[0.25]]])

lstm.bias = np.array([[0.65], [0.15],  [0.2],  [0.1]])
x = np.array([[[1,2],[0.5,3]]])
dedy = np.array([[[0.03631],[-0.47803]]])
lstm.forward(x)
lstm.backward(x, dedy)