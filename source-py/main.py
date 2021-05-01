import numpy as np

from loss_functions import CrossEntropyLoss
from activations import Softmax

Y_pred = np.array([[[4, 3, 1], [2, 7, 2]], [[1, 12, 0.77], [0.06, 0.09, 0.85]]])
Y = np.array([[[0., 0., 1.], [0., 1., 0.]], [[0., 0., 1.], [0., 0., 1.]]])

print(Softmax.forward(Y_pred))
