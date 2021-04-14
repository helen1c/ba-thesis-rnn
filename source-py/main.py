import numpy as np

from loss_functions import CrossEntropyLoss

Y_pred = np.array([[[0.1, 0.12, 0.78], [0.12, 0.80, 0.08]], [[0.12, 0.11, 0.77], [0.06, 0.09, 0.85]]])
Y = np.array([[[0., 0., 1.], [0., 1., 0.]], [[0., 0., 1.], [0., 0., 1.]]])

loss = CrossEntropyLoss.forward(Y, Y_pred)
print(loss)