import numpy as np
from activations import Softmax


class CrossEntropyLoss(object):
    def __init__(self):
        self.y_pred = None

    def forward(self, y, o):
        self.y_pred = Softmax.forward(o)
        return np.sum(-y * np.log(self.y_pred + 1e-15))/(np.prod(np.array(y.shape[0:len(y.shape)-1])))

    def backward(self, y):
        return (self.y_pred - y)/(y.shape[0])
