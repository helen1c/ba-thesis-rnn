import numpy as np
from activations import Softmax


class CrossEntropyLoss(object):
    def __init__(self):
        self.y_pred = None

    def forward(self, y, o):
        self.y_pred = Softmax.forward(o)
        loss = (-y * np.log(self.y_pred)).sum()

        return loss

    def backward(self, y):
        return self.y_pred - y
