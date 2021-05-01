import numpy as np
from activations import Softmax


class CrossEntropyLoss(object):

    def forward(self, y, o):
        loss = 0.
        self.y_pred = Softmax.forward(o)
        for i in range(y.shape[0]):
            loss += (-y * np.log(self.y_pred)).sum()

        return loss

    def backward(self, y):
        return self.y_pred - y
