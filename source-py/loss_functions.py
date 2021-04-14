import numpy as np


class CrossEntropyLoss(object):

    @staticmethod
    def forward(y, y_pred):
        # prvo za jedan primjer
        # ima li svaki podatak iz batcha svoj loss pa se zbraja, ili ce rezultat biti vektor velicina batch_size?
        loss = 0.
        for i in range(y.shape[0]):
            loss += (-y * np.log(y_pred)).sum()

        return loss
