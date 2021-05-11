import numpy as np


class SGD(object):
    def __init__(self, learning_rate, momentum=0):
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.last_momentum = []

    def update_parameters(self, model_parameters, gradients):
        for i in range(len(model_parameters)):
            if len(self.last_momentum) < i + 1:
                self.last_momentum.append(np.zeros_like(model_parameters[i]))

            new_momentum = self.momentum * self.last_momentum[i] + (1 - self.momentum) * gradients[i]
            model_parameters[i] -= self.learning_rate * new_momentum
            self.last_momentum[i] = new_momentum


class RMSProp(object):
    def __init__(self, learning_rate=0.001, gamma=0.9, eta=1e-8):
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.last_momentum = []
        self.eta = eta

    def update_parameters(self, model_parameters, gradients):
        for i in range(len(model_parameters)):
            if len(self.last_momentum) < i + 1:
                self.last_momentum.append(np.zeros_like(model_parameters[i]))

            new_momentum = self.gamma * self.last_momentum[i] + (1 - self.gamma) * (gradients[i] ** 2)
            model_parameters[i] -= self.learning_rate * (gradients[i] / (np.sqrt(new_momentum) + np.full_like(new_momentum, self.eta, dtype=np.double)))
            self.last_momentum[i] = new_momentum


class Adam(object):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, eta=1e-8):
        self.learning_rate = learning_rate

        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2

        self.last_momentum = []
        self.last_rms = []

        self.iteration_counter = 0

    def update_parameters(self, model_parameters, gradients):
        self.iteration_counter += 1
        for i in range(len(model_parameters)):

            if len(self.last_momentum) < i + 1:
                self.last_momentum.append(np.zeros_like(model_parameters[i]))
            if len(self.last_rms) < i + 1:
                self.last_rms.append(np.zeros_like(model_parameters[i]))

            new_momentum = self.beta1 * self.last_momentum[i] + (1 - self.beta1) * gradients[i]
            new_rms = self.beta2 * self.last_rms[i] + (1 - self.beta2) * (gradients[i] ** 2)

            #new_momentum = np.divide(1, 1 - np.power(self.beta1, self.iteration_counter)) * new_momentum
            #new_rms = np.divide(1, 1 - np.power(self.beta2, self.iteration_counter)) * new_rms

            model_parameters[i] -= ((self.learning_rate * new_momentum) / (np.sqrt(new_rms) + np.full_like(new_rms, self.eta, dtype=np.double)))
            self.last_momentum[i] = new_momentum
            self.last_rms[i] = new_rms
