import numpy as np
from models.layers import Convolutional, Fc, avg_pooling, Relu

class CNN:
    def __init__(self):
        # Initialisation des couches
        self.conv1 = Convolutional((32, 1, 32, 32), 5, 6, stride=1, padding=0)
        self.pool1 = avg_pooling()
        self.relu1 = Relu()
        self.conv2 = Convolutional((32, 1, 10, 10), 5, 16, stride=1, padding=0)
        self.pool2 = avg_pooling()
        self.relu2 = Relu()
        self.conv3 = Convolutional((32, 1, 5, 5), 5, 120, stride=1, padding=0)
        self.Fc1 = Fc(120, 84)
        self.Fc2 = Fc(84, 10)

    def forward(self, X):
        # Passer les données à travers les couches
        X = self.conv1.forward(X)
        X = self.pool1.forward(X, 2, 2)
        X = self.relu1.forward(X)
        X = self.conv2.forward(X)
        X = self.pool2.forward(X, 1, 2)
        X = self.relu2.forward(X)
        X = self.conv3.forward(X)
        X = X.reshape(32, -1)
        X = self.Fc1.forward(X)
        X = self.Fc2.forward(X)

        return X

    def backward(self, loss_gradient):
        # Calcul des gradients
        pass

    def update_weights(self, learning_rate):
        # Mise à jour des poids
        pass