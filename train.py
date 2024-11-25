import numpy as np
from utils.dataloader import load_mnist, DataLoader
from models.cnn import CNN
import matplotlib.pyplot as plt

# Charger les données MNIST
(train_images, train_labels), (test_images, test_labels) = load_mnist()

train_loader = DataLoader(train_images, train_labels, batch_size=32, shuffle=True)

# Prendre un lot d'images de la donnée d'entraînement
images_batch, labels_batch = next(iter(train_loader))

model = CNN()
output = model.forward(images_batch)