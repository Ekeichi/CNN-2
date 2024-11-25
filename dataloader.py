import numpy as np
import torchvision
from torchvision import transforms

def load_mnist(data_dir="./data"):
    """
    Télécharge le dataset MNIST et le retourne sous forme de tableaux NumPy.
    
    Args:
        data_dir (str): Dossier où télécharger les données.

    Returns:
        (tuple): (train_images, train_labels), (test_images, test_labels)
    """
    # Transformation pour convertir les images en tenseurs puis en tableaux NumPy
    transform = transforms.Compose([
        transforms.ToTensor(),  # Convertit en tenseur PyTorch (C, H, W) normalisé entre [0, 1]
        transforms.Pad(padding=2, fill=0),
        transforms.Lambda(lambda x: x.numpy())  # Convertit en tableau NumPy
    ])
    
    # Télécharger les données
    train_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=True,
        transform=transform,
        download=True
    )
    
    test_dataset = torchvision.datasets.MNIST(
        root=data_dir,
        train=False,
        transform=transform,
        download=True
    )
    
    # Extraire les images et les labels
    train_images = np.array([data[0] for data in train_dataset])  # Images d'entraînement
    train_labels = np.array([data[1] for data in train_dataset])  # Labels d'entraînement
    
    test_images = np.array([data[0] for data in test_dataset])  # Images de test
    test_labels = np.array([data[1] for data in test_dataset])  # Labels de test
    
    return (train_images, train_labels), (test_images, test_labels)

class DataLoader:
    def __init__(self, data, labels, batch_size=32, shuffle=True):
        """
        Initialisation du DataLoader.
        
        :param data: numpy array des données (e.g., images de taille (N, H, W))
        :param labels: numpy array des labels (e.g., (N,))
        :param batch_size: taille des batchs
        :param shuffle: bool, si les données doivent être mélangées à chaque epoch
        """
        self.data = data
        self.labels = labels
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(data))
        self.current_index = 0
        
        if self.shuffle:
            np.random.shuffle(self.indices)
    
    def __len__(self):
        """Retourne le nombre total de batchs."""
        return int(np.ceil(len(self.data) / self.batch_size))
    
    def __iter__(self):
        """Prépare un nouvel itérateur pour parcourir les données."""
        self.current_index = 0
        if self.shuffle:
            np.random.shuffle(self.indices)
        return self
    
    def __next__(self):
        """Renvoie le prochain batch."""
        if self.current_index >= len(self.data):
            raise StopIteration
        
        start = self.current_index
        end = start + self.batch_size
        batch_indices = self.indices[start:end]
        self.current_index = end
        
        return self.data[batch_indices], self.labels[batch_indices]
    