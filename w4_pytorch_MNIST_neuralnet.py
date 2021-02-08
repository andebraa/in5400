import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from utils.utility_functions import datasetFashionMNIST
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn import datasets
digits = datasets.load_digits() #mnist dataset


#Path to the MNIST Fashion files
dataPath = 'data/MNIST_fashion/'
#dataPath = '/projects/in5400/MNIST_fashion/'

# Create dataset objects
train_dataset = datasetFashionMNIST(dataPath=dataPath, train=True)
val_dataset   = datasetFashionMNIST(dataPath=dataPath, train=False)
print((val_dataset[0][0]))


# Visualize some examples from the dataset.
# We show a few examples of training images from each class.
"""
classes = ['T-shirt / top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
num_classes = len(classes)
samples_per_class = 7
plt.figure(figsize=(18, 16), dpi=80)
for y, cls in enumerate(classes):
    idxs = np.flatnonzero(np.array(train_dataset.labels) == y)
    idxs = np.random.choice(idxs, samples_per_class, replace=False)
    for i, idx in enumerate(idxs):
        plt_idx = i * num_classes + y + 1
        plt.subplot(samples_per_class, num_classes, plt_idx)
        img = (train_dataset.images[idx,:]).astype(np.uint8)
        img = np.resize(img, (28, 28))   # reshape to 28x28
        plt.imshow(img, cmap='gray')
        plt.axis('off')
        if i == 0:
            plt.title(cls)
plt.show()
"""

config = {
          'batch_size': 128,
          'use_cuda': False,      #True=use Nvidia GPU | False use CPU
          'log_interval': 20,     #How often to display (batch) loss during training
          'epochs': 20,           #Number of epochs
          'learningRate': 0.001
         }

# DataLoaders

train_loader = tud.DataLoader(train_dataset, batch_size = config['batch_size'], shuffle = True)
val_loader = tud.DataLoader(val_dataset, batch_size = config['batch_size'], shuffle = False)

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
