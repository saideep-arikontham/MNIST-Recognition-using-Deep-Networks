# =========================================================================================================
# Saideep Arikontham
# March 2025
# CS 5330 Project 5
# =========================================================================================================



# =========================================================================================================
# Import statements
# =========================================================================================================

import sys
import os
import sys
from pathlib import Path
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

from mnist_network import MyNetwork, get_path, evaluate, plot_loss_curve, train_network
from mnist_transfer_greek import get_greek_train_test_data, predict_and_plot_greek_test
from mnist_results import load_model

# =========================================================================================================
# Class definition
# =========================================================================================================

class GreekTransform:
    """
    Custom transform for Greek letter images to match MNIST format:
    - Converts RGB to grayscale
    - Rescales to 28x28 from 128x128 using affine transform
    - Inverts pixel values to match MNIST's white-on-black format
    """
    def __init__(self):
        pass

    def __call__(self, x):
        x = transforms.functional.rgb_to_grayscale(x)
        x = transforms.functional.affine(x, angle=0, translate=(0, 0), scale=36/128, shear=0)
        x = transforms.functional.center_crop(x, (28, 28))
        return transforms.functional.invert(x)
    

class GreekNetwork(nn.Module):
    def __init__(self):
        super(GreekNetwork, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, 3)
        self.conv2 = nn.Conv2d(10, 20, 3)
        self.conv3 = nn.Conv2d(20, 40, 3)
        self.dropout = nn.Dropout(0.2)
        self.pool = nn.MaxPool2d(2)

        # We’ll figure out the correct flatten size using dummy input
        self._to_linear = None
        self._get_flattened_size()

        self.fc1 = nn.Linear(self._to_linear, 50)
        self.fc3 = nn.Linear(50, 3)

    def _get_flattened_size(self):
        with torch.no_grad():
            x = torch.zeros(1, 1, 28, 28)  # 1 dummy grayscale image
            x = F.relu(self.conv1(x))
            x = self.pool(F.relu(self.dropout(self.conv2(x))))
            x = self.pool(F.relu(self.dropout(self.conv3(x))))
            self._to_linear = x.view(1, -1).shape[1]

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.dropout(self.conv2(x))))
        x = self.pool(F.relu(self.dropout(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x



# =========================================================================================================
# Useful Function
# =========================================================================================================


def get_greek_train_test_data(greek_path, greek_train_dir, greek_test_dir):
    """
    Loads Greek letter image data from the specified directory and applies transformations
    to prepare it for training and testing.

    Args:
        greek_path (str): Base path to the Greek data
        greek_train_dir (str): Directory name for training data
        greek_test_dir (str): Directory name for testing data

    Returns:
        Tuple[DataLoader, DataLoader]: Training and testing dataloaders
    """
    # Define data augmentation for training and a standard transform for testing
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))  # match MNIST
    ])
    
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))  # match MNIST
    ])
    
    # Load dataset using ImageFolder with the respective transforms
    greek_train = ImageFolder(f"{greek_path}/{greek_train_dir}", transform=train_transform)
    greek_test = ImageFolder(f"{greek_path}/{greek_test_dir}", transform=test_transform)

    return greek_train, greek_test


# =========================================================================================================
# Main Function
# =========================================================================================================

def main(argv):
    # handle any command line arguments in argv
    if(len(argv) < 5):
        print("Usage: python greek_network_extension1.py <learning_rate> <batch_size> <momentum> <epochs>")
        return
    learning_rate = float(argv[1])
    batch_size = int(argv[2])
    momentum = float(argv[3])
    epochs = int(argv[4])

    #Setting up path variable for the project
    path = get_path()

    # Load saved model
    network = GreekNetwork()

    # Get greek training and test data
    greek_train, greek_test = get_greek_train_test_data(f"{path}/data/Greek", 'greek_train', 'greek_test')

    # Train on Greek
    network = train_network(network, greek_train, greek_test, "cpu", epochs, batch_size, learning_rate, momentum)

    # Plot predictions on test
    predict_and_plot_greek_test(network, f"{path}/data/Greek", 'greek_test')
    
    torch.save(network.state_dict(), f"{path}/models/greek_model.pth")
    return

if __name__ == "__main__":
    main(sys.argv)