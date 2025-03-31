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
from itertools import product
import random
import time
import pandas as pd

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

from mnist_network import MyNetwork, get_path
from mnist_results import load_model

# =========================================================================================================
# Class definition
# =========================================================================================================

class FlexibleCNN(nn.Module):
    """
    A flexible CNN architecture for FashionMNIST classification.
    
    Args:
        conv_layers (int): Number of convolutional layers.
        linear_layers (int): Number of linear (fully connected) layers.
        dropout_rate (float): Dropout rate applied after flattening.
        hidden_units (int): Number of hidden units in the linear layers.
    """
    
    def __init__(self, conv_layers, linear_layers, dropout_rate, hidden_units):
        super().__init__()
        self.activation = F.relu
        self.convs = nn.Sequential()
        in_channels = 1
        H, W = 28, 28

        for i in range(conv_layers):
            out_channels = 32 * (i + 1)
            self.convs.add_module(f'conv{i}', nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            self.convs.add_module(f'relu{i}', nn.ReLU())
            if H >= 2 and W >= 2:
                self.convs.add_module(f'pool{i}', nn.MaxPool2d(2))
                H, W = H // 2, W // 2
            in_channels = out_channels

        dummy = torch.zeros(1, 1, 28, 28)
        with torch.no_grad():
            x = self.convs(dummy)
            self.flat_dim = x.view(1, -1).size(1)

        self.dropout = nn.Dropout(dropout_rate)
        self.linear_layers = nn.Sequential()
        if linear_layers > 0:
            # First linear layer: from flattened CNN output to hidden units
            self.linear_layers.add_module("fc1", nn.Linear(self.flat_dim, hidden_units))
            self.linear_layers.add_module("relu1", nn.ReLU())
            # Additional linear layers if any
            for i in range(1, linear_layers):
                self.linear_layers.add_module(f"fc{i+1}", nn.Linear(hidden_units, hidden_units))
                self.linear_layers.add_module(f"relu{i+1}", nn.ReLU())
            final_in_features = hidden_units
        else:
            # If no linear layers, pass the flattened features directly
            final_in_features = self.flat_dim

        self.final_layer = nn.Linear(final_in_features, 10)

    def forward(self, x):
        x = self.convs(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.linear_layers(x)
        x = self.activation(x)
        return F.log_softmax(self.final_layer(x), dim=1)


# =========================================================================================================
# Useful Function
# =========================================================================================================

def get_fashion_mnist_train_test_data(path, download):
    """
    Loads and returns the FashionMNIST train and test datasets with normalization.

    Args:
        path (str): Path to download/load the data.
        download (bool): Whether to download the dataset.

    Returns:
        Tuple[Dataset, Dataset]: Train and test datasets.
    """
    # Load FashionMNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.FashionMNIST(root=f'{path}/data', train=True, download=download, transform=transform)
    test_data = datasets.FashionMNIST(root=f'{path}/data', train=False, download=download, transform=transform)

    return train_data, test_data


def get_param_grid(max_conv_layers, max_lin_layers, max_dropout, max_hidden_units, max_batch_size, learning_rate, max_epochs):
    """
    Generates a randomized grid of hyperparameters for CNN training.

    Args:
        max_conv_layers (int): Maximum number of convolutional layers.
        max_lin_layers (int): Maximum number of linear layers.
        max_dropout (float): Maximum dropout rate.
        max_hidden_units (int): Maximum number of hidden units.
        max_batch_size (int): Maximum batch size.
        learning_rate (List[float]): List of learning rates to test.
        max_epochs (int): Maximum number of epochs.

    Returns:
        List[Tuple]: A shuffled list of hyperparameter combinations.
    """
    conv_layers_list = list(range(1, max_conv_layers + 1))
    linear_layers_list = list(range(1, max_lin_layers + 1))
    dropouts = np.arange(0.1, max_dropout + 0.1, 0.1).tolist()
    hidden_units_list = list(range(64, max_hidden_units + 1, 64))
    batch_sizes = list(range(32, max_batch_size + 1, 32))
    epochs_list = list(range(5, max_epochs + 1, 5))

    print("Parameter grid:")
    print("Conv layers:", conv_layers_list)
    print("Linear layers:", linear_layers_list)
    print("Dropouts:", dropouts)
    print("Hidden units:", hidden_units_list)
    print("Batch sizes:", batch_sizes)
    print("Learning rates:", learning_rate)
    print("Epochs:", epochs_list)

    grid = list(product(conv_layers_list, linear_layers_list, dropouts, hidden_units_list, batch_sizes, learning_rate, epochs_list))
    print("Total combinations:", len(grid))
    
    random.shuffle(grid)
    return grid[:75]


def train_and_evaluate(config, path):
    """
    Trains and evaluates a CNN with a given hyperparameter configuration.

    Args:
        config (tuple): A tuple of hyperparameters.
        path (str): Path to data.

    Returns:
        dict: Dictionary containing model configuration, test accuracy, and training time.
    """
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    
    conv_layers, linear_layers, dropout_rate, hidden_units, batch_size, learning_rate, epochs = config

    train_data, test_data = get_fashion_mnist_train_test_data(path, False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    model = FlexibleCNN(conv_layers, linear_layers, dropout_rate, hidden_units).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    start_time = time.time()
    model.train()
    for epoch in range(epochs):
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            optimizer.step()
    duration = time.time() - start_time

    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)

    acc = 100 * correct / total
    return {
        'conv_layers': conv_layers,
        'linear_layers': linear_layers,
        'dropout_rate': dropout_rate,
        'hidden_units': hidden_units,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'epochs': epochs,
        'test_acc': acc,
        'time_sec': round(duration, 2)
    }


def run_train_loop_for_all_configs(path, grid):
    """
    Runs training and evaluation for a list of hyperparameter configurations.

    Args:
        path (str): Path to dataset.
        grid (List[Tuple]): List of hyperparameter configurations.

    Returns:
        pd.DataFrame: Sorted DataFrame of results by test accuracy.
    """
    results = []

    # Run experiments
    for config in grid:
        print("Running config:", config)
        results.append(train_and_evaluate(config, path))

    return pd.DataFrame(results).sort_values(by='test_acc', ascending=False)

# =========================================================================================================
# Main Function
# =========================================================================================================

def main(argv):
    # handle any command line arguments in argv
    if(len(argv) < 7):
        print("Usage: python fashion_mnist_network.py <max_conv_layers> <max_lin_layers> <max_dropout> <max_hidden_units> <max_batch_size> <max_epochs>")
        return
    max_conv_layers = int(argv[1])
    max_lin_layers = int(argv[2])
    max_dropout = float(argv[3])
    max_hidden_units = int(argv[4])
    max_batch_size = int(argv[5])
    max_epochs = int(argv[6])
    learning_rate = [0.01, 0.001, 0.0001]


    #Setting up path variable for the project
    path = get_path()

    #Download data
    train_data, test_data = get_fashion_mnist_train_test_data(path, True)

    #Setup NN configuration grid
    grid = get_param_grid(max_conv_layers, max_lin_layers, max_dropout, max_hidden_units, max_batch_size, learning_rate, max_epochs)

    # Run Training loop with different configurations
    result_df = run_train_loop_for_all_configs(path, grid)
    print(result_df)
    
    return

if __name__ == "__main__":
    main(sys.argv)