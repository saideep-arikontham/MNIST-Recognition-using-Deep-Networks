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

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

from mnist_network import MyNetwork, get_path, plot_training_metrics
from mnist_results import load_model

# =========================================================================================================
# Class definition
# =========================================================================================================

class FlexibleCNN(nn.Module):
    def __init__(self, conv_layers, linear_layers, dropout_rate, hidden_units):
        super().__init__()
        self.activation = F.relu
        self.convs = nn.Sequential()
        in_channels = 1
        H, W = 28, 28

        for i in range(conv_layers):
            out_channels = 16 * (i + 1)
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

        if linear_layers == 1:
            self.linear_layers.add_module("fc1", nn.Linear(self.flat_dim, hidden_units))
        elif linear_layers == 2:
            self.linear_layers.add_module("fc1", nn.Linear(self.flat_dim, hidden_units))
            self.linear_layers.add_module("relu1", nn.ReLU())
            self.linear_layers.add_module("fc2", nn.Linear(hidden_units, hidden_units))

        self.final_layer = nn.Linear(hidden_units, 10)

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
    # Load FashionMNIST
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_data = datasets.FashionMNIST(root=f'{path}/data', train=True, download=download, transform=transform)
    test_data = datasets.FashionMNIST(root=f'{path}/data', train=False, download=download, transform=transform)

    return train_data, test_data


def get_param_grid():
    conv_layers_list = [2, 3, 4, 5]
    linear_layers_list = [1, 2]
    dropouts = [0.1, 0.3, 0.5]
    hidden_units_list = [64, 128]
    batch_sizes = [32, 64]
    
    grid = list(product(conv_layers_list, linear_layers_list, dropouts, hidden_units_list, batch_sizes))
    random.shuffle(grid)
    return grid[:30]


def train_and_evaluate(config, path):
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    
    conv_layers, linear_layers, dropout_rate, hidden_units, batch_size = config

    train_data, test_data = get_fashion_mnist_train_test_data(path, False)

    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64)

    model = FlexibleCNN(conv_layers, linear_layers, dropout_rate, hidden_units).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.NLLLoss()

    start_time = time.time()
    model.train()
    for epoch in range(3):
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
        'test_acc': acc,
        'time_sec': round(duration, 2)
    }


def run_train_loop_for_all_configs(path, grid):
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

    #Setting up path variable for the project
    path = get_path()

    #Download data
    train_data, test_data = get_fashion_mnist_train_test_data(path, True)

    #Setup NN configuration grid
    grid = get_param_grid()

    # Run Training loop with different configurations
    result_df = run_train_loop_for_all_configs(path, grid)
    
    return

if __name__ == "__main__":
    main(sys.argv)