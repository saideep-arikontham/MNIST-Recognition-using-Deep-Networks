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

from mnist_network import MyNetwork, get_path, get_mnist_train_test_data
from mnist_results import load_model


# =========================================================================================================
# Useful Function
# =========================================================================================================

def display_layer1_weights(network):
    # Get first layer weights
    weights = network.conv1.weight.data  # Shape: [10, 1, 5, 5]
    print("conv1.weight shape:", weights.shape)
    
    # Plot 10 filters
    fig, axes = plt.subplots(3, 4, figsize=(8, 6))
    for i, ax in enumerate(axes.flat[:10]):
        kernel = weights[i, 0].cpu().numpy()  # Get 5x5 filter of ith kernel
        print(f"Filter {i+1}:\n{kernel}")
        ax.imshow(kernel, cmap='viridis')
        ax.set_title(f"Filter {i}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide any extra subplots (3x4 = 12 total)
    for j in range(10, 12):
        axes.flat[j].axis("off")
    
    plt.tight_layout()
    plt.show()


def apply_filter_to_image(model, img, label):

    img_np = img.squeeze().numpy()  # Shape: (28, 28)
    
    # ----- Get filters from conv1 -----
    with torch.no_grad():
        filters = model.conv1.weight.data.cpu().numpy()  # Shape: [10, 1, 5, 5]
    
    
    plt.imshow(img_np, cmap='gray')
    plt.title(f"Number : {label}")
    plt.axis('off')
    plt.show()
    
    # ----- Apply filters using OpenCV -----
    filtered_images = []
    for i in range(10):
        kernel = filters[i, 0]  # 5x5 kernel
        filtered = cv2.filter2D(img_np, -1, kernel)
        filtered_images.append(filtered)
    
    # ----- Plot the 10 filtered outputs -----
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        ax.imshow(filtered_images[i], cmap='gray')
        ax.set_title(f"Filter {i}")
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.show()

# =========================================================================================================
# Main Function
# =========================================================================================================

def main(argv):
    # handle any command line arguments in argv

    #Setting up path variable for the project
    path = get_path()

    # Load saved model
    network = load_model(f"{path}/models/mnist_e10_model.pth", set_to_eval = True)

    # Print & Visualize Layer 1 filters
    display_layer1_weights(network)

    # Apply filter to first train data image
    train_data, test_data = get_mnist_train_test_data(path, download=False)
    img, label = train_data[0]  # First image
    apply_filter_to_image(network, img, label)    
    
    return

if __name__ == "__main__":
    main(sys.argv)