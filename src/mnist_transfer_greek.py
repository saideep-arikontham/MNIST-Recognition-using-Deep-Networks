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


# =========================================================================================================
# Useful Function
# =========================================================================================================

def update_network(network):
    """
    Modifies the final fully connected layer of a pre-trained network
    to adapt it for 3-class classification (Greek letters).
    Freezes earlier layers to retain learned features.

    Args:
        network (nn.Module): Pre-trained MNIST network

    Returns:
        nn.Module: Modified network with new output layer
    """
    network.to("cpu")
    for param in network.parameters():
        param.requires_grad = False

    network.fc2 = nn.Linear(50, 3)  # Replace with new head
    return network

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
    # Compose the full transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))  # match MNIST
    ])
    
    # Load dataset using ImageFolder
    greek_train = ImageFolder(f"{greek_path}/{greek_train_dir}", transform=transform)
    
    greek_test = ImageFolder(f"{greek_path}/{greek_test_dir}", transform=transform)

    return greek_train, greek_test


def predict_and_plot_greek_test(network, greek_path, greek_test_dir):
    """
    Runs inference on the Greek test dataset and plots predictions alongside true labels.

    Args:
        network (nn.Module): Trained neural network
        greek_path (str): Path to the Greek data
        greek_test_dir (str): Directory name for testing data
    """
    # Compose the full transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))  # match MNIST
    ])
    
    # Load entire dataset in order, no shuffling
    greek_test_all = DataLoader(
        ImageFolder(f"{greek_path}/{greek_test_dir}", transform=transform),
        batch_size=1,
        shuffle=False
    )
    
    # Class names from folder names
    class_names = greek_test_all.dataset.classes  # ['alpha', 'beta', 'gamma']
    
    # Collect all predictions
    images_list = []
    labels_list = []
    preds_list = []
    
    network.eval()
    with torch.no_grad():
        for img, label in greek_test_all:
            output = network(img)
            pred = output.argmax(dim=1)
    
            images_list.append(img.squeeze(0))  # Remove batch dimension
            labels_list.append(label.item())
            preds_list.append(pred.item())
    
    # Plot all 15 images in a 3x5 grid
    fig, axes = plt.subplots(3, 5, figsize=(12, 7))
    for i, ax in enumerate(axes.flat):
        img = images_list[i].squeeze().numpy()
        true_label = class_names[labels_list[i]]
        pred_label = class_names[preds_list[i]]
    
        ax.imshow(img, cmap='gray')
        ax.set_title(f"True: {true_label}\nPred: {pred_label}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


# =========================================================================================================
# Main Function
# =========================================================================================================

def main(argv):
    # handle any command line arguments in argv
    if(len(argv) < 6):
        print("Usage: python mnist_transfer_greek.py <model_name> <learning_rate> <batch_size> <momentum> <epochs>")
        return
    model_name = argv[1]
    learning_rate = float(argv[2])
    batch_size = int(argv[3])
    momentum = float(argv[4])
    epochs = int(argv[5])

    #Setting up path variable for the project
    path = get_path()

    # Load saved model
    network = load_model(MyNetwork, f"{path}/models/{model_name}", set_to_eval = False)

    # Update network
    network = update_network(network)

    # Get greek training and test data
    greek_train, greek_test = get_greek_train_test_data(f"{path}/data/Greek", 'greek_train', 'greek_test')

    # Train on Greek
    network = train_network(network, greek_train, greek_test, "cpu", epochs, batch_size, learning_rate, momentum)

    # Plot predictions on test
    predict_and_plot_greek_test(network, f"{path}/data/Greek", 'greek_test')

    # Saving the network
    torch.save(network.state_dict(), f"{path}/models/greek_transferred_model.pth")
    
    return

if __name__ == "__main__":
    main(sys.argv)