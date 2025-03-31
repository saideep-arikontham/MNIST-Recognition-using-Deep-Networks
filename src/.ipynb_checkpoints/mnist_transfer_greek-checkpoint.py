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

from mnist_network import MyNetwork, get_path, plot_training_metrics
from mnist_results import load_model

# =========================================================================================================
# Class definition
# =========================================================================================================

class GreekTransform:
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
    network.to("cpu")
    for param in network.parameters():
        param.requires_grad = False

    network.fc2 = nn.Linear(50, 3)  # Replace with new head
    return network

def get_greek_train_test_data(greek_path, greek_train_dir, greek_test_dir):
    
    # Compose the full transform
    transform = transforms.Compose([
        transforms.ToTensor(),
        GreekTransform(),
        transforms.Normalize((0.1307,), (0.3081,))  # match MNIST
    ])
    
    # Load dataset using ImageFolder
    greek_train = DataLoader(
        ImageFolder(f"{greek_path}/{greek_train_dir}", transform=transform),
        batch_size=5,
        shuffle=True
    )
    
    greek_test = DataLoader(
        ImageFolder(f"{greek_path}/{greek_test_dir}", transform=transform),
        batch_size=5,
        shuffle=True
    )

    return greek_train, greek_test


def transfer_train_greek(network, greek_train, greek_test, lr, momentum, epochs):
    
    optimizer = optim.SGD(network.fc2.parameters(), lr=lr, momentum=momentum)
    criterion = nn.CrossEntropyLoss()

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []
    
    for epoch in range(epochs):
        network.train()
        total_loss = 0
        correct = 0
        total = 0
    
        for data, target in greek_train:
            optimizer.zero_grad()
            output = network(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    
        train_acc = 100. * correct / total
    
        # --- Evaluate on test set ---
        network.eval()
        test_loss = 0
        test_correct = 0
        test_total = 0
    
        with torch.no_grad():
            for data, target in greek_test:
                output = network(data)
                loss = criterion(output, target)
    
                test_loss += loss.item()
                pred = output.argmax(dim=1)
                test_correct += (pred == target).sum().item()
                test_total += target.size(0)
    
        test_acc = 100. * test_correct / test_total
        test_loss_avg = test_loss / len(greek_test)

        train_losses.append(total_loss)
        test_losses.append(test_loss_avg)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        
        print(f"Epoch {epoch+1}")
        print(f"  Train Loss: {total_loss:.4f}, Train Accuracy: {train_acc:.2f}%")
        print(f"  Test  Loss: {test_loss_avg:.4f}, Test  Accuracy: {test_acc:.2f}%\n")

    plot_training_metrics(train_losses, test_losses, train_accuracies, test_accuracies)
    
    network.train()
    return network


def predict_and_plot_greek_test(network, greek_path, greek_test_dir):

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

    #Setting up path variable for the project
    path = get_path()

    # Load saved model
    network = load_model(f"{path}/models/mnist_e10_model.pth", set_to_eval = False)

    # Update network
    network = update_network(network)

    # Get greek training and test data
    greek_train, greek_test = get_greek_train_test_data(f"{path}/data/Greek", 'greek_train', 'greek_test')

    # Train on Greek
    network = transfer_train_greek(network, greek_train, greek_test, 0.0001, 0.9, 15)

    # Plot predictions on test
    predict_and_plot_greek_test(network, f"{path}/data/Greek", 'greek_test')
    
    return

if __name__ == "__main__":
    main(sys.argv)