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

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


# =========================================================================================================
# Class definition
# =========================================================================================================

class MyNetwork(nn.Module):
    def __init__(self):
        """
        Initializes the layers of the convolutional neural network.
        Architecture:
        - Conv1: 10 filters of size 5x5
        - MaxPool + ReLU
        - Conv2: 20 filters of size 5x5
        - Dropout
        - MaxPool + ReLU
        - Fully connected with 50 neurons + ReLU
        - Output layer with 10 neurons + log_softmax
        """
        super(MyNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.dropout = nn.Dropout(p=0.2) 
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.fc1 = nn.Linear(320, 50) 
        self.fc2 = nn.Linear(50, 10)

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        """
        computes a forward pass for the network
        """
        x = self.pool1(F.relu(self.conv1(x)))  # Conv1 -> ReLU -> MaxPool
        x = self.pool2(F.relu(self.dropout(self.conv2(x))))  # Conv2 -> Dropout -> ReLU -> MaxPool
        x = torch.flatten(x, 1)  # Flatten all dimensions except batch
        x = F.relu(self.fc1(x))  # Fully connected layer + ReLU
        x = F.log_softmax(self.fc2(x), dim=1)  # Output layer with log_softmax
        return x


# =========================================================================================================
# Useful functions
# =========================================================================================================

def get_path():
    #Setting up path variable for the project
    path = Path(os.path.dirname(os.getcwd()))
    path = str(path)
    print(path)
    sys.path.insert(1, path)
    return path


def evaluate(model, data_loader, device, criterion):
    """Helper function to evaluate model loss and accuracy"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)
    avg_loss = total_loss / len(data_loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def plot_training_metrics(train_losses, test_losses, train_accuracies, test_accuracies):
    # Plot Loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label="Train Loss")
    plt.plot(test_losses, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Testing Loss")
    plt.legend()

    # Plot Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label="Train Accuracy")
    plt.plot(test_accuracies, label="Test Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Testing Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.show()    


def train_network(model, train_data, test_data, device, epochs=5, batch_size=64, learning_rate=0.01):
    """
    Trains the model and plots train/test loss and accuracy after each epoch.
    """
    model.to(device)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)
    criterion = nn.NLLLoss()

    train_losses, test_losses = [], []
    train_accuracies, test_accuracies = [], []

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for data, target in train_loader:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += target.size(0)

        # Training stats
        train_loss = total_loss / len(train_loader)
        train_acc = 100. * correct / total

        # Evaluation on test set
        test_loss, test_acc = evaluate(model, test_loader, device, criterion)

        train_losses.append(train_loss)
        test_losses.append(test_loss)
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)

        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}%, Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}%")

    plot_training_metrics(train_losses, test_losses, train_accuracies, test_accuracies)

def get_mnist_train_test_data(path, download):
    # Download training data from open datasets.
    training_data = datasets.MNIST(
        root=f"{path}/data",
        train=True,
        download=download,
        transform=ToTensor(),
    )

    # Download test data from open datasets.
    test_data = datasets.MNIST(
        root=f"{path}/data",
        train=False,
        download=download,
        transform=ToTensor(),
    )

    return training_data, test_data


def plot_data_samples(data):

    # Plot the first 6 images
    fig, axes = plt.subplots(1, 6, figsize=(12, 2))  # 1 row, 6 columns
    
    for i in range(6):
        img, label = data[i]
        axes[i].imshow(img.squeeze(), cmap="gray")  # Remove channel dimension
        axes[i].set_title(f"Label: {label}")
        axes[i].axis("off")  # Hide axis ticks
    
    plt.tight_layout()
    plt.show()



# =========================================================================================================
# Main Function
# =========================================================================================================

def main(argv):
    # handle any command line arguments in argv

    # Get path for project directory
    path = get_path()
    
    # Get MNIST Train and test data
    train_data, test_data = get_mnist_train_test_data(path, True)

    # 1A : Plotting first 6 samples of the test data
    plot_data_samples(test_data)

    # 1B : Building the Network
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    model = MyNetwork()

    # 1C : Training the Network
    train_network(model, train_data, test_data, device, epochs=10)

    # 1D : Saving the network
    torch.save(model.state_dict(), f"{path}/models/mnist_e10_model.pth")
    
    return

if __name__ == "__main__":
    main(sys.argv)