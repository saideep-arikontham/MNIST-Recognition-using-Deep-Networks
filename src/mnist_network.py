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
    """
    Sets up and returns the path to the parent directory of the current working directory.
    Also inserts the path into sys.path for module imports.
    """
    path = Path(os.path.dirname(os.getcwd()))
    path = str(path)
    print(path)
    sys.path.insert(1, path)
    return path


def evaluate(model, data_loader, device, criterion):
    """
    Evaluates the given model's performance on a dataset.
    
    Args:
        model (torch.nn.Module): The trained model to evaluate.
        data_loader (DataLoader): DataLoader for the evaluation dataset.
        device (torch.device): The device to run the evaluation on (CPU or GPU).
        criterion (Loss function): The loss function to use for evaluation.

    Returns:
        Tuple of (average loss, accuracy percentage).
    """
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


def plot_loss_curve(train_losses, test_losses):
    """
    Plots training loss and test loss on the same graph.

    Args:
        train_losses (list of tuples): List of (examples_seen, loss) for training.
        test_losses (list of tuples): List of (examples_seen, loss) for testing.
    """
    train_x, train_y = zip(*train_losses)
    test_x, test_y = zip(*test_losses)

    plt.figure()
    plt.plot(train_x, train_y, label='Train loss', color='blue')
    plt.scatter(test_x, test_y, label='Test loss', color='red', zorder=5)
    plt.xlabel("number of training examples seen")
    plt.ylabel("negative log likelihood loss")
    plt.legend()
    plt.show()

def plot_accuracy_curve(train_accuracies, test_accuracies):
    """
    Plots training and test accuracies over epochs.
    
    Args:
        train_accuracies (list of tuples): Each tuple contains (epoch, training accuracy).
        test_accuracies (list of tuples): Each tuple contains (epoch, test accuracy).
    """
    train_epochs, train_acc = zip(*train_accuracies)
    test_epochs, test_acc = zip(*test_accuracies)
    
    plt.figure()
    plt.plot(train_epochs, train_acc, label='Train Accuracy', color='green')
    plt.plot(test_epochs, test_acc, label='Test Accuracy', color='orange')
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.legend()
    plt.show()


def train_network(model, train_data, test_data, device, epochs=5, batch_size=64, learning_rate=0.01, momentum=0.9):
    """
    Trains the neural network and evaluates it after each epoch.

    Args:
        model (torch.nn.Module): The model to train.
        train_data (Dataset): The training dataset.
        test_data (Dataset): The testing dataset.
        device (torch.device): Device to use for training (CPU/GPU).
        epochs (int): Number of training epochs.
        batch_size (int): Size of each training batch.
        learning_rate (float): Learning rate for the optimizer.
    """
    model.to(device)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum)
    criterion = nn.NLLLoss()

    train_losses = []
    test_losses = []
    test_checkpoints = []
    train_accuracies = []
    test_accuracies = []
    
    best_loss = float('inf')
    best_state = None
    
    examples_seen = 0
    for epoch in range(epochs):
        train_correct = 0
        train_total = 0
        
        model.train()
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            pred = output.argmax(dim=1)
            train_correct += (pred == target).sum().item()
            train_total += target.size(0)

            if batch_idx % 50 == 0:
                train_losses.append((examples_seen, loss.item()))
            examples_seen += len(data)

        test_loss, test_accuracy = evaluate(model, test_loader, device, criterion)
        test_losses.append((examples_seen, test_loss))
        test_accuracies.append((epoch+1, test_accuracy))
        
        if test_loss < best_loss:
            best_loss = test_loss
            best_state = model.state_dict()

        print(f"Epoch {epoch+1} [{examples_seen} examples] Test Loss: {test_loss:.4f} Test Accuracy: {test_accuracy:.2f}%")

        train_accuracy = 100. * train_correct / train_total
        train_accuracies.append((epoch+1, train_accuracy))
        print(f"Epoch {epoch+1} Training Accuracy: {train_accuracy:.2f}%")

    plot_loss_curve(train_losses, test_losses)
    plot_accuracy_curve(train_accuracies, test_accuracies)

    # Load best model weights based on test loss
    if best_state is not None:
        model.load_state_dict(best_state)
        print("Loaded best model weights with test loss: {:.4f}".format(best_loss))

    return model

def get_mnist_train_test_data(path, download):
    """
    Loads the MNIST training and testing datasets.

    Args:
        path (str): Path to the directory where data should be stored.
        download (bool): Whether to download the dataset if not already present.

    Returns:
        Tuple[Dataset, Dataset]: Training and testing datasets.
    """
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
    """
    Plots the first 6 images and their labels from the given dataset.

    Args:
        data (Dataset): Dataset containing image-label pairs.
    """
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
    if(len(argv) < 4):
        print("Usage: python mnist_network.py <epochs> <batch_size> <learning_rate> <momentum>")
        return
    epochs = int(argv[1])
    batch_size = int(argv[2])
    learning_rate = float(argv[3])
    momentum = float(argv[4])
    print(f"Epochs: {epochs}, Batch Size: {batch_size}, Learning Rate: {learning_rate}, Momentum: {momentum}")

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
    model = train_network(model, train_data, test_data, device, epochs=epochs, batch_size=batch_size, learning_rate=learning_rate, momentum=momentum)

    print(model)
    # 1D : Saving the network
    torch.save(model.state_dict(), f"{path}/models/mnist_model.pth")
    
    return

if __name__ == "__main__":
    main(sys.argv)