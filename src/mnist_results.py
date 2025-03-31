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


# =========================================================================================================
# Useful Function
# =========================================================================================================


def load_model(ModelNetwork, model_weights_path, set_to_eval):
    """
    Loads a trained PyTorch model from the given path.

    Args:
        model_weights_path (str): Path to the saved model weights (.pth file).
        set_to_eval (bool): Whether to set the model to evaluation mode.

    Returns:
        nn.Module: The loaded PyTorch model.
    """
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    
    # Load the model
    network = ModelNetwork()
    network.load_state_dict(torch.load(model_weights_path))
    network.to(device)

    if(set_to_eval):
        network.eval()
    return network


def predict_on_mnist(data, network, print_output):
    """
    Runs inference on a batch of MNIST test data and optionally prints predictions.

    Args:
        data (Dataset): The MNIST dataset to run predictions on.
        network (nn.Module): The trained neural network model.
        print_output (bool): Whether to print predictions and probabilities.

    Returns:
        Tuple[Tensor, Tensor]: The input images and the model's raw outputs.
    """
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    loader = DataLoader(data, batch_size=10, shuffle=False)
    
    # --- Get test samples ---
    images, labels = next(iter(loader))
    images, labels = images.to(device), labels.to(device)
    
    with torch.no_grad():
        outputs = network(images)    

    if(print_output):
        # --- Process and Print Output ---
        for i in range(10):
            output = outputs[i]
            probs = torch.exp(output)  # Convert from log_softmax to probabilities
            values = [f"{v:.2f}" for v in probs.tolist()]
            predicted = output.argmax().item()
            true_label = labels[i].item()
            
            print(f"Image {i+1}:")
            print(f"  Output values: {values}")
            print(f"  Predicted: {predicted}, True Label: {true_label}\n")

    return images, outputs

def plot_outputs(outputs, images, plot_rows, plot_cols):
    """
    Plots a grid of images with their predicted labels.

    Args:
        outputs (Tensor): Model predictions for the images.
        images (Tensor): Input images corresponding to the predictions.
        plot_rows (int): Number of rows in the plot grid.
        plot_cols (int): Number of columns in the plot grid.
    """
    # --- Plot first 9 images in a 3x3 grid with predictions ---
    fig, axes = plt.subplots(plot_rows, plot_cols, figsize=(6, 6))
    for i, ax in enumerate(axes.flat):
        ax.imshow(images[i].cpu().squeeze(), cmap='gray')
        pred = outputs[i].argmax().item()
        ax.set_title(f"Prediction: {pred}")
        ax.axis('off')
    
    plt.tight_layout()
    plt.show()


def predict_and_plot_handwritten_digits(network, image_dir):
    """
    Loads custom handwritten digit images from a directory, preprocesses them,
    and displays the model's predictions.

    Args:
        network (nn.Module): Trained PyTorch model.
        image_dir (str): Directory containing custom handwritten digit images.
    """
    device = torch.device("mps" if torch.mps.is_available() else "cpu")
    # --- Define preprocessing ---
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Grayscale(),                # Ensure grayscale
        transforms.Resize((28, 28)),           # Resize to 28x28
        transforms.ToTensor(),                 # Convert to tensor
        transforms.Normalize((0.1307,), (0.3081,))  # Match MNIST normalization
    ])
    
    # --- Load handwritten images ---
    image_files = sorted(os.listdir(image_dir))  # Ensure they're in order: 0.png, 1.png, ...
    custom_images = []
    labels = []
    
    for file in image_files:
        if(file.endswith(".png")):
            img_path = os.path.join(image_dir, file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Read grayscale
            img = cv2.bitwise_not(img)                    # Invert: MNIST is white-on-black
            img = cv2.resize(img, (28, 28))               # Resize if needed
            tensor_img = transform(img)                   # Apply same MNIST transforms
            custom_images.append(tensor_img.unsqueeze(0))  # Add batch dimension
            labels.append(file.split('.')[0])             # Use file name as label (optional)
    
    # --- Run through model and plot ---
    fig, axes = plt.subplots(2, 5, figsize=(10, 4))
    for i, ax in enumerate(axes.flat):
        img = custom_images[i].to(device)
        with torch.no_grad():
            output = network(img)
            pred = output.argmax(dim=1).item()
        
        ax.imshow(custom_images[i].squeeze().cpu(), cmap="gray")
        ax.set_title(f"Prediction: {pred}")
        ax.axis("off")
    
    plt.tight_layout()
    plt.show()

# =========================================================================================================
# Main Function
# =========================================================================================================

def main(argv):
    # handle any command line arguments in argv
    if(len(argv) < 2):
        print("Usage: python mnist_results.py <model_name>")
        return
    model_name = argv[1]

    #Setting up path variable for the project
    path = get_path()

    # Load saved model
    network = load_model(MyNetwork, f"{path}/models/{model_name}", set_to_eval = True)

    # --- Load test data from existing path ---
    train_data, test_data = get_mnist_train_test_data(path, download = False)

    # --- Predict outputs ---
    images, outputs = predict_on_mnist(test_data, network, print_output=True)
    
    # --- Plot Predictions ---
    plot_outputs(outputs, images, 3, 3)

    # --- Plot handwritten digits
    predict_and_plot_handwritten_digits(network, f"{path}/images")
    
    return

if __name__ == "__main__":
    main(sys.argv)