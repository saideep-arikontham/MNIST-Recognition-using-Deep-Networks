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

from mnist_network import MyNetwork, get_path
from mnist_results import load_model


# =========================================================================================================
# Useful Function
# =========================================================================================================



# =========================================================================================================
# Main Function
# =========================================================================================================

def main(argv):
    # handle any command line arguments in argv

    #Setting up path variable for the project
    path = get_path()

    # Load saved model
    network = load_model(f"{path}/models/mnist_e10_model.pth", set_to_eval = True)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
    
        # Preprocessing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        roi = gray[100:300, 100:300]  # Region of Interest
        roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
        roi_inverted = cv2.bitwise_not(roi_resized)  # MNIST digits are white-on-black
    
        # Normalize & reshape
        tensor = transform(roi_inverted).unsqueeze(0)  # [1, 1, 28, 28]
    
        # Predict
        with torch.no_grad():
            output = network(tensor)
            pred = output.argmax(dim=1).item()
    
        # Show prediction
        cv2.rectangle(frame, (100, 100), (300, 300), (255, 0, 0), 2)
        cv2.putText(frame, f"Prediction: {pred}", (100, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
        # Display
        cv2.imshow("Digit Recognition", frame)
    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
        
    
    return

if __name__ == "__main__":
    main(sys.argv)