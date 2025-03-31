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

import torch
import torchvision.transforms as transforms
import cv2
import numpy as np

from mnist_network import MyNetwork, get_path
from greek_network_extension1 import GreekNetwork
from mnist_results import load_model


# =========================================================================================================
# Useful Function
# =========================================================================================================



# =========================================================================================================
# Main Function
# =========================================================================================================

def main(argv):
    # handle any command line arguments in argv

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    #Setting up path variable for the project
    path = get_path()

    # Load saved model
    mnist_network = load_model(MyNetwork, f"{path}/models/mnist_model.pth", set_to_eval = True)
    greek_network = load_model(GreekNetwork, f"{path}/models/greek_model.pth", set_to_eval = True)
    current_network = mnist_network
    current_pred_classes = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}

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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inverted_frame = cv2.bitwise_not(gray)
        height, width = inverted_frame.shape

        region_size = 300
        # Compute center region coordinates
        center_x = width // 2
        center_y = height // 2
        x = center_x - region_size // 2
        y = center_y - region_size // 2

        roi = inverted_frame[y:y+region_size, x:x+region_size]
        white_pixels = cv2.countNonZero(roi)

        if white_pixels > 50:
            roi_resized = cv2.resize(roi, (28, 28), interpolation=cv2.INTER_AREA)
            tensor = transform(roi_resized).unsqueeze(0).to(device)  # [1, 1, 28, 28]

            # Predict
            values = []
            with torch.no_grad():
                output = current_network(tensor)
                probs = torch.exp(output)  # Convert from log_softmax to probabilities
                values = probs.tolist()[0]
                pred = output.argmax(dim=1).item()

            cv2.rectangle(frame, (x, y), (x+region_size, y+region_size), (255, 0, 0), 2)

            if max(values) > 0.9:
                # Show prediction in center region
                cv2.putText(frame, "Pred: "+current_pred_classes[pred], (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(frame, "Pred: Unclear", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # Display
        cv2.imshow("Digit Recognition", frame)
    
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('m'):
            current_network = mnist_network
            current_pred_classes = {0: "0", 1: "1", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 8: "8", 9: "9"}
            print("Switched to MNIST classification")
        elif key == ord('g'):
            current_network = greek_network
            current_pred_classes = {0: "alpha", 1: "beta", 2: "gamma"}
            print("Switched to Greek classification")
    
    cap.release()
    cv2.destroyAllWindows()
        
    
    return

if __name__ == "__main__":
    main(sys.argv)