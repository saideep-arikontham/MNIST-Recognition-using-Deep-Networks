# MNIST-Recognition-using-Deep-Networks

## Overview
This project demonstrates the training of different deep neural networks on MNIST dataset, Greek letters and Fashion MNIST.

---

## Different Areas explored in the project
- Training a simple CNN to train and predict on MNIST dataset.
- Test the network against hand written images.
- Examining the filters in the first convolution layer.
- Using the trained MNIST network to train on Greek letters using transfer learning.
- Performing an experimentation to find the best architecture for training Fashion MNIST dataset.
- Performing a full training on greek letters in an attempt to get better performance (extension 1)
- Live digit and letter detection using the above trained models (extension 2)

---

## Project Structure

```
├── bin/
│   ├── #Executable binaries
│
├── data
│   ├── # To store the datasets
│
├── images/                                
│   ├── # Hand written digits for testing MNIST model
│
├── models/
│   ├── # Trained weights of different networks
│
├── src/                                    # Source files
│   ├── fashion_mnist_network.py            # Experimentation to find best architecture for Fashion MNIST
│   ├── greek_network_extension1.py         # Full training on Greek Letters
│   ├── mnist_live_detection_extension2.py  # Live detection of digits and greek letters
│   └── mnist_network_examination.py        # Examining weights of trained MNIST layers
│   └── mnist_network.py                    # Training MNIST network
│   └── mnist_results.py                    # Examining trained MNIST network on hand written images
│   └── mnist_transfer_greek.py             # Transfer learning for greek letters
│
├── .gitignore                              # Git ignore file
├── makefile                                # Build configuration
├── Project5_Report.pdf                     # Project report
```

---

## Tools used
- `OS`: MacOS
- `IDE`: Visual Studio code
- `Camera source`: Iphone (Continuity camera)

---

## Dependencies

```
name: prcv
channels:
  - conda-forge
  - defaults
dependencies:
  - ipykernel
  - matplotlib
  - numpy
  - pandas
  - pytorch
  - seaborn
  - torchvision
  - scikit-learn
  - opencv
prefix: /Users/saideepbunny/anaconda3/envs/prcv
```


---

## Installation

1. Clone this repository:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

---

## Running the code

- To train the MNIST network, run the below command with your choice of command line argument values.

```
python mnist_network.py <epochs> <batch_size> <learning_rate> <momentum>
```

- To train the results of the trained MNIST network, run the below command.

```
python mnist_results.py mnist_model.pth
```

- To Examine the first convolution layer of the network, use the following command.

```
python mnist_network_examination.py mnist_model.pth
```

- To perform transfer learning to train the MNIST model on greek network, use the command below with your choice of agrument values.

```
python mnist_transfer_greek.py <model_name> <learning_rate> <batch_size> <momentum> <epochs>
```

- To peform experimentation and fine the best architecture for Fashion MNIST model, use the below command with appropriate command line arguments.

```
python fashion_mnist_network.py <max_conv_layers> <max_lin_layers> <max_dropout> <max_hidden_units> <max_batch_size> <max_epochs>
```

- To perform a full training on greek letters, use the below command with your choice of command line arguments.

```
 python greek_network_extension1.py <learning_rate> <batch_size> <momentum> <epochs>
```

- To perform live digit/letter detection in a video stream, use the below command.

```
python mnist_live_detection_extension2.py
```

---

## Live video detection usage

Key press functionality of OpenCV gives the luxary of using either digit detection or letter detection. Below are the choices available for live detection

1. For `mnist_live_detection_extension2.py`:

- `g`: To predict greek letters.
- `m`: To predict mnist digits

Small live demonstration is included in **[Project5_Report.pdf](https://github.com/saideep-arikontham/MNIST-Recognition-using-Deep-Networks/blob/main/Project5_Report.pdf)**

Access the video directly here: https://drive.google.com/file/d/1chQ-1dJ1XscJLVVgansy_SQ3sqYKW-eV/view?usp=drive_link


---

## Highlights
- Function written in files like mnist_network, etc., are utilized in other python files to increase reusability.

- Below is the link to hand written greek letters used for testing:

https://drive.google.com/file/d/1P2hp5bTPXqs2GR31K2RVtnG6xTIG2Zvy/view?usp=drive_link

---

## Note

Used 3 travel days for this project

---

## Contact
- **Name**: Saideep Arikontham
- **Email**: arikontham.s@northeastern,edu