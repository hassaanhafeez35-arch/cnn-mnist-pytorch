# CNN MNIST Digit Classifier — PyTorch

A convolutional neural network (CNN) built with PyTorch to classify handwritten digits from the MNIST dataset.

## Overview

This project improves on a basic feedforward network by using convolutional layers to classify handwritten digits (0–9) from the MNIST dataset. Convolutions let the model learn spatial patterns (edges, curves, shapes) directly from the 2D image, rather than treating it as a flat list of pixels.

## Model Architecture

**Feature extractor (convolutional layers):**
- Conv2d (1 → 8 channels, 3x3 kernel) + ReLU + MaxPool2d
- Conv2d (8 → 16 channels, 3x3 kernel) + ReLU + MaxPool2d

**Classifier (fully connected layers):**
- Flatten → Linear (16×7×7 → 64) + ReLU
- Linear (64 → 10) — one output per digit class

## Tools & Libraries

- Python
- PyTorch (`torch`, `torch.nn`, `torch.optim`)
- torchvision (for the MNIST dataset)
- Matplotlib (for visualizing predictions)

## Training Details

- Loss function: Cross-Entropy Loss
- Optimizer: Adam (learning rate = 0.001)
- Epochs: 3
- Batch size: 64

## How to Run

```bash
pip install torch torchvision matplotlib
python projectcnn.py
```

The script will:
1. Download the MNIST dataset automatically (if not already present)
2. Train the CNN for 3 epochs, printing loss per epoch
3. Evaluate accuracy on the test set
4. Save the trained model as `simple_cnn_mnist.pth`
5. Display a sample test image with the model's predicted label vs. the actual label

## Results

The CNN achieves strong accuracy on the MNIST test set, improving on a plain fully-connected network by learning spatial features through convolution and pooling layers.

## What I Learned

- How convolutional layers (`Conv2d`) and pooling (`MaxPool2d`) extract spatial features from images
- Structuring a model with separate feature-extraction and classification stages
- Correctly reshaping conv output before feeding it into fully connected layers
- Running inference on a single image and visualizing the prediction with Matplotlib
- Comparing CNN performance against a simpler feedforward baseline
