# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 16:20:23 2025

@author: abram
"""

import torch
import torch.nn as nn

# Simple 2 layer neural network
# Fully connected linear layers
# ReLU activation
# Sigmoid output
# Use for classification
class SimpleNN_2Layer_Classification(nn.Module):
    def __init__(self, numInputs, numOutputs=1, numFeatures=100):
        super(SimpleNN_2Layer_Classification, self).__init__()
        self.fc1 = nn.Linear(numInputs, numFeatures)  # First layer
        self.fc2 = nn.Linear(numFeatures, numOutputs)  # Second layer
        self.activation1 = nn.ReLU()  # Activation function
        self.output = nn.Sigmoid()  # Output activation for binary classification

    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.output(self.fc2(x))
        return x
    
# Simple 3 layer neural network
# Fully connected linear layers
# ReLU activation
# Sigmoid output
# Use for classification
class SimpleNN_3Layer_Classification(nn.Module):
    def __init__(self, numInputs, numOutputs=1, numFeatures=100):
        super(SimpleNN_3Layer_Classification, self).__init__()
        self.fc1 = nn.Linear(numInputs, numFeatures)  # First layer
        self.fc2 = nn.Linear(numFeatures, numFeatures)  # Second layer
        self.fc3 = nn.Linear(numFeatures, numOutputs)  # Second layer
        self.activation1 = nn.ReLU()  # Activation function
        self.activation2 = nn.ReLU()  # Activation function
        self.output = nn.Sigmoid()  # Output activation for binary classification

    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        x = self.output(self.fc3(x))
        return x