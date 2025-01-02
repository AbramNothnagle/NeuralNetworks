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
# Use for binary classification
class SimpleNN_2Layer_Classification(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=100):
        super(SimpleNN_2Layer_Classification, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer
        self.fc2 = nn.Linear(hidden_size, output_size)  # Second layer
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
# Use for binary classification
class SimpleNN_3Layer_Classification(nn.Module):
    def __init__(self, input_size, output_size=1, hidden_size=100):
        super(SimpleNN_3Layer_Classification, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # First layer
        self.fc2 = nn.Linear(hidden_size, hidden_size)  # Second layer
        self.fc3 = nn.Linear(hidden_size, output_size)  # Second layer
        self.activation1 = nn.ReLU()  # Activation function
        self.activation2 = nn.ReLU()  # Activation function
        self.output = nn.Sigmoid()  # Output activation for binary classification

    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.activation2(self.fc2(x))
        x = self.output(self.fc3(x))
        return x
    
# Simple 2 layer parallel neural network
# Fully connected linear layers, basically a parallel ensembled trained together
# ReLU activation
# Sigmoid output
# Use for binary classification
# Include a use_sigmoid choice: set to False if using BCEWithLogitsLoss
class ParallelNN_2Layer_Classification(nn.Module):
    """
    Simple 2-layer parallel neural network for binary classification.
    Fully connected layers with ReLU activation and optional Sigmoid output.
    
    Args:
        input_size (int): Size of the input features.
        output_size (int): Number of output features (default=1 for binary classification).
        hidden_size (int): Number of hidden units in each layer.
        use_sigmoid (bool): If True, applies a Sigmoid activation to the output layer.
                            Set to False if using BCEWithLogitsLoss for better numerical stability.
    """
    def __init__(self, input_size, output_size=1, hidden_size=100, use_sigmoid = True):
        super(ParallelNN_2Layer_Classification, self).__init__()
        #use_sigmoid = True for BCELoss
        #use_sigmoid = False for BCEWithLogitsLoss
        self.use_sigmoid = use_sigmoid
        
        #Branch 1
        self.branch1_fc1 = nn.Linear(input_size, hidden_size)
        self.branch1_fc2 = nn.Linear(hidden_size, hidden_size)
        self.branch1_activation1 = nn.ReLU()
        self.branch1_activation2 = nn.ReLU()
        
        #Branch 2
        self.branch2_fc1 = nn.Linear(input_size, hidden_size)
        self.branch2_fc2 = nn.Linear(hidden_size, hidden_size)
        self.branch2_activation1 = nn.ReLU()
        self.branch2_activation2 = nn.ReLU()
        
        #Shared output
        self.output_layer = nn.Linear(hidden_size * 2, output_size)
        if use_sigmoid:
            self.output_activation = nn.Sigmoid()

    def forward(self, x):
        #Branch 1
        branch1 = self.branch1_activation1(self.branch1_fc1(x))
        branch1 = self.branch1_activation2(self.branch1_fc2(branch1))
        
        #Branch 2
        branch2 = self.branch2_activation1(self.branch2_fc1(x))
        branch2 = self.branch2_activation2(self.branch2_fc2(branch2))
        
        #Combine the two branches
        combined = torch.cat((branch1, branch2), dim=1)
        
        output = self.output_layer(combined)
        #If using a loss function such as BCEWithLogitsLoss you don't want to use a sigmoid activation layer
        if self.use_sigmoid:
            output = self.output_activation(output)
        
        return output