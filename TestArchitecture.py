# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 17:44:55 2025

@author: abram
"""

#Imports

#Import the model to be used from ExperimentalNNs

#Setup variables

#Read in and alter data as needed

#Split data to train and test

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from nnUtilities import train, test

#TODO: Update this to dynamically get the paramters file name from something like
#system arguments or another script that calls this script
#for now hard code it
#parameterFileName is the JSON file that stores experiment parameters
parameterFileName = "DummyExperimentTest.json"
parameters = pd.read_json(parameterFileName)

import ExperimentalNNs
MUT =  getattr(ExperimentalNNs, parameters["Parameters"]["Model"])

# Startup Parameters go here:
num_variables = parameters["Parameters"]["input_size"]
num_outputs = parameters["Parameters"]["output_size"]
epochs = parameters["Parameters"]['Epochs']
final_num_features = parameters["Parameters"]["Stop"]
start = parameters["Parameters"]["Start"]
fn_hash = parameters["Parameters"]['fn_hash']
# Load the dataset
fileName = parameters["Parameters"]["FileName"]
data = pd.read_csv(f'DataGenerators\\{fileName}.csv')

# Extract features and labels
# Expect that all variables will be labeled x1, ..., xn (n = num_variables)
input_labels = []
for i in range(1,num_variables+1):
    input_labels.append(f'x{i}')
X = data[input_labels].values
# Expect that all outputs will be labeled y1, ..., ym (m = num_outputs)
output_labels = []
for i in range(1,num_outputs+1):
    output_labels.append(f'y{i}')
y = data[output_labels].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

accuracies = []
for i in range(start, final_num_features):
    model = MUT(input_size=num_variables,output_size=1,hidden_size=i)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train(epochs, criterion, optimizer, model, X_train, y_train)
    accuracies.append(test(model, X_test, y_test))

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(list(range(start,final_num_features)), accuracies, label='Test Accuracy')
plt.legend()
plt.title('Test Accuracy After 100 Epochs in 2 Layer NN')
plt.xlabel('Num FEatures')
plt.ylabel('Test Accuracy')
# Save the plot
#plt.savefig(f'experiment1_accuracy_{start}_{final_num_features}_{fn_hash}.png', format='png')  # Save as PNG
plt.show()

