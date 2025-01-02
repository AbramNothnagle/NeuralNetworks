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

model_as_text = "SimpleNN_2Layer_Classification"
#from ExperimentalNNs import SimpleNN_2Layer_Classification as MUT #MUT = Model Under Test
import ExperimentalNNs
MUT =  getattr(ExperimentalNNs, model_as_text)



# Startup Parameters go here:
num_variables = 5
epochs = 100
final_num_features = 30
start = 1
fn_hash = '5d735a1c'
# Load the dataset
data = pd.read_csv('DataGenerators\\logi_fn_dataset_5_1000_lopn_5d735a1c.csv')

# Extract features and labels
labels = []
for i in range(1,num_variables+1):
    labels.append(f'x{i}')
X = data[labels].values
y = data['y'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

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

