# -*- coding: utf-8 -*-
"""
Created on Mon Dec 30 00:14:00 2024

@author: abram
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# Load the dataset
data = pd.read_csv('DataGenerators\\basicLogicSample.csv')

# Extract features and labels
X = data[['x1', 'x2', 'x3']].values
y = data['y'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
#scaler = StandardScaler()
#X_train = scaler.fit_transform(X_train)
#X_test = scaler.transform(X_test)

# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define the neural network
class SimpleNN(nn.Module):
    def __init__(self, numFeatures):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(3, numFeatures)  # First layer
        self.fc2 = nn.Linear(numFeatures, 1)  # Second layer
        self.activation1 = nn.ReLU()  # Activation function
        self.output = nn.Sigmoid()  # Output activation for binary classification

    def forward(self, x):
        x = self.activation1(self.fc1(x))
        x = self.output(self.fc2(x))
        return x

def train(epochs, criterion, optimizer, model):
    # Training loop
    num_epochs = epochs
    for epoch in range(num_epochs):
        model.train()

        # Forward pass
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
    
    #return model

def test(model):
    # Testing loop
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        predictions = (predictions > 0.5).float()
        accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
        print(f'Test Accuracy: {accuracy:.4f}')
    
    return accuracy

accuracies = []
epochs = 100
final_num_features = 30
start = 1
for i in range(start, final_num_features):
    model = SimpleNN(i)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss for classification
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    train(epochs, criterion, optimizer, model)
    accuracies.append(test(model))

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(list(range(start,final_num_features)), accuracies, label='Test Accuracy')
plt.legend()
plt.title('Test Accuracy After 100 Epochs in 2 Layer NN')
plt.xlabel('Num FEatures')
plt.ylabel('Test Accuracy')
# Save the plot
plt.savefig(f'experiment1_accuracy_{start}_{final_num_features}.png', format='png')  # Save as PNG
plt.show()