# -*- coding: utf-8 -*-
"""
Created on Thu Jan  2 18:14:52 2025

@author: Abram Nothnagle
"""

import torch
from sklearn.metrics import accuracy_score

def train(epochs, criterion, optimizer, model, X_train, y_train):
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

def test(model, X_test, y_test):
    # Testing loop
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        print(predictions[1:10])
        predictions = (predictions > 0.5).float()
        print(predictions[1:10])
        print(y_test[1:10])
        accuracy = accuracy_score(y_test.numpy(), predictions.numpy())
        print(f'Test Accuracy: {accuracy:.4f}')
    
    return accuracy