# -*- coding: utf-8 -*-
"""
Created on Mon Jan 27 19:02:36 2025

@author: Abram Nothnagle
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from ExperimentalNNs import SimpleNN_3Layer_Classification
from nnUtilities import train, test


###############################################################################
# SECTION 1
# LOAD UP THE DATA
###############################################################################

# Get the data and Shape it correctly
data = pd.read_csv(f'DataGenerators\\Data\\SPY_signals.csv')

#print(data.columns)

# The first 242 rows don't have complete data because it takes 60 days to fill up SMA60 etc.
# Split the data into the y (results) and x (input data)
# My current system is only set up for binary classification... so we'll just look at buys
y = data.loc[242:, ['Buy']]
x = data.loc[242:, ['Month','MonthDay','Day','LastClose','High','Low','SMA3','SMA10','SMA20', 'SMA60',
         'STD3','STD10','STD20','STD60','WMA10','WMA20','WMA60','Min15','Max15','Price']]
# Print out the shape so we know what we're dealing with
print(y.shape)
print(x.shape)

# Now split the data into train and test data
# Train on the first 9000 datapoints (approximately 9.6 years)
x_train = x.loc[:9600]
y_train = y.loc[:9600]
# Test on the remaining datapoints (approximately 0.4 year)
x_test = x.loc[9600:]
y_test = y.loc[9600:]

# Transform them into tensors
x_train = torch.tensor(x_train.values, dtype=torch.float32)
y_train = torch.tensor(y_train.values, dtype=torch.float32)
x_test = torch.tensor(x_test.values, dtype=torch.float32)
y_test = torch.tensor(y_test.values, dtype=torch.float32)

###############################################################################
# SECTION 2
# LOAD UP THE MODEL
###############################################################################

in_size = 20
out_size = 1
h = 300
model = SimpleNN_3Layer_Classification(in_size, out_size, h)

###############################################################################
# SECTION 3
# SET UP TRAINING PARAMETERS
###############################################################################

# Pick a loss function. We'll want to do a custom loss function eventually
# For now BCE is fine just to get the ball rolling
criterion = nn.BCELoss()  # Binary Cross Entropy Loss for classification
# Select an optimizer
optimizer = optim.Adam(model.parameters(), lr=0.01)
# Set up the number of Epochs to run
epochs = 1000

###############################################################################
# SECTION 4
# TRAIN AND TEST
###############################################################################
train(epochs, criterion, optimizer, model, x_train, y_train)
print(test(model, x_test, y_test))