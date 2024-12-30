# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 20:28:32 2024

@author: abram
"""

import numpy as np
import pandas as pd

def logic1(x, threshold):
    if x > threshold:
        return 1
    else:
        return 0

def logic2(x, threshold):
    if x < threshold:
        return 1
    else:
        return 0

def logic3(x1, x2, x3):
    if (logic1(x1, 0.3) and logic2(x2, 0.6)) or x3 < 0.3:
        return 1
    else:
        return 0
    
def getRandomTuple():
    x1 = np.random.rand()
    x2 = np.random.rand()
    x3 = np.random.rand()
    y = logic3(x1, x2, x3)
    return [x1, x2, x3, y]

sampleData = []
for i in range(10000):
    sampleData.append(getRandomTuple())

df = pd.DataFrame(sampleData, columns=['x1','x2','x3','y'])

df.to_csv('basicLogicSample.csv', index=False)
print('done')