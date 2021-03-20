import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


label = np.load('../data/scores/concat_labels.npy')
preds = np.load('../data/scores/concat_pred.npy')

print(np.shape(label))
print(np.shape(preds))

def tailacc(t):
    pass
