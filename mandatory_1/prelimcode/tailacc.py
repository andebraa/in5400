import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(z):
	return(1/(1 + np.exp(-z)))


def tailacc(t):
    label = np.load('../data/scores/concat_labels.npy')
    preds = np.load('../data/scores/concat_pred.npy')

    #NOTE! sigmoid really pushes some values way down. is it really okay to do this?
    preds = sigmoid(preds) #force predictions withing [0,1]
    print(preds)
    threshold_mask = preds < t #making mask of values lESS THAN THRESHOLD
    preds[threshold_mask] = 0
    preds[np.logical_not(threshold_mask)]= 1 #all things not 0 is now 1
    norm = np.sum(preds, axis=0) #we wish to work all the values for each clas
    #by itself, adn then average at the end

    val = np.sum((preds*label), axis=0)
    return np.average(val/norm)

tailacc = np.vectorize(tailacc)
t = np.linspace(0,1,80)
plt.plot(t,tailacc(t))
plt.xlabel('threshold')
plt.ylabel('tailacc ')
plt.show()
