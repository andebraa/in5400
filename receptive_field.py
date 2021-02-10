import numpy as np
import matplotlib.pyplot as plt
import time
%matplotlib inline

def receptive_field(f, s):
    # Computes the theoretical receptive field for each layer of a convolutional neural network.

    # Inputs:
    # f (list): Filter size for each layer
    # s (list): Stride for each layer

    # Output
    # R: The calculated receptive field for each layer as a numpy array

    r = np.zeros(len(f))
    r[0] = 1
    for k in range(1, len(f)):
        r[k] =  r[k-1] + ((f[k] - 1)*np.prod(s[:k]))

    return r

# Compute theoretical receptive field for the architectures.

# Architecture 1
A1_filterSize = [3, 3, 3, 3, 3, 3]
A1_stride     = [1, 1, 1, 1, 1, 1]
A1_Recept     = receptive_field(A1_filterSize, A1_stride)
print(A1_Recept)

# Architecture 2
A2_filterSize = [3, 3, 3, 3, 3, 3]
A2_stride     = [2, 1, 2, 1, 2, 1]
A2_Recept     = receptive_field(A2_filterSize, A2_stride)
print(A2_Recept)

# Architecture 3
A3_filterSize = [3, 3, 3, 3, 3, 3]
A3_stride     = [2, 2, 2, 2, 2, 2]
A3_Recept     = receptive_field(A3_filterSize, A3_stride)
print(A3_Recept)

# Architecture 4
A4_filterSize = [5, 5, 5, 5, 5, 5]
A4_stride     = [1, 1, 1, 1, 1, 1]
A4_Recept     = receptive_field(A4_filterSize, A4_stride)
print(A4_Recept)

# Architecture 5
A5_filterSize = [5, 5, 5, 5, 5, 5]
A5_stride     = [2, 1, 2, 1, 2, 1]
A5_Recept     = receptive_field(A5_filterSize, A5_stride)
print(A5_Recept)



plt.figure(figsize=(18, 16), dpi= 80, facecolor='w', edgecolor='k')
ax = plt.subplot(1, 1, 1)
plt.plot(A1_Recept, 'r', label='Architecture 1')
plt.plot(A2_Recept, 'b', label='Architecture 2')
plt.plot(A3_Recept, 'g', label='Architecture 3')
plt.plot(A4_Recept, 'k', label='Architecture 4')
plt.plot(A5_Recept, 'm', label='Architecture 5')
plt.ylabel('Receptive field (R)', fontsize=18)
plt.xlabel('Layer $k$', fontsize=18)
ax.grid()
plt.ylim([0, 140])
plt.xlim([0, 6])
ax.legend(loc='upper left', fontsize=16)
