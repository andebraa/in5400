import numpy as np
from matplotlib import pyplot as plt
import imageio
import time

plt.rcParams['figure.figsize'] = (14.0, 12.0)

def convolution_loops(image, kernel):
    """
    Convolves a MxNxC image with a MkxNk kernel.
    """
    out = np.zeros(image.shape)

    kernel = np.rot90(kernel, 2) # rotate 180 degrees to perform convolution (not correlation)
    mk = kernel.shape[0]
    nk = kernel.shape[1]
    m = image.shape[0]
    n = image.shape[1]
    c = image.shape[2]
    for i in range(m-mk+1):
        for j in range(n-nk+1):
            for k in range(c):
                for l in range(mk):
                    for m in range(nk):
                        out[i,j,k] += image[i+l,j+m,k]*kernel[l,m]


    return out
def convolution(image, kernel):
    """
    Convolves a MxNxC image with a MkxNk kernel.
    """
    out = np.zeros(image.shape)

    kernel = np.rot90(kernel, 2) # rotate 180 degrees to perform convolution (not correlation)
    mk = kernel.shape[0]
    nk = kernel.shape[1]
    m = image.shape[0]
    n = image.shape[1]
    c = image.shape[2]
    for i in range(m-mk +1):
        for j in range(n-nk +1):
            for k in range(c):
                out[i,j,k] = np.sum(image[i:i+mk, j:j+nk, k]*kernel)
    return out
