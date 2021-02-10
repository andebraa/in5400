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


# Check that your implementations provide the same result for a small 2D image.
f1 = np.arange(4)
f2 = np.arange(5)
f = f1[:, np.newaxis, np.newaxis] + f2[np.newaxis, :, np.newaxis]
print(f[:,:,0])
print(f.shape)

h = np.arange(9).reshape(3, 3)
print(h)
print(h.shape)

out1 = convolution_loops(f, h)
print(out1[:,:,0])

out2 = convolution(f, h)
print(out2[:,:,0])


# Check that your implementations filter an image correctly.
img = imageio.imread('images/cat.png').astype(np.float64)

kernel = np.arange(25).reshape((5, 5))

start = time.time()
out1 = convolution_loops(img, kernel)
print('Calculation time with inner loops:', time.time()-start, 'sec')

start= time.time()
out2 = convolution(img, kernel)
print('Calculation time without inner loops:', time.time()-start, 'sec')

out1 -= out1.min()
out1 /= out1.max()
out1 *= 255
out1 = out1.astype(np.uint8)

out2 -= out2.min()
out2 /= out2.max()
out2 *= 255
out2 = out2.astype(np.uint8)

correct = imageio.imread('images/convolution_cat.png')
print(np.shape(correct))
plt.figure()
plt.subplot(1, 3, 1)
plt.imshow(correct)
plt.title('Correct')
plt.subplot(1, 3, 2)
plt.imshow(out1)
plt.title('Max abs diff {}'.format(np.max(np.abs(correct-out1))))
plt.subplot(1, 3, 3)
plt.imshow(out2)
plt.title('Max abs diff {}'.format(np.max(np.abs(correct-out2))))
plt.show()



def blur_filter(image):
    """
    Blurs a MxNxC image with an average filter (box filter) with kernel size of 11.
    """
    out = np.zeros(image.shape)


    mk = 11
    nk = 11
    m = image.shape[0]
    n = image.shape[1]
    c = image.shape[2]
    for i in range(m-mk +1):
        for j in range(n-nk +1):
            for k in range(c):
                out[i,j,k] = np.average(image[i:i+mk, j:j+nk, k])
    return out


# Check that your blurring implementation is correct.
img = imageio.imread('images/cat.png').astype(np.float64)

start = time.time()
out = blur_filter(img)
print('Calculation time:', time.time()-start, 'sec')

out -= out.min()
out /= out.max()
out *= 255
out = out.astype(np.uint8)

correct = imageio.imread('images/blur_cat.png')
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(correct)
plt.title('Correct')
plt.subplot(1, 2, 2)
plt.imshow(out)
plt.title('Max abs diff {}'.format(np.max(np.abs(correct-out))))
plt.show()


def gradient_magnitude(img):
    """
    Computes the gradient magnitude of a MxNxC image using the Sobel kernels.
    """
    sobel_y = np.array([[1, 2, 1],
                        [0, 0, 0],
                        [-1, -2, -1]])
    sobel_x = np.array([[1, 0, -1 ],
                       [2, 0, -2],
                       [1, 0, -1]])

    out_y = convolution(img, sobel_y)
    out_x = convolution(img, sobel_x)
    print(np.shape(out_y))
    print(np.shape(out_x))
    gm = np.linalg.norm((out_y, out_x), axis=0)
    print(np.shape(gm))
    gm_max = max_filter(gm)
    return gm_max

def max_filter(image):
    """
    Blurs a MxNxC image with an average filter (box filter) with kernel size of 11.
    """
    out = np.zeros(image.shape)


    mk = 11
    nk = 11
    m = image.shape[0]
    n = image.shape[1]
    c = image.shape[2]
    for i in range(m-mk +1):
        for j in range(n-nk +1):
            out[i,j,k] = np.max(image[i:i+mk, j:j+nk, :])
    return out

image = np.arange(1,26).reshape(5,5,1)

gradient_magnitude(image)


# Check that your gradient magnitude implementation is correct.
img = imageio.imread('images/cat.png').astype(np.float64)

start = time.time()
out = gradient_magnitude(img)
print('Calculation time:', time.time()-start, 'sec')

out -= out.min()
out /= out.max()
out *= 255
out = out.astype(np.uint8)

correct = imageio.imread('images/sobel_cat.png')
plt.figure()
plt.subplot(1, 2, 1)
plt.imshow(correct, cmap='gray')
plt.title('Correct')
plt.subplot(1, 2, 2)
plt.imshow(out, cmap='gray')
plt.title('Max abs diff {}'.format(np.max(np.abs(correct-out))))
plt.show()
