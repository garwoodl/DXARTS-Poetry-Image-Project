import numpy as np
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from scipy import datasets

def raccoon_example():
    # get raccoon image
    face = datasets.face()
    
    # save the image
    imageio.imsave('raccoon.png', face)

    # read the image
    img = imageio.imread('raccoon.png')
 
    print(img.shape)
    print(img.dtype)
    
    plt.imshow(img)
    plt.show()
 
    x, y, _ = img.shape
    
    # Cropping the image
    crop = img[x//3: - x//8, y//3: - y//8]

    imageio.imsave('crop.png', crop)


def linear_transfrom(A, B, steps):
    """
    Parameters:
        A and B are numpy arrays corresponding to images
        steps is the number of images between A and B inclusive
    Returns:
        A list of stpes matrices beginning with A and ending with B where each 
        setp is a linear transformation of pixels from A to B
        Returns None if A and B are different sizes
    """
    if A.shape != B.shape:
        return None
    
    # the difference matrix is delta
    delta = B - A

    # step size matrix
    step_size = (1 / steps) * delta

    # list of image matrices
    matrices = [A + i * step_size for i in range(steps + 1)]

    return matrices


def main():


if __name__ == '__main__':
    main()