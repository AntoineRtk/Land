"""
@author Antoine Ratouchniak
"""

import numpy as np
from numpy.fft import fft2, ifft2
from numpy.fft import fftfreq
from numpy import meshgrid
import matplotlib.pyplot as plt

def blur_land(img):
    
    x = img
    
    ff = fft2(x) # Compute the 2D DFT
    
    h, w = x.shape
    
    p, q = meshgrid(fftfreq(w), fftfreq(h)) # Return the grid of frequency
    # bin centers in cycle
    
    s = 1
    
    f = (p ** 2 + q ** 2) ** (-s / 2) # The Fourier Transform of k is k itself
    f[0, 0] = 0
    
    y = f * ff # Pointwise product
    
    r = ifft2(y).real
    
    return r

def blur_gaussian(img, sigma):
    
    x = img
    
    ff = fft2(x)
    
    h,w = x.shape
    
    p,q = meshgrid(fftfreq(w), fftfreq(h))
    
    f = np.exp(-((p * p + q * q) * sigma * sigma) / 2) # Fourier Transform of the Gaussian
    
    y = f * ff
    
    r = ifft2(y).real
    
    return r

def retinex(path):
    
    img = np.load(path)
    
    blurred_img = blur_gaussian(img, 3) # Gaussian as a surrounding function
    #b_img = blur_land(img) # Land's kernel as a surrounding function
    
    blurred_img -= np.min(blurred_img) - 0.1
    blurred_img *= 254.9 / np.max(blurred_img) # We set images between 0 and 255
    final_img = np.log(img + 0.1) - np.log(blurred_img)
    
    prod = np.ones(final_img.shape)
    for i in range(len(prod)):
        for j in range(len(prod)):
            C = 1000
            #prod[i][j] =  np.log(1 + C * img[i][j]) # if we want to apply color restoration
    
    final_img = final_img * prod
    
    plt.imshow(final_img, cmap = 'gray')