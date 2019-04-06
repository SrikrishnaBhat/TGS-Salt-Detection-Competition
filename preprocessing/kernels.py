import numpy as np
import cv2

def gabor_kernel(num):
    kernels = []

    ksize = (3, 3)

    theta=0

    while np.pi-theta>1e-6:
        kernel = cv2.getGaborKernel(ksize, 4.0, theta, 10.0, 0.5, 0, cv2.CV_32F)
        kernels.append(kernel/(1.5*kernel.sum()))
        theta += np.pi/num

    return kernels

def process_gabor(img, kernels):
    accum = np.zeros_like(img)
    for kernel in kernels:
        fimg = cv2.filter2D(img, cv2.CV_8UC1, kernel)
        accum = np.maximum(accum, fimg.reshape(accum.shape))
    return accum

def process_gabor_stacked(img, kernels):
    accum = []
    for kernel in kernels:
        accum.append(cv2.filter2D(img, cv2.CV_8UC1, kernel))

    return accum