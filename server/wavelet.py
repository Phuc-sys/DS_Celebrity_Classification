import pywt
import cv2
import numpy as np


def w2d(img, mode='haar', level=1):
    imArray = img
    #Datatype conversion
    #convert to grayscale
    imArray = cv2.cvtColor(imArray, cv2.COLOR_RGB2GRAY)
    #convert to float
    imArray = np.float32(imArray)
    imArray /= 255
    #compute coefficients
    coeffs = pywt.wavedec2(imArray, mode, level=level)

    #Process coefficients
    coeffs_h = list(coeffs)
    coeffs_h[0] *= 0

    #Reconstruction
    imArray_h = pywt.waverec2(coeffs_h, mode)
    imArray_h *= 255
    imArray_h = np.uint8(imArray_h)

    return imArray_h
