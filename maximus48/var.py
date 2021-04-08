#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 17:46:49 2018

@author: mpolikarpov
"""


# import matplotlib.pyplot as plt
# import cv2
import os
#from PIL import Image
import numpy as np
# from numpy import mean, square, sqrt
from numpy.fft import fft2, fftshift
# from joblib import Parallel, delayed
import skimage
import scipy



def im_folder(path):
    """
    lists images in the folder 
    
    Parameters
    __________
    path : str
    """
    
    fileformat = 'ppm','PPM','tiff','TIFF','tif','TIF','png','PNG', 'raw', 'jpg', 'JPG'
    curfol = os.getcwd()
    
    os.chdir(path)
    imfiles = os.listdir(path)
    imlist = [filename for filename in imfiles if filename.endswith(fileformat) and not (filename.startswith('.'))]
    imlist.sort()
    os.chdir(curfol)
    
    return(imlist)
    
    
    
    
    
    
# def fourimage(image):
#     """
#     does inverse fourier transform and shows the image
    
#     Parameters
#     __________
#     data : ndarray 
#         input image data 2D or 3D array
#     bitnum: int 
#         number of bits to save. by default is 8 bit
#     """
#     ft = fft2(image)
#     out = np.log(abs(fftshift(ft)))
#     show(out)
#     return 
    
    
def imrescale(data, bitnum=16):
    """
    increases brightness/contrast and returns it as int array
    
    Parameters
    __________
    data : ndarray 
        input image data 2D or 3D array
    bitnum: int 
        number of bits to save. by default is 8 bit
    """
    bit = 2**bitnum-1
    out = (data-np.min(data))*bit/(np.max(data)-np.min(data))
    out = out.astype('uint'+str(bitnum))
    return out
    


def wavelen(energy):
    """
    Calculates the wavelength out of Energy in keV
    
    Parameters
    __________
    energy : int
    """
    
    h = 4.135667662 * 1e-18 # plank constant, keV*sec
    c = 299792458           # speed of light , m/sec
    Y = (h*c)/energy
    return Y


def maximal_intensity(image, angle = None):
    
    """
    Метод максимальной интенсивности (томо)
    
    Parameters
    __________
    image : ndarray
        stack of images where the 0 axis corresponds to the tomo slice
    angle : int
        угол обзора 

    """
    from scipy.ndimage import rotate
    if angle:
        image = rotate(image, angle, axes=(2,1))
    IM_MAX= np.max(image, axis=2)

    return IM_MAX



def filt_gauss_laplace(image, sigma = 3):
    """
    Consistently applies Gaussian + Laplace (edge) filters to the image
    You need to specify the kernel of the Gaussian filter (sigma)
    Normally, sigma = 3 is good enough to remove white noise.
    """
    image =  skimage.filters.gaussian(image, sigma)
    return scipy.ndimage.laplace(image)


def shift_distance(image1, image2, accuracy = 100):
    """
    Finds lateral shift between two images 
    
    Parameters
    __________
    image1 : 2D array

    image2 : 2D array
        y axis
    accuracy: int
        Upsampling factor. Images will be registered within 1 / upsample_factor of a pixel. For example upsample_factor == 20 means the images will be registered within 1/20th of a pixel.    
    
    Returns 
    __________
    
    shifts: ndarray
        Shift vector (in pixels) required to register target_image with src_image. 
        Axis ordering is consistent with numpy (e.g. Z, Y, X)
    """
    shift, error, diffphase = skimage.feature.register_translation(image1, image2, accuracy)
    return shift

