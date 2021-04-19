#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 16:23:15 2019

@author: mpolikarpov
"""
# import os
import sys
# os.environ['OMP_NUM_THREADS'] ='1'
# os.environ['OPENBLAS_NUM_THREADS'] = '1'
# os.environ['MKL_NUM_THREADS'] = '1'

from maximus48 import var
import tomopy
import numpy as np
from multiprocessing import Array
from maximus48.var import shift_distance as shift
# from maximus48.var import filt_gauss_laplace
from skimage.transform import rescale
from numpy.fft import fft2, fftshift
from scipy import special
import itertools


import dask.array as da
from dask_image.ndinterp import affine_transform
from dask_image import ndfilters as ndf

# from scipy.ndimage.interpolation import _ni_support



"""
I wanted to make clases F and Processor as light as possible for parallel processing 
so, in a sence, I use them almost as fast dictionaries
"""

# =============================================================================
# additional functions
# =============================================================================

def tonumpyarray(shared_array, shape, dtype):
    '''Create numpy array from shared memory.'''
    nparray = np.frombuffer(shared_array, dtype=dtype).reshape(shape)    
    #assert nparray.base is shared_array
    return nparray
        

def correct_shifts(shifts, median_dev = 5):
    """find any bad numbers which deviate more than 5 pixels from the median
    and correct them to median of the array"""
    
    shifts = tonumpyarray(shifts.shared_array_base, shifts.shape, shifts.dtype)
    
    for i in range(shifts.shape[1]):
        for j in range(shifts.shape[2]):
            shifts[:,i,j] = np.where((abs(shifts[:,i,j] - np.median(shifts[:,i,j])) > median_dev), np.median(shifts[:,i,j]), shifts[:,i,j])
    print('adjusted shifts') 


def fit_polynom(x, y, poly_deg):
    '''fits the y(x) with a cubic function. 
    Returns the y-coordinate of the fit'''

    pfit = np.polyfit(x,y, poly_deg)
    return np.polyval(pfit, x)



# =============================================================================
# # Set of functions for identification of the rotation axis
# =============================================================================
def rotaxis_rough(proj,i, N_steps = 10):
    """calculate the rotation axis comparing 0 and 180 projection shift
    
    Parameters
    __________
    proj: 3D array
    N_steps: projections per degree
    by default it compares the central part of images (between 1/4 and 3/4 of shape)
    """
    a = proj.shape[1]//4
    b = 3 * proj.shape[1]//4
    c = proj.shape[2]//4
    d = 3 * proj.shape[2]//4
        
                
    cent = []
    
    correction = 180 * N_steps
    
    if proj.shape[0] < correction:
        correction = 0
    
    N_rot = proj.shape[0] - correction
    
    # for i in range(N_rot):
    distances = shift(proj[i, a:b, c:d], np.flip(proj[i + correction, a:b, c:d] ,1))
    # cent.append(proj[i].shape[1]/2 + distances[1]/2)

    return proj[i].shape[1]/2 + distances[1]/2 #cent


def rotrough_compute(proj,i,**kwargs):#a=a,b=b,c=c,d=d,sigma=sigma,accuracy=accuracy,N_steps=N_steps):
    
    # rotrough_compute(proj,i,a=a,b=b,c=c,d=d,sigma=sigma,accuracy=accuracy,N_steps=N_steps):
    
    
    im0_1 =  ndf.gaussian_filter(proj[i, a:b, c:d], sigma)
    im1 = ndf.laplace(im0_1)
    
    # im1 = filt_gauss_laplace(proj[i, a:b, c:d], sigma)
    im0_2 =  ndf.gaussian_filter(proj[i + N_steps*180, a:b, c:d], sigma)
    im2 = np.flip(ndf.laplace(im0_2),1)
    
    # im2 = np.flip(filt_gauss_laplace(proj[i + N_steps*180, a:b, c:d], sigma),1)
    
    distances = shift(im1.compute(), im2.compute() , accuracy)  
    return proj[i].shape[1]/2 + distances[1]/2
    

def rotaxis_rough_filt(proj, N_steps = 10, sigma = 5, accuracy = 10):
    """calculate the rotation axis comparing 0 and 180 projection shift
    
    Parameters
    __________
    proj: 3D array
    N_steps: projections per degree
    by default it compares the central part of images (between 1/4 and 3/4 of shape)
    """
    a = proj.shape[1]//4
    b = 3 * proj.shape[1]//4
    c = proj.shape[2]//4
    d = 3 * proj.shape[2]//4
        
    cent = []
    N_rot = proj.shape[0] - 180 * N_steps
    
    for i in range(N_rot):
        
        im0_1 =  ndf.gaussian_filter(proj[i, a:b, c:d], sigma)
        im1 = ndf.laplace(im0_1)
        
        # im1 = filt_gauss_laplace(proj[i, a:b, c:d], sigma)
        im0_2 =  ndf.gaussian_filter(proj[i + N_steps*180, a:b, c:d], sigma)
        im2 = np.flip(ndf.laplace(im0_2),1)
        
        # im2 = np.flip(filt_gauss_laplace(proj[i + N_steps*180, a:b, c:d], sigma),1)
        
        distances = shift(im1.compute(), im2.compute() , accuracy)
        cent.append(proj[i].shape[1]/2 + distances[1]/2)
    return cent

    

def rotaxis_precise(projections, rotaxis_scan_interval, rot_step = 10, crop_ratio = 2, downscale = 0.25, ncore = None, nchunk = None):
    """
    This function calculates tomo-reconstructions and tells you 
    which tomo-slice is the sharpest one
    it works like auto-focus in a smartphone
    
    Check this post for the idea (I use FFT)
    https://www.pyimagesearch.com/2015/09/07/blur-detection-with-opencv/
   
    Parameters
    __________
    projections: 3D array
        0 direction is different images
    rotaxis_scan_interval: array
        range of integers for potential rotation axis values
    rot_step: int
        number of radiographic projections per 1 degree (typically 1 or 10)
    crop_ratio: int
        which portion of the original image will be considered by this function
        Example: if 2, then the new ROI will be from 1/4 to 3/4 of the original ROI
    downscale: int
        number<1, corresponds to the downscale factor of the image
            for resolution estimation
    elements: 2D array
        0dim: rotaxis coordinate for the most sharpest image
        1dim: standard deviation (contrast) of the tomo-image at this rotaxis
                higher std means better sharpness
    """
    
            
    # calculate angles
    n = projections.shape[0]
    angle = np.pi*np.arange(n)/(rot_step*180)
     
    # counter for best std
    elements = []
    # 0 direction to store rotaxis
    elements.append([])
    # 1 direction to store contrast values
    elements.append([])    
    
    # body
    for i in rotaxis_scan_interval:
      
        # make the reconsturction
        image = tomopy.recon(projections, angle, center = i, 
                             algorithm = 'gridrec', filter_name = 'shepp',ncore = ncore, nchunk = nchunk)[0]
        
        # crop squared tomo-reconstructio so you use only the ROI with a sample.
        a = int(image.shape[0] * (crop_ratio-1)/(2*crop_ratio))
        b = int(image.shape[0] * (crop_ratio+1)/(2*crop_ratio))
        
        # downscale the image
        image = rescale(image[a:b, a:b],
                        downscale, anti_aliasing=True, multichannel = False)
        
        # calculate standard deviation and save results
        image = np.std(np.log(abs(fftshift(fft2(image)))))
        elements[1].append(image)
        elements[0].append(i)
        
    return np.asarray(elements)



def rotaxis_scan(projections, N_slice = 1000, rot_step = 10, ncore = None, nchunk = None):
    """
    The function combines rotaxis_rough() and rotaxis_precise()
    It does three iterations to find the best match for the rotation axis
   
    Parameters
    __________
    projections: 3D array
        0 direction is different images. 
        Please note that 1st direction should have 2 or more values. 
        If it has >2 values, only the first slice will be considered 
            for resolution measurements
    rot_step: int
        number of radiographic projections per 1 degree (typically 1 or 10)
    cent: int
        center of rotation
    """
    print('WARNING! This function will be deprecated')
    
    # rough alignment
    cent = rotaxis_rough(projections, rot_step)
    cent = np.median(cent)

    # first iteration
    opa = rotaxis_precise(projections[:,N_slice:N_slice+1], 
                          np.arange(cent - 100, cent + 100, 5), rot_step,ncore = ncore, nchunk = nchunk)
    cent = opa[0, np.argmax(opa[1])]

    # second iteration
    opa = rotaxis_precise(projections[:,N_slice:N_slice+1],
                          np.arange(cent - 5, cent + 5, 1), rot_step, ncore = ncore, nchunk = nchunk)
    cent = opa[0, np.argmax(opa[1])]
    
    # third iteration
    opa = rotaxis_precise(projections[:,N_slice:N_slice+1],
                          np.arange(cent - 2, cent + 2, 0.1), rot_step, ncore = ncore, nchunk = nchunk)
    cent = opa[0, np.argmax(opa[1])]
    
    #final fit
    fit = fit_polynom(opa[0], opa[1], poly_deg = 3)
    cent = opa[0, np.argmax(fit)]
    
    return cent


def rotscan(proj, N_steps, slice_mode = False, ncore = None, nchunk = None):
    """
    The function combines rotaxis_rough() and rotaxis_precise()
    It does three iterations to find the best match for the rotation axis
    It also finds the inclination of the rotation axis
   
    Parameters
    __________
    projections: 3D array
        0 direction is different images. 
        (1,2) direction - projection views
    N_steps: int
        number of radiographic projections per 1 degree (typically 1 or 10)
    slice_mode: boolean
        if True, the rotation axis will be calculated 
        individually for each row of the projection
    cent: int
        center of rotation
    """
    
        
    
    
    ### rough rotaxis scan
    cent = rotaxis_rough(proj, N_steps)
    cent = np.median(cent)

    ### fine rotaxis scan (optional, only if 360 deg projections)
    # Note: you schould use only the region with a sample. 
    #The noisy region with no data will introduce errors
    list_to_scan = (proj.shape[1]*np.array([1,2,4,6,7])/8).astype(int)
    
    rotslice = []

    for i, sliceN in enumerate(list_to_scan):
        cent_iter = cent
        for scan_step in (10,2):
            calc = rotaxis_precise(proj[sliceN:sliceN+1,:],
                               np.arange(cent_iter - scan_step, cent_iter + scan_step, scan_step/10), 
                               N_steps,ncore = ncore, nchunk = nchunk)
            cent_iter = calc[0, np.argmax(calc[1])]
        rotslice.append(cent_iter)
   
    # calculate tilt of the rotation axis
    # counter-clockwise rotation of the rotaxis => positive angles
    pfit = np.polyfit(list_to_scan, rotslice, 1)
    # inclination = (-np.arctan(pfit[0]*180/np.pi))
    
    if slice_mode:
        # build new coordination axis
        Ycoord = np.arange(proj.shape[1])
        cent = np.polyval(pfit, Ycoord)
    else:
        cent = np.median(rotslice)
    
    return cent#, inclination)



# =============================================================================
# # Additional custom functions for rotaxis 
# =============================================================================
def axis_raws(image1, image2, Npad = 0, RotROI = None, level = 5, window = 50):
    """Finds an axis of rotation by comparing the 1st and the 180deg image:
    This function is based on rotaxis_rough()
    It is useful when you want to search for rotaxis in different regions of 
        your image independently
        
    Parameters
    __________
    image1: 2D array 
        first 2D image
    image2: 2D array
        second 2D image
    Npad: int 
        type the number here if the x axis of your image was padded
    RotROI: tuple
        ROI where to compare the images. Note that it is better to exclude some regions at the edge of the camera
        Coordinate legend:  RotROI[0] - begin line (if you look at the image, numpy logic)
                            RotROI[1] - end line
                            RotROI[2] - begin column
                            RotROI[3] - end column
    level: int 
        the number of pixels to be taken into account, please type None if you don't want to use it and want to export the whole data.
    window: int
        window is the number of pixels on the image (height) to be taken into account during comaprison
    """
    
    if not RotROI:
        RotROI = (50, image1.shape[0],
                  Npad + image1.shape[1]//8, image1.shape[1] - Npad - image1.shape[1]//8)
    

    all_cent=[]
    for i in range(RotROI[0], RotROI[1], window//2):

        im_1 = image1[i:i+window,RotROI[2]:RotROI[3]]
        im_2 = np.flip(image2[i:i+window,RotROI[2]:RotROI[3]],1)

        distances = shift(im_1, im_2)
        cent = im_1.shape[1]/2 + distances[1]/2 + RotROI[2]

        all_cent.append(cent)
        #print('center for the slice #',i,' is: ', cent)


            
    if level:
        x=[]
        y=[]

        for i in range(len(all_cent)):
            if np.absolute(all_cent[i] - np.median(all_cent)) <level:
                x.append(i * window//2)
                y.append(all_cent[i])
    
        all_cent = np.column_stack((x,y))
    
    else:
        all_cent = np.column_stack((np.linspace(0, len(all_cent), len(all_cent), dtype = 'uint16'), all_cent))
    
    #plt.plot(all_cent[:,0], all_cent[:,1], 'bo')
                
    return all_cent



def interpolate(cent, level = None):
    """interpolates the coordinates for the rotation axis with the line
     basically finds the inclination of the rotation axis through the image.
     The input for this function typically comes from axis_raws() 
    
    Parameters
    __________   
    cent: 2D array 
        comes from axis_raws    
    level:int 
        is the number of pixels to be taken into account if you want 
        to discard all values that have more than 5 degrees difference with the median
    """  

    step = cent[1,0] - cent[0,0]
    
    if not level:
        x = cent[:,0]
        y = cent[:,1]
    
    else:
        x = []
        y = []
        for i in range(len(cent)):
            if np.absolute(cent[i,1] - np.median(cent[:,1])) < level:
                x.append(i*step)
                y.append(cent[i,1])
                
    
    pfit = np.polyfit(x, y, 1)                                                       # returns polynomial coefficients
    #yp = np.polyval(pfit, x)                                                         # fits the curve,
    
    return pfit








# =============================================================================
# Process classesm and functions to define ans store tomo-parameters 
# =============================================================================

#actually should be a part of the Processor class - check tomo_proc.py

def init_Npad(ROI, compression = 8):
    """Calculate the Npad for padding
    can be adjusted with compression parameter
    By default, 8 times smaller than ROI
    """
                    
    if (ROI[2]-ROI[0])>(ROI[3]-ROI[1]):
        Npad = (ROI[2]-ROI[0])//compression       
    else:
        Npad = (ROI[3]-ROI[1])//compression   
        
    return Npad 



def init_names_custom(data_name, distance_indexes):
    """set proper data_names when you manually define indexes for distances"""
    
    data_names = []
    ff_names = []
    
    for i in distance_indexes:
        data_names.append(data_name + '_' + str(i))
        ff_names.append('ff_' + data_name + '_' + str(i))
    
    return data_names, ff_names 
    

def rotate(input, angle, axes=(1, 0), reshape=True, output=None, order=3,
           mode='constant', cval=0.0, prefilter=True):
    """
    Rotate an array.

    The array is rotated in the plane defined by the two axes given by the
    `axes` parameter using spline interpolation of the requested order.

    Parameters
    ----------
    %(input)s
    angle : float
        The rotation angle in degrees.
    axes : tuple of 2 ints, optional
        The two axes that define the plane of rotation. Default is the first
        two axes.
    reshape : bool, optional
        If `reshape` is true, the output shape is adapted so that the input
        array is contained completely in the output. Default is True.
    %(output)s
    order : int, optional
        The order of the spline interpolation, default is 3.
        The order has to be in the range 0-5.
    %(mode_interp_constant)s
    %(cval)s
    %(prefilter)s

    Returns
    -------
    rotate : ndarray
        The rotated input.

    Notes
    -----
    For complex-valued `input`, this function rotates the real and imaginary
    components independently.

    .. versionadded:: 1.6.0
        Complex-valued support added.

    Examples
    --------
    >>> from scipy import ndimage, misc
    >>> import matplotlib.pyplot as plt
    >>> fig = plt.figure(figsize=(10, 3))
    >>> ax1, ax2, ax3 = fig.subplots(1, 3)
    >>> img = misc.ascent()
    >>> img_45 = ndimage.rotate(img, 45, reshape=False)
    >>> full_img_45 = ndimage.rotate(img, 45, reshape=True)
    >>> ax1.imshow(img, cmap='gray')
    >>> ax1.set_axis_off()
    >>> ax2.imshow(img_45, cmap='gray')
    >>> ax2.set_axis_off()
    >>> ax3.imshow(full_img_45, cmap='gray')
    >>> ax3.set_axis_off()
    >>> fig.set_tight_layout(True)
    >>> plt.show()
    >>> print(img.shape)
    (512, 512)
    >>> print(img_45.shape)
    (512, 512)
    >>> print(full_img_45.shape)
    (724, 724)

    """
    input_arr = input#np.asarray(input)
    ndim = input_arr.ndim

    if ndim < 2:
        raise ValueError('input array should be at least 2D')

    axes = list(axes)

    if len(axes) != 2:
        raise ValueError('axes should contain exactly two values')

    if not all([float(ax).is_integer() for ax in axes]):
        raise ValueError('axes should contain only integer values')

    if axes[0] < 0:
        axes[0] += ndim
    if axes[1] < 0:
        axes[1] += ndim
    if axes[0] < 0 or axes[1] < 0 or axes[0] >= ndim or axes[1] >= ndim:
        raise ValueError('invalid rotation plane specified')

    axes.sort()

    c, s = special.cosdg(angle), special.sindg(angle)

    rot_matrix = np.array([[c, s],
                              [-s, c]])

    img_shape = np.asarray(input_arr.shape)
    in_plane_shape = img_shape[axes]
    if reshape:
        # Compute transformed input bounds
        iy, ix = in_plane_shape
        out_bounds = rot_matrix @ [[0, 0, iy, iy],
                                   [0, ix, 0, ix]]
        # Compute the shape of the transformed input plane
        out_plane_shape = (out_bounds.ptp(axis=1) + 0.5).astype(int)
    else:
        out_plane_shape = img_shape[axes]

    out_center = rot_matrix @ ((out_plane_shape - 1) / 2)
    in_center = (in_plane_shape - 1) / 2
    offset = in_center - out_center

    output_shape = img_shape
    output_shape[axes] = out_plane_shape
    output_shape = tuple(output_shape)

    # complex_output = np.iscomplexobj(input_arr)
    # output = _ni_support._get_output(output, input_arr, shape=output_shape,
    #                                   complex_output=complex_output)

    output = da.zeros(output_shape)

    if ndim <= 2:
        affine_transform(input_arr, rot_matrix, offset=offset, output_shape=out_plane_shape,
                             order=order, mode=mode, cval=cval,prefilter=False)
    else:
        # If ndim > 2, the rotation is applied over all the planes
        # parallel to axes
        planes_coord = itertools.product(
            *[[slice(None)] if ax in axes else range(img_shape[ax])
              for ax in range(ndim)])

        out_plane_shape = tuple(out_plane_shape)

        for coordinates in planes_coord:
            ia = input_arr[coordinates]
            oa = output[coordinates]
            affine_transform(ia, rot_matrix, offset=offset, output_shape=out_plane_shape,
                             output=oa, order=order, mode=mode, cval=cval,prefilter=False)

    return output

# classes themselves

class Processor:
    __slots__ = ['ROI', 'ROI_ff', 'Npad', 'im_shape', 'images', 'flats',
                 'N_files', 'N_start']
        
    def __init__(self, ROI, folder, N_start, N_finish, compNpad = 8):
        """Initialize parameters. 
        Normally should contain ROI, N_distances, etc
        N_distances = tuple
            corresponds to the indexes of the distances
        """
        self.N_start = N_start
        self.ROI = ROI
        self.N_files = (N_finish - N_start) 
        self.im_shape = (ROI[3]-ROI[1], ROI[2]-ROI[0])  
        self.Npad = init_Npad(ROI, compression = compNpad)
        
        
    def init_paths(self, data_name, path, distance_indexes):
        """Generate paths images & flatfields"""
    
        #set data_names
        data_names, ff_names = init_names_custom(data_name = data_name,
                                                 distance_indexes = distance_indexes)
        
        #find images
        imlist = var.im_folder(path)
        
        # create a filter for unnecessary images
        tail = '_00000.tiff'
        
        if len(data_names[0])!=len(data_names[-1]):
            print("""
            WARNING! Different distances in your dataset 
            have different naming lengths. 
            File names can't be arranged. 
            Try to reduce the number of distances (<10) or modify the script.
            """)
            sys.exit()
        else:
            data_lencheck = len(data_names[0]+tail)
            ff_lencheck = len(ff_names[0]+tail)
        

        
        #set proper paths
        N_distances = len(distance_indexes) 
        images = np.zeros(N_distances, 'object') 
        flats = np.zeros(N_distances, 'object')
                
        for i in np.arange(len(images)):
            
            #sort image paths
            images[i] = [path+im for im in imlist 
                         if (im.startswith(data_names[i])) 
                         and not (im.startswith('.'))
                         and (len(im) == data_lencheck)]
            
            flats[i] = [path+im for im in imlist 
                        if im.startswith(ff_names[i])
                        and (len(im)==ff_lencheck)]
            
        self.images = images
        self.flats = flats

                        

class F:
    __slots__ = ['shape', 'dtype', 'shared_array_base']
    
    def __init__(self, shape, dtype = 'd'):
        """Create shared value array for processing.
        """
        self.shape = shape
        self.dtype = dtype
        
        ncell = int(np.prod(self.shape))
        self.shared_array_base = Array(dtype, ncell,lock=False)       
        pass
     
        
    

