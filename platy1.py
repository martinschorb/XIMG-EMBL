#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 22 16:34:02 2019

@author: mpolikarpov

"""
import os
os.environ['OMP_NUM_THREADS'] ='1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
#os.system('taskset -cp 0-100 %d' % os.getpid())

import sys, time
import dxchange, tomopy
import numpy as np
import tifffile
from scipy.ndimage import rotate
from contextlib import closing
from multiprocessing import Pool
import gc

from maximus48 import var
from maximus48 import SSIM_131119 as SSIM_sf 
from maximus48 import multiCTF2 as multiCTF
from maximus48.SSIM_131119 import SSIM_const 
from maximus48.tomo_proc3 import (Processor, F, tonumpyarray, rotscan)
from maximus48 import FSC

from pybdv import make_bdv 


#import matplotlib.pyplot as plt
#from maximus48 import monochromaticCTF as CTF 
#import h5py
#from maximus48.tomo_proc3 import axis_raws, interpolate

folder = '/g/emcf/schorb/data/HH_platy/raw/'
folder_result = '/g/emcf/schorb/data/HH_platy/rec1/'
folder_proj = '/scratch/schorb/HH_platy/'
data_name = 'Platy-12601'

file = folder + data_name + '_1_00001.tiff'
#ROI = (100,100,1900,1900)
ROI = (0,100,2048,2048)

distances = (1,2,3,4)
N_distances  = 4      


image = var.read_image(file, ROI = ROI)
# var.show(image)

# =============================================================================
#  initialize parameters for phase retrieval with CTF
# =============================================================================
N_steps = 10                                                                   # Number of projections per degree
N_start = 1                                                                    # index of the first file
N_finish = 3600                                                                # index of the last file

pixel = 0.1625 * 1e-6                                                          # pixel size 
distance = np.array((6.1, 6.5, 7.1, 8), dtype = 'float32') * 1e-2              # distances of your measurements 
energy = 18                                                                    # photon energy in keV
beta_delta = 0.17
zero_compensation = 0.05
#ROI = (500,300,1500,1300)                                                      # ROI of the image to be read (x,y,x1,y1 at the image - inverse to numpy!)
cp_count = 63                                                                 # number of cores for multiprocessing
inclination = -0.23    

wavelength = var.wavelen(energy)
fresnelN = pixel**2/(wavelength*distance)




#create save folder if it doesn't exist
if not os.path.exists(folder_result):
    os.makedirs(folder_result)

if not os.path.exists(folder_proj):
    os.makedirs(folder_proj) 

# create a class to store all necessary parameters for parallelization
Pro = Processor(ROI, folder, N_start, N_finish, compNpad = 256)                 

#set proper paths
Pro.init_paths(data_name, folder, distances) 

#allocate memory to store flatfield
shape_ff = (N_distances, len(Pro.flats[0]), Pro.im_shape[0], Pro.im_shape[1]) 
ff_shared = F(shape = shape_ff, dtype = 'd')

#read ff-files to memory
ff = tonumpyarray(ff_shared.shared_array_base, ff_shared.shape, ff_shared.dtype)
for i in range(N_distances):
    ff[i] = tifffile.imread(Pro.flats[i])[:,ROI[1]:ROI[3], ROI[0]:ROI[2]] 

#calculate ff-related constants
Pro.ROI_ff = (ff.shape[3]//4, ff.shape[2]//4,3 * ff.shape[3]//4, 3 * ff.shape[2]//4)    # make ROI for further flatfield and shift corrections, same logic as for normal ROI
ff_con = np.zeros(N_distances, 'object')                                                # array of classes to store flatfield-related constants
for i in np.arange(N_distances):    
    ff_con[i] = SSIM_const(ff[i][:,Pro.ROI_ff[1]:Pro.ROI_ff[3], 
                                   Pro.ROI_ff[0]:Pro.ROI_ff[2]].transpose(1,2,0))


#allocate memory to store ff-indexes
indexes = F(shape = (N_finish - N_start, N_distances), dtype = 'i' )

#allocate memory to store shifts
shifts = F(shape = (N_finish - N_start, N_distances, 2), dtype = 'd')

#allocate memory to store filtered files
proj = F(shape = (Pro.N_files, shape_ff[2], shape_ff[3] + 2*Pro.Npad), dtype = 'd' )

#print('finished calculation of ff-constants and memory allocation in ', time.time()-time1)



# =============================================================================
# =============================================================================
# # Processing module
# =============================================================================
# =============================================================================

# =============================================================================
# functions for parallel processing 
# =============================================================================

def init():
    global Pro, ff_shared, ff_con, proj


def read_flat(j):    
    """
    j: int
        an index of the file that should be processed 
    Please note, j always starts from zero
    To open correct file, images array uses images[i][j + N_start-1]
    """
    
    #global ff_shared, ff_con, shifts, indexes, Pro
    
    #set local variables
    ff = tonumpyarray(ff_shared.shared_array_base, ff_shared.shape, ff_shared.dtype)
    proj_loc = tonumpyarray(proj.shared_array_base, proj.shape, proj.dtype)
    shift = tonumpyarray(shifts.shared_array_base, shifts.shape, shifts.dtype)
    #LoI = tonumpyarray(indexes.shared_array_base, indexes.shape, indexes.dtype)
    
    ROI_ff = Pro.ROI_ff
    ROI = Pro.ROI
    images = Pro.images
    N_start = Pro.N_start
    Npad = Pro.Npad
    
    #read image and do ff-retrieval    
    filt = []
    for i in np.arange(len(images)):
        im = tifffile.imread(images[i][j + N_start-1])[ROI[1]:ROI[3], ROI[0]:ROI[2]]
        
        index = SSIM_sf.SSIM(SSIM_const(im[ROI_ff[1]:ROI_ff[3], ROI_ff[0]:ROI_ff[2]]), 
                                        ff_con[i]).ssim()
        
        im = im/ff[i][np.argmax(index)]
        filt.append(im)
        #LoI[j,i] = np.argmax(index)


    #calculate shift for holograms
    im_gau0 = var.filt_gauss_laplace(filt[0][ROI_ff[1]:ROI_ff[3], ROI_ff[0]:ROI_ff[2]],
                                    sigma = 5)
    for i in np.arange(len(filt)):
        im_gau1 = var.filt_gauss_laplace(filt[i][ROI_ff[1]:ROI_ff[3], ROI_ff[0]:ROI_ff[2]],
                                    sigma = 5)
        shift[j,i] = (var.shift_distance(im_gau0, im_gau1, 10))


    #shift images
    filt = multiCTF.shift_imageset(np.asarray(filt), shift[j])
    filt = np.asarray(filt)


    #do CTF retrieval 
    filt = np.pad(filt, ((0,0),(Npad, Npad),(Npad, Npad)), 'edge')               # padding with border values
    filt = multiCTF.multi_distance_CTF(filt, beta_delta, 
                                           fresnelN, zero_compensation)
    filt = filt[Npad:(filt.shape[0]-Npad),:]                                     # unpad images from the top
    
    #rotate the image to compensate for the inclined rotation axis
    #filt = rotate(filt, inclination, mode = 'nearest', axes = (1,0))
    
    #save to memory
    proj_loc[j] = filt    
    #print('sucessfully processed file: ', images[0][j + N_start-1])                                   # unpad images from the top
        

# =============================================================================
# Process projections
# =============================================================================
    
#do phase retrieval
time1 = time.time()
with closing(Pool(cp_count, initializer = init)) as pool:    
    pool.map(read_flat, np.arange(Pro.N_files))
print('time for ff+shifts: ', time.time()-time1)

proj = tonumpyarray(proj.shared_array_base, proj.shape, proj.dtype)

np.save(folder_proj + data_name +  '_proj.npy', proj)


#remove vertical stripes with wavelet-fourier filtering
time1 = time.time()            
proj = tomopy.prep.stripe.remove_stripe_fw(proj,level=3, wname=u'db25', sigma=2, pad = False,ncore=cp_count)
print('time for stripe removal ', time.time()-time1)

np.save(folder_proj + data_name +  '_stripes.npy', proj)


ff = None
ff_con = None
ff_shared = None
indexes = None
gc.collect()



# =============================================================================
#  find rotation axis
# =============================================================================

# scan the original array to find the inclination
#cent, inclination = rotscan(proj, N_steps)

#rotate array to compensate for the tilt
proj = rotate(proj, inclination, mode='nearest', axes=(2,1))

gc.collect()

# scan finally
cent, inclination2 = rotscan(proj, N_steps)

print('rotscan done')

# =============================================================================
# # save what you need and release memory

if not os.path.exists(folder_proj):
    os.makedirs(folder_proj) 

shifts_2_save = tonumpyarray(shifts.shared_array_base, shifts.shape, shifts.dtype)
np.save(folder_proj + data_name +  '_rotate.npy', proj)
np.save(folder_proj + data_name +  '_shifts.npy', shifts_2_save)
# =============================================================================




# =============================================================================
#  tomo reconstruction 
# =============================================================================
# 1st reconstruction - all files
n = proj.shape[0]
angle = np.pi*np.arange(n)/(N_steps*180)

time1 = time.time()
outs = tomopy.recon(proj, angle, center = cent, algorithm = 'gridrec', filter_name = 'shepp',ncore=cp_count)
#print('time for tomo_recon ', time.time()-time1)

#crop
outs = outs[:,Pro.Npad : outs.shape[1]- Pro.Npad,Pro.Npad : outs.shape[2]- Pro.Npad]

#crop additionally
#outs = outs[:,270:840, 125:1020]
    
np.save(folder_proj + data_name +  '_rec.npy', outs)

print('reconstruction done')

# =============================================================================
# save as h5    
# =============================================================================
#cast
data = outs
data -= data.min()
data /= data.max()
data *= 32767
data = data.astype('int16')
gc.collect()

# release memory
outs = None
gc.collect() 

# set the factors for downscaling, for example 2 times isotropic downsampling by a factor of 2
scale_factors = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]] 

# set the downsampling mode, 'mean' is good for image data, for binary data or labels
# use 'nearest' instead
mode = 'interpolate'

# resolution of the data, set appropriately
resolution = [pixel*1e6 , pixel*1e6 , pixel*1e6] 

#save big data format
folder_h5 = folder_result + 'bdv/'

if not os.path.exists(folder_h5):
    os.makedirs(folder_h5) 
    
make_bdv(data, folder_h5 + data_name, downscale_factors=scale_factors,
                 downscale_mode=mode, resolution=resolution,
                 unit='micrometer', setup_name = data_name) 

# =============================================================================
# save as tiff
# =============================================================================
folder_tiff = folder_result + 'tiff/'

if not os.path.exists(folder_tiff):
    os.makedirs(folder_tiff)

dxchange.write_tiff_stack(data, fname= folder_tiff + data_name + '/tomo')




# =============================================================================
# save all parameters to the txt file
# =============================================================================
folder_param = folder_result + 'parameters/' 
if not os.path.exists(folder_param):
    os.makedirs(folder_param) 
os.mknod(folder_param + data_name + '_parameters.txt')
with open(folder_param + data_name + '_parameters.txt', 'w') as f:
    f.write(time.ctime() + '\n')
    f.write("data_path = %s\n" % folder)
    f.write("ROI =  %s\n" % str(ROI))
    f.write("pixel size = %s\n" %str(pixel))
    f.write("distances = %s\n" %str(distance))
    f.write("energy = %s\n" %str(energy))
    f.write("beta_delta = %s\n" %str(beta_delta))
    f.write("fresnel Number = %s\n" %str(fresnelN))
    f.write("zero_compensation = %s\n" %str(zero_compensation))
    f.write("Npad = %s\n" %str(Pro.Npad))
    f.write("center of rotation = %s\n" %str(cent))
    f.write("projections per degree = %s\n" %str(N_steps))
    f.write("inclination of rotaxis = %s\n" %str(inclination))
   
    
    
    






