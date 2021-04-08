#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:18:47 2021

@author: schorb
"""



import numpy as np
import os
import sys



from maximus48 import var
from maximus48.tomo_proc3 import init_Npad, init_names_custom, F
from maximus48 import SSIM_131119 as SSIM_sf 
from maximus48.SSIM_131119 import SSIM_const
from maximus48 import multiCTF2 as multiCTF


from skimage.io import imread, imsave

# from pybdv import make_bdv 

# import dask
# import dask.array as da
from skimage.io import imread,imsave

from dask.distributed import Client, progress

from dask_jobqueue import SLURMCluster




#%% 


#-----------------------------------------


#         DASK 

# ----------------------------------------




client = Client('10.11.12.87:46558')

#%%


# =============================================================================
#           initial parameters for phase retrieval with CTF
# =============================================================================
N_steps = 10                                                                   # Number of projections per degree
N_start = 1                                                                    # index of the first file
N_finish = 3600                                                                # index of the last file

pixel = 0.1625 * 1e-6                                                          # pixel size 
distance = np.array((6.1, 6.5, 7.1, 8), dtype = 'float32') * 1e-2              # distances of your measurements 
energy = 18                                                                    # photon energy in keV
beta_delta = 0.15
zero_compensation = 0.05
ROI = (0,100,2048,2048) 
                                                      # ROI of the image to be read (x,y,x1,y1 at the image - inverse to numpy!)
cp_count = 60   
chunksz = 512

                                                               # number of cores for multiprocessing
inclination = -0.23                                                            # angle to compansate the tilt of the rotaxis

#data_name = 'ew21_5'
data_name = 'Platy-12601'
folder_base = '/scratch/schorb/HH_platy'
folder = os.path.join(folder_base,'raw/')
folder_temp = os.path.join(folder_base,'tmp/')
folder_result = os.path.join(folder_base,'rec/')
distances = (1,2,3,4)
N_distances  = 4  


#calculate parameters for phase-retrieval
wavelength = var.wavelen(energy)
fresnelN = pixel**2/(wavelength*distance)


#create save folder if it doesn't exist

if not os.path.exists(folder_temp):
    os.makedirs(folder_temp)


if not os.path.exists(folder_result):
    os.makedirs(folder_result)


# Variable structure:
            # ['N_files',
            #  'N_start',
            #  'Npad',
            #  'ROI',
            #  'ROI_ff',            
            #  'flats',
            #  'im_shape',
            #  'images',
            #  'init_paths']
            
            
Npad = init_Npad(ROI, compression = 8)

#%%

# Support functions
# =============================

def init_paths(data_name, path, distance_indexes):
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

    return images,flats


# =============================


## FLAT field correction

def flat_correct(j,images=[],flats=[],distances=(),ff_con=[]):
    filt = []
    for i in np.arange(len(images)):
        im = imread(images[i][j])[ROI[1]:ROI[3], ROI[0]:ROI[2]]
        maxcorridx=np.argmax(SSIM_sf.SSIM(SSIM_const(im[ROI_ff[1]:ROI_ff[3], ROI_ff[0]:ROI_ff[2]]), 
                                        ff_con[i]).ssim())
        
        filt.append(im/ff[i][maxcorridx])
    
    return filt

# =============================


## holographic reconstruction



def read_flat(j, images=[], ROI_ff=[], ROI=[],flats=[],distances=(),ff_con=[], N_start=0, Npad=0): 
    
    """
    j: int
        an index of the file that should be processed 
    Please note, j always starts from zero
    To open correct file, images array uses images[i][j + N_start-1]
    """
     
    #read image and do ff-retrieval    
    filt = flat_correct(j,images=images,flats=flats,distances=distances,ff_con=ff_con)
    
    im_gau0 = var.filt_gauss_laplace(filt[0][ROI_ff[1]:ROI_ff[3], ROI_ff[0]:ROI_ff[2]],
                                    sigma = 5)
    thisshift = []
    
    for i in range(len(filt)):
        im_gau1 = var.filt_gauss_laplace(filt[i][ROI_ff[1]:ROI_ff[3], ROI_ff[0]:ROI_ff[2]],
                                    sigma = 5)
        thisshift.append(var.shift_distance(im_gau0, im_gau1, 10))
    
    
    filt0 = multiCTF.shift_imageset(filt, thisshift)

    filt0 = np.pad(filt0, ((0,0),(Npad, Npad),(Npad, Npad)), 'edge')               # padding with border values
    filt0 = multiCTF.multi_distance_CTF(filt0, beta_delta, 
                                          fresnelN, zero_compensation)
    filt0 = filt0[Npad:(filt0.shape[0]-Npad),:]

    imsave(os.path.join(folder_temp,''.join(os.path.basename(images[0][j]).partition('_'+str(distances[0]))[0:3:2])))
    # pda = da.from_array(filt0)
    # da.to_zarr(pda,'/scratch/schorb/HH_platy/Platy-12601_'+str(j)+'.zarr')

    # return filt0




#%%


# =============================


# RUN  SCRIPT

# =============================


images, flats = init_paths(data_name, folder, distances)

im_shape = (ROI[3]-ROI[1], ROI[2]-ROI[0])

shape_ff = (N_distances, len(flats[0]), im_shape[0], im_shape[1])
ff_shared = F(shape = shape_ff, dtype = 'd')


#read ff-files to memory

ff = np.zeros(shape_ff)

for i in range(N_distances):
    for j,fname in enumerate(flats[i]):
        ff[i][j]=imread(fname)[ROI[1]:ROI[3], ROI[0]:ROI[2]]
        


#calculate ff-related constants
ROI_ff = (ff.shape[3]//4, ff.shape[2]//4,3 * ff.shape[3]//4, 3 * ff.shape[2]//4)    # make ROI for further flatfield and shift corrections, same logic as for normal ROI
ff_con = np.zeros(N_distances, 'object')                                                # array of classes to store flatfield-related constants
for i in np.arange(N_distances):    
    ff_con[i] = SSIM_const(ff[i][:,ROI_ff[1]:ROI_ff[3], 
                                   ROI_ff[0]:ROI_ff[2]].transpose(1,2,0))






#read_flat(j, images=images, ROI_ff=ROI_ff, ROI=ROI,flats=flats,distances=distances,ff_con=ff_con, N_start=N_start, Npad=Npad)

