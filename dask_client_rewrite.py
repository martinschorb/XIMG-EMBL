#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  7 09:18:47 2021

@author: schorb
"""



import numpy as np
import os
import sys
import time
import glob

import gc


from maximus48 import var
from maximus48.tomo_proc3 import init_Npad, init_names_custom, F,rotscan
from maximus48 import SSIM_131119 as SSIM_sf 
from maximus48.SSIM_131119 import SSIM_const
from maximus48 import multiCTF2 as multiCTF


from maximus48.tomo_proc3_parallel import rotaxis_rough


from skimage.io import imread, imsave
from scipy.ndimage import rotate

from pybdv import make_bdv 

from dask import delayed
# import dask.array as da

from dask.distributed import Client, progress

# from dask_jobqueue import SLURMCluster

import tomopy


#%% 


#-----------------------------------------


#         DASK 

# ----------------------------------------




# client = Client('10.11.12.87:33797')

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


## holographic reconstruction





def read_flat(j,images=[], ROI_ff=[], ROI=[],flats=[],ff_file='',ffcon_file='',distances=(), N_start=0, Npad=0):
    """
    j: int
        an index of the file that should be processed 
    Please note, j always starts from zero
    To open correct file, images array uses images[i][j + N_start-1]
    """
    
        
        
    ff_con = np.load(ffcon_file,allow_pickle=True)
    
    ff = np.load(ff_file,allow_pickle=True)
   
    #read image and do ff-retrieval 
    
    # =============================


    
    ## FLAT field correction
    
    
    filt = []
    
    for i in np.arange(len(images)):
        im = imread(images[i][j])[ROI[1]:ROI[3], ROI[0]:ROI[2]]
     
        maxcorridx=np.argmax(SSIM_sf.SSIM(SSIM_const(im[ROI_ff[1]:ROI_ff[3], ROI_ff[0]:ROI_ff[2]]),ff_con[i]).ssim())        
        filt.append(im/ff[i][maxcorridx])

        
        
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

    imsave(os.path.join(folder_temp,''.join(os.path.basename(images[0][j]).partition('_'+str(distances[0]))[0:3:2])),filt0)
    # pda = da.from_array(filt0)
    # da.to_zarr(pda,'/scratch/schorb/HH_platy/Platy-12601_'+str(j)+'.zarr')

    return 'done processing image '+str(j)




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

ffcon_file = folder_temp+'ffcon.npy'
np.save(ffcon_file,ff_con)



ff_file = folder_temp+'ff.npy'
np.save(ff_file,ff)


#read_flat(j,images=images, ROI_ff=ROI_ff, ROI=ROI,flats=flats,distances=distances,ffcon_file=ffcon_file,ff_file=ff_file, N_start=N_start, Npad=Npad)

#%%
# s1=client.map....

status = 'p'

while status != 'done':
    for st in s1:
        
        if st.status in ['error']:
            print('retrying '+st.key)
            
            st.retry()
            status = 'p'
            time.sleep(1)
        elif st.status in ['finished']:
            status = 'done'
    
    

#%%

import dask.array as da

from dask.array.image import imread

cp_count = 96


imfiles = sorted(glob.glob(folder_temp+'*.tiff'))

im = imread(imfiles[0])

proj = np.zeros((3600,im.shape[0],im.shape[1]))

for idx,imf in enumerate(imfiles):proj[idx,:]=imread(imf)

print('stripe removal\n\n================================\n\n')

proj = tomopy.prep.stripe.remove_stripe_fw(proj,level=3, wname=u'db25', sigma=2, pad = False,ncore=cp_count,nchunk=chunksz)

pshape = proj.shape

stripe_file = folder_temp+'/stripe.npy'
np.save(stripe_file,proj)


def parallel_rotate(instack,i):
    im = rotate(instack[i,:], inclination, mode='nearest')
    imsave(os.path.join(folder_temp,'rotated_'+data_name+'_'+str(i).zfill(4)+'.tif'),im)


projd = da.from_array(np.memmap(folder_temp+'stripe.npy',shape=pshape,mode='r'))



print('rotate\n\n================================\n\n')
proj = rotate(proj, inclination, mode='nearest', axes=(2,1))

pshape = proj.shape

rot_file = folder_temp+'/rotate.npy'
np.save(rot_file,proj)


os.environ["TOMOPY_PYTHON_THREADS"]=str(cp_count)

print('rotscan\n\n================================\n\n')
# cent = rotscan(proj, N_steps,ncore=cp_count,nchunk=chunksz)


# rotscan function....


cent = rotaxis_rough(proj, N_steps)
cent = np.median(cent)


rotsc_file = folder_temp+'/rotscan.npy'
np.save(rotsc_file,cent)


n = proj.shape[0]

angle = np.pi*np.arange(n)/(N_steps*180)

time1 = time.time()


print('reconstruct\n\n================================\n\n')

outs = tomopy.recon(proj, angle, center = cent, algorithm = 'gridrec', filter_name = 'shepp',ncore=cp_count,nchunk=chunksz)

outs = outs[:,Npad : outs.shape[1]- Npad,Npad : outs.shape[2]- Npad]


    
data = outs
data -= data.min()
data /= data.max()
data *= 32767
data = data.astype('int16')
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


# t=dask.delayed(make_bdv)(data, folder_h5 + data_name, downscale_factors=scale_factors,
#     ...:                  downscale_mode=mode, resolution=resolution,
#     ...:                  unit='micrometer', setup_name = data_name)    

# AssertionError: daemonic processes are not allowed to have children

make_bdv(data, folder_h5 + data_name, downscale_factors=scale_factors,
                 downscale_mode=mode, resolution=resolution,
                 unit='micrometer', setup_name = data_name,n_threads=cp_count) 
