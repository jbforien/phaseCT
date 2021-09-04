import sys
sys.path.insert(0, r"your path to phase retrieval directory here")

import a_resizeMultiProj_v4
import b_alignMultiProj_v2
import c_phaseRetrieval_v3
from distutils.version import StrictVersion 
import numpy as np
import os
import glob
from natsort import natsorted
from skimage import io
import joblib
from joblib import Parallel, delayed
import multiprocessing
import warnings
from PIL import Image
import time
import datetime
import matplotlib.pyplot as plt
from tqdm import tqdm

#check if joblib version package is at least 0.13.1. Otherwise following error will pop-up:
'''
ImportError: [joblib] Attempting to do parallel computing without protecting your import on a system that does not support forking.
To use parallel-computing in a script, you must protect you main loop using "if __name__ == '__main__'". 
Please see the joblib documentation on Parallel for more information
'''
if StrictVersion(joblib.__version__)<StrictVersion("0.13.1"):
    print("please update joblib to version 0.13.1")
    print("> pip install joblib --upgrade")
    exit()

########################################################################
########################## MAKE CHANGES BELOW ##########################
########################################################################

inputDir = r"main directory path"
pathProj1 = inputDir + os.sep + 'folder_1'
pathProj2 = inputDir + os.sep + 'folder_2'
pathProj3 = inputDir + os.sep + 'folder_3'
pathProj4 = inputDir + os.sep + 'folder_4'
listPathProj = [pathProj1, pathProj2, pathProj3, pathProj4]


#adapt experimental setup parameters
'''
Parameter of the experiment
'''
sod = 21000                                             # source to detector distance in mm
dists = [15,40,86,132]                                    # object to detector distance in mm
energy = 35                                                # in keV
pix_width = 0.65e-3                                     # in mm
preProcessingInterval = 100
finalInterval = 1

# add one mask per dataset
# further work needed
removeOutlier = 0
outliersMaskPath1 = inputDir + os.sep + 'folder_1' + os.sep + '000000_mask.tif'
outliersMaskPath2 = inputDir + os.sep + 'folder_2' + os.sep + '000000_mask.tif'
outliersMaskPath3 = inputDir + os.sep + 'folder_3' + os.sep + '000000_mask.tif'
outliersMaskPath4 = inputDir + os.sep + 'folder_4' + os.sep + '000000_mask.tif'
outliersMaskPath5 = inputDir + os.sep + 'folder_5' + os.sep + '000000_mask.tif'
outliersMaskPath6 = inputDir + os.sep + 'folder_6' + os.sep + '000000_mask.tif'
listOutliersMaskPath = [outliersMaskPath1, outliersMaskPath2, outliersMaskPath3, outliersMaskPath4, outliersMaskPath5, outliersMaskPath6]
gaussFltrSig = 7

performPaganin = 1                                        # enhance constrast with Paganin
delta = 1.40904535E-06  
beta= 5.39533251E-10
unsharpMask = 0                                            # perform unsharp mask
gaussianBlur = 0; sigmaBlur = 1                            # perform gaussian blur

registration_algorithm = 'crossCorr_imreg_dft'            # crossCorr_imreg_dft, crossCorr_skimage_fourier, crossCorr_skimage_real, mutualInfo_dipy

fitting_method = 'meanNoOutliers'                        # lin_RANSAC, polynomial, meanNoOutliers
polynomial_order = 0                                    # choose polynomial order, not used for 'lin_RANSAC'
plotShift = 0

shiftMethod = 'real'                                    # choose shift method - 'real' or 'fft'

nbCPU = 0.75                                             # Nbr of CPU to use in percent

########################################################################
################# NO NEED TO CHANGE BELOW THESE LINES ##################
########################################################################
now = datetime.datetime.now()
now = now.strftime('%y%m%d_%H%M%S')

dists = np.asarray(dists)
mag =(sod+dists)/sod
maxMag = np.amax(mag)

pix_width = pix_width/maxMag
wlen = c_phaseRetrieval_v3.Energy2Wavelength(energy)

totalNbrCores = multiprocessing.cpu_count()                # find the number of cpu available
nbCores = int(totalNbrCores*nbCPU)                        # define the number of cpu that will be used


'''
Create array with path of projections
'''
print("Create list of projections")
listFile = []
for i in range(0,len(listPathProj)): #loop through each folder    
    listFile_temp = glob.glob(listPathProj[i]+os.sep+"*.tif")    
    listFile_temp = natsorted(listFile_temp) #natural sorting of the files        
    listFile.append(listFile_temp)

'''
pre-alignment of image - align limited number of projections and retrieve
shifts
'''
print('aligned limited number of projections')    
refImg_collection = Parallel(n_jobs=nbCores)(delayed(a_resizeMultiProj_v4.readAndResize_task)(listFile[0][j], 
                        maxMag/((sod+dists[0])/sod)) for j in tqdm(range(0, len(listFile[0]),preProcessingInterval)))
refImg_collection = np.stack(refImg_collection, axis=0)


''' correct for foreground outliers '''
if removeOutlier == 1:
    outlierMask_collection = Parallel(n_jobs=nbCores)(delayed(a_resizeMultiProj_v4.readAndResize_task)(listOutliersMaskPath[j], 
                        maxMag/((sod+dists[0])/sod)) for j in tqdm(range(0, listOutliersMaskPath)))
    outlierMask_collection = np.stack(outlierMask_collection, axis=0)
        
    corrImg_collection = Parallel(n_jobs=nbCores)(delayed(a_resizeMultiProj_v4.removeOutlier)(refImg_collection[j], 
                        outlierMask_collection[0,:,:], gaussFltrSig) for j in tqdm(range(0, refImg_collection.shape[0],1)))
    refImg_collection = np.stack(corrImg_collection, axis=0)


''' convert to Paganin if option selected '''
if performPaganin == 1:
    fx = np.fft.fftfreq(refImg_collection.shape[2],d=pix_width)
    fy = np.fft.fftfreq(refImg_collection.shape[1],d=pix_width)
    fx, fy = np.meshgrid(fx,fy)
    phase = Parallel(n_jobs=nbCores)(delayed(c_phaseRetrieval_v3.Paganin)(np.exp(-1*refImg_collection[j]),
                wlen, dists[i],  delta, beta, fx, fy, Rm=0, alpha=0) for j in tqdm(range(0, refImg_collection.shape[0])))
    refImg_collection = np.stack(phase, axis=0)


''' perform unsharp mask'''
if unsharpMask == 1:
    from skimage.filters import unsharp_mask
    unsharp = Parallel(n_jobs=nbCores)(delayed(unsharp_mask)(refImg_collection[j], radius=5, amount=2)
                                        for j in tqdm(range(0, refImg_collection.shape[0])))
    refImg_collection = np.stack(unsharp, axis=0)
    
''' perform gaussian blur'''
if gaussianBlur == 1:
    from skimage.filters import gaussian
    unsharp = Parallel(n_jobs=nbCores)(delayed(gaussian)(refImg_collection[j], sigma=sigmaBlur)
                                        for j in tqdm(range(0, refImg_collection.shape[0])))
    refImg_collection = np.stack(unsharp, axis=0)


''' read, resize, and register target images '''
''' optional - outlier corrections and Paganin convertion '''
shiftArr = []
for i in range(1,len(listFile)): #loop through each stack of projections
    ''' read and resize images '''
    targetImg_collection = Parallel(n_jobs=nbCores)(delayed(a_resizeMultiProj_v4.readAndResize_task)(listFile[i][j], 
                        maxMag/((sod+dists[i])/sod)) for j in tqdm(range(0, len(listFile[i]),preProcessingInterval)))
    targetImg_collection = np.stack(targetImg_collection, axis=0)

    ''' correct outlier if option selected '''
    if removeOutlier == 1: 
        corrImg_collection = Parallel(n_jobs=nbCores)(delayed(a_resizeMultiProj_v4.removeOutlier)(targetImg_collection[j], 
                    outlierMask_collection[i,:,:], gaussFltrSig) for j in tqdm(range(0, targetImg_collection.shape[0],1)))
        targetImg_collection = np.stack(corrImg_collection, axis=0)
    
    ''' convert to Paganin if option selected '''
    if performPaganin == 1:
        phase = Parallel(n_jobs=nbCores)(delayed(c_phaseRetrieval_v3.Paganin)(np.exp(-1*targetImg_collection[j]),
                    wlen, dists[i],  delta, beta, fx, fy, Rm=0, alpha=0) for j in tqdm(range(0, targetImg_collection.shape[0],1)))    
        targetImg_collection = np.stack(phase, axis=0)

    ''' perform unsharp mask'''
    if unsharpMask == 1:
        from skimage.filters import unsharp_mask
        unsharp = Parallel(n_jobs=nbCores)(delayed(unsharp_mask)(targetImg_collection[j], radius=5, amount=2)
                                            for j in tqdm(range(0, targetImg_collection.shape[0],1)))     
        targetImg_collection = np.stack(phase, axis=0)
        
    ''' perform gaussian blur'''
    if gaussianBlur == 1:
        from skimage.filters import gaussian
        unsharp = Parallel(n_jobs=nbCores)(delayed(gaussian)(targetImg_collection[j], sigma=1)
                                            for j in tqdm(range(0, targetImg_collection.shape[0],1))) 
        targetImg_collection = np.stack(unsharp, axis=0)

    ''' perform registration'''
    shift = []    
    if registration_algorithm == 'crossCorr_imreg_dft':    
        shift = Parallel(n_jobs=nbCores)(delayed(b_alignMultiProj_v2.crossCorr_imreg_dft)
                        (refImg_collection[j], targetImg_collection[j]) 
                        for j in tqdm(range(0, refImg_collection.shape[0],1)))
                        
    
    elif registration_algorithm == 'crossCorr_skimage_fourier':
        shift = Parallel(n_jobs=nbCores)(delayed(b_alignMultiProj_v2.crossCorr_skimage_fourier)
                        (refImg_collection[j], targetImg_collection[j]) 
                        for j in tqdm(range(0, refImg_collection.shape[0],1)))
        
    elif registration_algorithm == 'crossCorr_skimage_real':
        shift = Parallel(n_jobs=nbCores)(delayed(b_alignMultiProj_v2.crossCorr_skimage_real)
                        (refImg_collection[j], targetImg_collection[j]) 
                        for j in tqdm(range(0, refImg_collection.shape[0],1)))
                        
    elif registration_algorithm == 'mutualInfo_dipy':
        shift = Parallel(n_jobs=nbCores)(delayed(b_alignMultiProj_v2.mutualInfo_dipy)
                        (refImg_collection[j], targetImg_collection[j]) 
                        for j in tqdm(range(0, refImg_collection.shape[0],1)))

    shift = np.stack(shift, axis=0) #Convert list to array                                        

    shiftArr.append(shift) #Add shift to list
    
shiftArr = np.stack(shiftArr,axis=0) #array with [nbr propagation distance, nbr projections, [dim x, dim y]]


'''
Fit shifts with polynomial or linear ransac fitting
'''
print('alignment - shift fitting')
x = np.arange(0,len(listFile[0]), preProcessingInterval)
xProj = np.arange(0,len(listFile[0]))
shift = [np.zeros((len(xProj),2))]
for i in range(0,shiftArr.shape[0]): #loop through each stack of projection minus reference stack        
    
    if fitting_method == 'polynomial':        
        shift_x = np.polyfit(x, shiftArr[i,:,0], polynomial_order)
        p_x = np.poly1d(shift_x)
        shift_y = np.polyfit(x, shiftArr[i,:,1], polynomial_order)
        p_y = np.poly1d(shift_y)
        shift.append(np.column_stack((p_y(xProj),p_x(xProj))))
        
    elif fitting_method == 'lin_RANSAC':
        shift_x = b_alignMultiProj_v2.lin_RANSAC(x, shiftArr[i,:,0])
        p_x = np.poly1d(shift_x)
        shift_y = b_alignMultiProj_v2.lin_RANSAC(x, shiftArr[i,:,1])
        p_y = np.poly1d(shift_y)
        shift.append(np.column_stack((p_y(xProj),p_x(xProj))))
        
    elif fitting_method == 'meanNoOutliers':
        shift_x = b_alignMultiProj_v2.mean_witout_outliers(x, shiftArr[i,:,0])
        shift_x = np.median(shiftArr[i,:,0])
        p_x = np.poly1d(shift_x)
        shift_y = b_alignMultiProj_v2.mean_witout_outliers(x, shiftArr[i,:,1])
        shift_y = np.median(shiftArr[i,:,1])
        p_y = np.poly1d(shift_y)
        shift.append(np.column_stack((p_y(xProj),p_x(xProj))))
    
    if plotShift == 1:
        if fitting_method == 'polynomial':
            plt.suptitle(fitting_method+' order '+str(int(polynomial_order))
                            +'\ndistance: ' + str(int(i+1)))
        else: 
            plt.suptitle(fitting_method+'\ndistance: ' + str(int(i+1)))
        plt.plot(x, shiftArr[i,:,0], 'o', label='x-shift data')
        plt.plot(xProj, p_x(xProj), '-', label='x-shift fit')
        plt.plot(x, shiftArr[i,:,1], 'o', label='y-shift data')
        plt.plot(xProj, p_y(xProj), '-', label='y-shift fit')
        plt.xlabel('projections [#]')
        plt.ylabel('shift [pxl]')
        plt.legend()
        plt.show()
        plt.close('all')
    


'''
calculate crop margins projections
'''
print('calculate crop margins')
offset = []
for i in range(0, len(shift)):
    offset.append([np.amin(shift[i][:,0]), np.amin(shift[i][:,1])])
    offset.append([np.amax(shift[i][:,0]), np.amax(shift[i][:,1])])
offset = np.asarray(offset)

crop_y = np.array([np.floor(abs(np.amax(offset[:,0]))), 
                            np.ceil(refImg_collection.shape[1] - abs(np.amin(offset[:,0])))])
crop_x = np.array([np.floor(abs(np.amax(offset[:,1]))),
                            np.ceil(refImg_collection.shape[2] - abs(np.amin(offset[:,1])))])


'''
process projections
'''
print('read, resize, shift, crop and save all projections')
for i in range(0,len(listFile)): #loop through each stack of projection minus reference stack
    print('\t' + str(os.path.dirname(listPathProj[i])))

    scale = maxMag/((sod+dists[i])/sod)            
    ouputDir = os.path.dirname(listPathProj[i]) + os.sep + "RESIZE_ALIGN"
    check_dir = os.path.isdir(ouputDir)
    if check_dir==False:
        os.mkdir(ouputDir)
    outPath = [(str(ouputDir)+os.sep+"attenRad_"+str(n).zfill(4)) for n in range(len(listFile[0]))]    
    
    ''' correct outlier if option selected '''
    if removeOutlier == 1: 
        Parallel(n_jobs=nbCores)(delayed(a_resizeMultiProj_v4.readResizeOutShiftCrop_task)(listFile[i][j], scale, 
                                                outlierMask_collection[i,:,:], gaussFltrSig, 
                                                shift[i][j], shiftMethod, 
                                                crop_x, crop_y, outPath[j]) 
                                                for j in tqdm(range(0, len(listFile[0]),finalInterval)))

    else:
        Parallel(n_jobs=nbCores)(delayed(a_resizeMultiProj_v4.readResizeShiftCrop_task)(listFile[i][j], scale,
                                                shift[i][j], shiftMethod,
                                                crop_x, crop_y, outPath[j]) 
                                                for j in tqdm(range(0, len(listFile[0]),finalInterval)))
    
