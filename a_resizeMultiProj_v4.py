#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Holotomography module for resizing image
"""

import numpy as np
from skimage import io, transform
from PIL import Image
import b_alignMultiProj_v2
import warnings
from scipy.ndimage import gaussian_filter
import scipy 

def rescale(img, scale):
	'''
	Open, rescale, cut image with original width and height in pixel
		
	Parameters
	----------
	img : 2D-array
		projection image
	scale : float
		scaling factor to apply
			
	Returns
	-------
	im : 2D-array
		resized image
	'''
	 
	row = img.shape[0] #initial number of row	
	column = img.shape[1] #initial number of column	
	# ~ im = transform.rescale(img, scale, order=1, mode='reflect', preserve_range=1, anti_aliasing =0, multichannel=0) #rescale image
	# ~ im = transform.rescale(img, scale, order=1)#, mode='reflect', preserve_range=1, anti_aliasing =0, multichannel=0) #rescale image	
	im = np.array(Image.fromarray(img).resize(size=(int(column*scale), int(row*scale)), resample=Image.NEAREST))  #NEAREST #BOX
	im = im.astype(np.float32) #convert image to float32
	rowOffset = int((im.shape[0] - row)/2) #calculate number of row created and define offset to cut middle part of img only
	columnOffset = int((im.shape[1] - column)/2) #calculate number of column created created and define offset to cut middle part of img only
	im = im[rowOffset:row+rowOffset, columnOffset:column+columnOffset]
		
	return im



def readAndResize_task(img_path, scale):	
	img = np.asarray(Image.open(img_path))
	img = rescale(img,scale)
	return img



def removeOutlier(img, mask, sig):
	imgGaus = gaussian_filter(img, sigma=sig)
	imgCorr = np.copy(img)
	imgCorr[mask<1] = imgGaus[mask<1]
	return imgCorr



def readResizeShiftCrop_task(img_path, scale, shift, shiftMethod, crop_x, crop_y, outPath):
	img = np.asarray(Image.open(img_path))
	resized_img = rescale(img,scale)
		if shiftMethod = 'real':
		shifted_img = b_alignMultiProj_v2.shiftImage_real(resized_img, shift)
	elif shiftMethod = 'fft'
		shifted_img = b_alignMultiProj_v2.shiftImage_fft(resized_img, shift)
	cropped_img = shifted_img[int(crop_y[0]):int(crop_y[1]),int(crop_x[0]):int(crop_x[1])]
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		io.imsave(outPath+'.tif',cropped_img.astype('float32'))	


# TODO - work further on the implementation of the outlier correction
def readResizeOutShiftCrop_task(img_path, scale, mask, sig, shift, shiftMethod crop_x, crop_y, outPath):
	img = np.asarray(Image.open(img_path))
	resized_img = rescale(img, scale)
	resized_img = removeOutlier(resized_img, mask, sig)	
	if shiftMethod = 'real':
		shifted_img = b_alignMultiProj_v2.shiftImage_real(resized_img, shift)
	elif shiftMethod = 'fft'
		shifted_img = b_alignMultiProj_v2.shiftImage_fft(resized_img, shift)
	cropped_img = shifted_img[int(crop_y[0]):int(crop_y[1]),int(crop_x[0]):int(crop_x[1])]
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		io.imsave(outPath+'.tif',cropped_img.astype('float32'))
	
	
	
def saveImage(img, idx, outDir):
	with warnings.catch_warnings():
		warnings.simplefilter("ignore")
		io.imsave(outDir+os.sep+'attenRad_'+str(idx).zfill(4)+'.tif',img.astype('float32'))
		
