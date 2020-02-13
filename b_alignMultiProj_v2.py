#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Holotomography module for aligning multi ATTEN projections
"""


from distutils.version import StrictVersion 
import numpy as np
import os
import glob
from natsort import natsorted
import joblib
from joblib import Parallel, delayed
import multiprocessing
import warnings
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, segmentation, color, filters, io, morphology, measure#, img_as_uint, img_as_float
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift
from sklearn import linear_model, datasets
import matplotlib
import multiprocessing
from scipy import stats


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

def shiftImage(img, shift):	
	offset_img = fourier_shift(np.fft.fftn(img), shift)
	offset_img = np.fft.ifftn(offset_img)
	offset_img = np.real(offset_img)
	offset_img = offset_img.astype(np.float32)	
	return offset_img

def crossCorrelateImage(img1, img2):
	shift, error, diffphase = register_translation(img1, img2, 20)
	return shift

def fit_RANSAC(x, y):		
	#calculate shift
	yy = y
	xx = x
	X = xx.reshape((len(xx), 1))
	y = yy
	# Fit line using all data
	lr = linear_model.LinearRegression()
	lr.fit(X, y)
	# Robustly fit linear model with RANSAC algorithm
	ransac = linear_model.RANSACRegressor()
	# ~ ransac.fit(X, y)
	try:
		ransacError = 0
		ransac.fit(X, y)			
	except ValueError:
		print('RANSAC error')
		ransacError = 1				
	if ransacError == 1:
		inlier_mask = len(x)*[True]
		outlier_mask = np.logical_not(inlier_mask)
		# Predict data of estimated models
		line_X = np.arange(X.min(), X.max())[:, np.newaxis]
		line_y = lr.predict(line_X)
		line_y_ransac = line_y		
		slope, intercept, r_value, p_value, std_err = stats.linregress(line_X[:,0],line_y[:])
		shiftCorr = x * slope + intercept
	else:
		inlier_mask = ransac.inlier_mask_
		outlier_mask = np.logical_not(inlier_mask)
		# Predict data of estimated models
		line_X = np.arange(X.min(), X.max())[:, np.newaxis]
		line_y = lr.predict(line_X)
		line_y_ransac = ransac.predict(line_X)
		slope = ransac.estimator_.coef_[0]
		intercept = ransac.estimator_.intercept_	
		shiftCorr = x * slope + intercept
	
	
	# ~ '''plot linear RANSAC fitting results '''
	# ~ lw = 2
	# ~ plt.scatter(x[inlier_mask], y[inlier_mask], color='violet', marker='o',
				# ~ label='y-Inliers')
	# ~ plt.scatter(x[outlier_mask], y[outlier_mask], color='lightcoral', marker='o',
				# ~ label='y-Outliers')
	# ~ plt.plot(line_X, line_y, color='darkorchid', linewidth=lw, label='y-Linear reg')
	# ~ plt.plot(line_X, line_y_ransac, color='pink', linewidth=lw,
			 # ~ label='y-RANSAC reg')
	# ~ plt.show()
	
	# ~ return shiftCorr
	return slope, intercept
