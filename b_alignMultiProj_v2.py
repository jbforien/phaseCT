#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Holotomography module for aligning multi ATTEN projections
"""

import numpy as np
from scipy.ndimage import fourier_shift
from scipy import stats
from skimage import io
from skimage.feature import register_translation
from sklearn import linear_model

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
	
	return slope, intercept
