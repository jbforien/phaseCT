#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Holotomography module for aligning multi ATTEN projections
"""

import imreg_dft as ird
import matplotlib.pyplot as plt
import numpy as np
import os
import pyfftw #requires scipy 1.3.3 (pip install scipy==1.3.3)

from skimage.registration import phase_cross_correlation
from skimage.registration._phase_cross_correlation import _upsampled_dft

from scipy.ndimage import fourier_shift
from sklearn import linear_model
from scipy import stats

def shiftImage(img, shift):	
	offset_img = fourier_shift(pyfftw.interfaces.numpy_fft.fftn(img), shift)
	offset_img = pyfftw.interfaces.numpy_fft.ifftn(offset_img)
	offset_img = np.real(offset_img)
	offset_img = offset_img.astype(np.float32)	
	return offset_img

def crossCorr_imreg_dft(img1, img2):
	# ~ filter_pcorr (int) – Radius of the minimum spectrum filter for translation detection, use the filter when detection fails. Values > 3 are likely not useful.
	shift = ird.translation(img1, img2, filter_pcorr=8, odds=1)
	shift = shift["tvec"].round(4)	
	return np.asarray([shift[1],shift[0]])


def crossCorr_skimage_fourier(img1, img2):
	shift, error, diffphase = phase_cross_correlation(img1, img2, upsample_factor=100, space='fourier')
	return shift

def crossCorr_skimage_real(img1, img2):
	shift, error, diffphase = phase_cross_correlation(img1, img2, upsample_factor=1, space='real')
	return np.asarray([shift[1],shift[0]])


from dipy.viz import regtools
from dipy.data import fetch_stanford_hardi, read_stanford_hardi
from dipy.data.fetcher import fetch_syn_data, read_syn_data
from dipy.align.imaffine import (transform_centers_of_mass,
									AffineMap,
									MutualInformationMetric,
									AffineRegistration)
from dipy.align.transforms import (TranslationTransform2D,
									TranslationTransform3D,
									RigidTransform2D,
									RigidTransform3D,
									AffineTransform2D,
									AffineTransform3D)

def mutualInfo_dipy(img1, img2):	
	img1_grid2world = np.identity(3)	
	img2_grid2world = np.identity(3)	
		
	# compute center of mass	
	c_of_mass = transform_centers_of_mass(img1, img1_grid2world,
										img2, img2_grid2world)	
	
	x_shift = c_of_mass.affine[1,-1]
	y_shift = c_of_mass.affine[0,-1]
	
	# prepare affine registration
	nbins = 32
	sampling_prop = None
	metric = MutualInformationMetric(nbins, sampling_prop)
	level_iters = [10000, 1000, 100]
	sigmas = [3.0, 1.0, 0.0]
	factors = [4, 2, 1]	
	affreg = AffineRegistration(metric=metric,
									level_iters=level_iters,
									sigmas=sigmas,
									factors=factors)

	# translation								
	translation = affreg.optimize(img1, img2, TranslationTransform2D(), None,
									img1_grid2world, img2_grid2world,
									starting_affine=c_of_mass.affine)
	
	x_shift = translation.affine[1,-1]
	y_shift = translation.affine[0,-1]
	return np.asarray([-x_shift,-y_shift])
	



def lin_RANSAC(x, y):		
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
