import numpy as np
from scipy import special
import pyfftw


	
def Energy2Wavelength(keV):
	"""
	convert energy in keV to wavelength in mm
	"""	
	h = 6.62607004e-34
	c = 299792458 			
	JperkeV = 1.60218e-19
	return 1000*h*c / (JperkeV*keV*1000)
	
			
def CTF(rads, rads_freq, wlen, dists, pix_width, fx, fy, alpha):
	"""
	Phase retrieval method based on Contrast Transfer Function.	This 
	method assumes weak absoprtion and slowly varying phase shift.
	Derived from Langer et al., 2008: Quantitative comparison of direct
	phase retrieval algorithms.

	Parameters
	----------
	rads : list of 2D-array
		Elements of the list correspond to projections of the sample
		taken at different distance. One projection per element.
	rads_freq : list of 2D-matrix
		Elements of the list correspond to the fourier transform of the 
		projections at different distance. One FFT per element.	
	wlen : float
		X-ray wavelentgth assumes monochromatic source.
	dists : list of float
		Object to detector distance (propagation distance) in mm. One 
		distance per element.
	pix_width : float
		Pixel width in mm
	fx, fy : ndarray
		Fourier conjugate / spatial frequency coordinates of x and y.
	alpha : float
		regularization factor.
		
	Return
	------
	
	phase retrieved projection in real space
	
	"""
	
	A = np.zeros((rads[0].shape[0], rads[0].shape[1]))
	B = np.zeros((rads[0].shape[0], rads[0].shape[1]))
	C = np.zeros((rads[0].shape[0], rads[0].shape[1]))
	E = np.zeros((rads[0].shape[0], rads[0].shape[1]))
	F = np.zeros((rads[0].shape[0], rads[0].shape[1]))
	for d in range(0,len(dists)):
		A = A + np.sin(np.pi*wlen*dists[d]*(fx**2+fy**2)) * np.cos(np.pi*wlen*dists[d]*(fx**2+fy**2))
		B = B + np.sin(np.pi*wlen*dists[d]*(fx**2+fy**2)) * np.sin(np.pi*wlen*dists[d]*(fx**2+fy**2))
		C = C + np.cos(np.pi*wlen*dists[d]*(fx**2+fy**2)) * np.cos(np.pi*wlen*dists[d]*(fx**2+fy**2))
		E = E + rads_freq[d] * np.sin(np.pi*wlen*dists[d]*(fx**2+fy**2))
		F = F + rads_freq[d] * np.cos(np.pi*wlen*dists[d]*(fx**2+fy**2))
	Delta = B * C - A * A
	phase = (1 / (2*Delta+alpha)) * (C * E - A * F)	
	phase[0,0] = 0. + 0.j
	# ~ phase = np.fft.ifft2(phase)
	phase = pyfftw.interfaces.numpy_fft.ifft2(phase)
	phase = phase.real	
	
	return phase




	
def invLaplacian(rads,pix_width,fx,fy,alpha):	
	"""
	calculate inverse laplacian according to equation (21) from Langer 
	et al., 2008: Quantitative comparison of direct phase retrieval 
	algorithms.

	Parameters
	----------	
	rads : 2D-array
		2D projection in real space.
	pix_width : float
		Pixel width in mm
	fx, fy : ndarray
		Fourier conjugate / spatial frequency coordinates of x and y.
	alpha : float
		regularization factor.
		
	Return
	------
	
	phase retrieved projection in real space
	
	"""
		
	# ~ rads_freq = np.fft.fft2(rads,pix_width)
	rads_freq = pyfftw.interfaces.numpy_fft.fft2(rads)
	res = rads_freq/(fx**2+fy**2+alpha)
	res[0,0] = 0. + 0.j	
	# ~ res = -(1/(4*np.pi**2))*get_inv_fft(res).real
	res = -(1/(4*np.pi**2))*pyfftw.interfaces.numpy_fft.ifft2(res).real
	return res





def TIE(rad0,rad1,wlen,dists,pix_width,fx,fy,alpha):
	"""
	Transport of Intensity Equation
	Derived from Langer et al., 2008: Quantitative comparison of direct
	phase retrieval algorithms.

	Parameters
	----------
	rad0 and rad1 : 2D-array
		Images of the projections in real space taken at the first and 
		second propagation distance.	
	wlen : float
		X-ray wavelentgth assumes monochromatic source.
	dists : list of float
		Object to detector distance (propagation distance) in mm. One 
		distance per element.
	pix_width : float
		Pixel width in mm
	fx, fy : ndarray
		Fourier conjugate / spatial frequency of x and y.
	alpha : float
		regularization factor.
			
	Return
	------
	
	phase retrieved projection in real space
	
	"""
	
	res = (rad1-rad0) / (dists[1]-dists[0])
	res = invLaplacian(res,pix_width,fx,fy,alpha)
	res_y, res_x = np.gradient(res)
	res_x = res_x/rad0
	res_y = res_y/rad0
	res = np.gradient(res_x, axis=1) + np.gradient(res_y, axis=0)
	res = invLaplacian(res,pix_width,fx,fy,alpha)
	res = (-2*np.pi/wlen)*res	
	return res





def WTIE(rad0,rad1,wlen,dists,pix_width,fx,fy,alpha):
	"""
	TIE for weak absorption. Similar method to the TIE but combining
	phase retrieval and inverse Radon transform in one step. 
	Derived from Langer et al., 2008: Quantitative comparison of direct
	phase retrieval algorithms.

	Parameters
	----------
	rad0 and rad1 : 2D-array
		Images of the projections in real space taken at the first and 
		second propagation distance.	
	wlen : float
		X-ray wavelentgth assumes monochromatic source.
	dists : list of float
		Object to detector distance (propagation distance) in mm. One 
		distance per element.
	pix_width : float
		Pixel width in mm
	fx, fy : ndarray
		Fourier conjugate / spatial frequency of x and y.
	alpha : float
		regularization factor.
			
	Return
	------
	
	phase retrieved projection in real space
	
	"""
	
	res = (rad1/rad0) - 1
	res = -((2*np.pi)/(wlen*(dists[1])))*invLaplacian(res,pix_width,fx,fy,alpha)
	res[0,0] = 0.
	res = res.real
	return res





def A_D(wlen, dist, fx, fy):	
	"""
	used in mixed approach algorithm

	Parameters
	----------
	wlen : float
		X-ray wavelentgth assumes monochromatic source.
	dist : float
		Object to detector distance (propagation distance) in mm.
	fx, fy : ndarray
		Fourier conjugate / spatial frequency of x and y.	
	"""
	
	return 2 * np.sin(np.pi*wlen*dist*(fx**2+fy**2))

def delta_D(I0, phase, wlen, dist, pix_width, fx, fy):							 
	"""
	used in mixed approach algorithm

	Parameters
	----------
	I0 : float
		absorption contrast projection.
	phase : 2D-array
	wlen : float
		X-ray wavelentgth assumes monochromatic source.		
	dist : float
		Object to detector distance (propagation distance) in mm.
	pix_width : float
		Pixel width in mm
	fx, fy : ndarray
		Fourier conjugate / spatial frequency of x and y.	
	"""
	res_y, res_x = np.gradient(np.log(I0))	
	res = np.gradient(phase * res_x, axis=1) + np.gradient(phase * res_y, axis=0)	
	# ~ return np.cos(np.pi*wlen*dist*(fx**2+fy**2)) * (wlen*dist*(1/(2*np.pi))) * np.fft.fft2(res) #* np.sqrt(fx**2 + fy**2)
	return np.cos(np.pi*wlen*dist*(fx**2+fy**2)) * (wlen*dist*(1/(2*np.pi))) * pyfftw.interfaces.numpy_fft.fft2(res) #* np.sqrt(fx**2 + fy**2)

def phi_0(delta, beta, I0, fc, pix_width, fx, fy):
	"""
	used in mixed approach algorithm

	Parameters
	----------
	delta : float	
		refractive index decrement
	beta : float	
		absorption index
	I0 : float
		absorption contrast projection.
	fc : float
		first maximum of the transfer function to the longest distance
	pix_width : float
		Pixel width in mm
	fx, fy : ndarray
		Fourier conjugate / spatial frequency of x and y.	
	"""
	fr = np.sqrt(fx**2 + fy**2)
	# ~ return special.erfc(fr-fc)* np.fft.fft2((delta*I0*np.log(I0))/(2*beta))	
	return special.erfc(fr-fc)* pyfftw.interfaces.numpy_fft.fft2((delta*I0*np.log(I0))/(2*beta))	



def mixedAppr_homo(rads, wlen, dists, pix_width, delta, beta, fx, fy, alpha):
	"""
	mixed approach with phase attenuation duality as prior. Combination 
	of TIE and CTF algorithm.
	Derived from Langer et al., 2010: Regularization of phase retrieval
	with phase-attenuation dulaity prior for 3-D holotomography

	Parameters
	----------
	rads : 2D-array
		2D projection in real space.
	wlen : float
		X-ray wavelentgth assumes monochromatic source.
	dists : list of float
		Object to detector distance (propagation distance) in mm. One 
		distance per element.
	pix_width : float
		Pixel width in mm
	delta : float	
		refractive index decrement
	beta : float	
		absorption index
	fx, fy : ndarray
		Fourier conjugate / spatial frequency of x and y.
	alpha : float
		regularization factor.
			
	Return
	------
	
	phase retrieved projection in real space
	
	"""
		
	fc = 1 / np.sqrt(2*wlen*np.amax(dists))		
	
	for i in range(4): #minimum of 3 iterations required		
		numerator = rads[0] * 0.0
		denominator = rads[0] * 0.0
		if i == 0:
			phase = rads[0] * 0.0						
			
		for j in range(0, len(dists)):	
			# ~ numerator = numerator + ( A_D(wlen, dists[j], fx, fy) *
				# ~ (np.fft.fft2(rads[j]) - np.fft.fft2(rads[0]) -
				# ~ delta_D(rads[0], phase, wlen, dists[j], pix_width, fx, fy)) )
			numerator = numerator + ( A_D(wlen, dists[j], fx, fy) *
				(pyfftw.interfaces.numpy_fft.fft2(rads[j]) -  pyfftw.interfaces.numpy_fft.fft2(rads[0]) -
				delta_D(rads[0], phase, wlen, dists[j], pix_width, fx, fy)) )
			denominator = denominator + (A_D(wlen, dists[j], fx, fy))**2		
										
		numerator = numerator + alpha * phi_0(delta, beta, rads[0], fc, pix_width, fx, fy)
		denominator = denominator + alpha
		phase = numerator / denominator		
		phase[0,0] = 0. + 0.j
		# ~ phase = np.fft.ifft2(phase)
		phase = pyfftw.interfaces.numpy_fft.ifft2(phase)
		phase = phase.real
		
	return phase

def M(rads, rads_freq, phase, dists, delta, beta, wlen, pix_width, fx, fy, alpha):
	"""
	used in mixed approach algorithm to determine regularizing parameter
	alpha
	
	Parameters
	----------
	rads : list of 2D-array
		Elements of the list correspond to projections of the sample
		taken at different distance. One projection per element.
	rads_freq : list of 2D-matrix
		Elements of the list correspond to the fourier transform of the 
		projections at different distance. One FFT per element.	
	phase : 2D-array
	dists : list of float
		Object to detector distance (propagation distance) in mm. One 
		distance per element.
	delta : float	
		refractive index decrement
	beta : float	
		absorption index
	wlen : float
		X-ray wavelentgth assumes monochromatic source.	
	pix_width : float
		Pixel width in mm
	fx, fy : ndarray
		Fourier conjugate / spatial frequency of x and y.	
	alpha : float
		regularization factor.
	"""
	
	res = 0
	for j in range(1,len(dists)):
		# ~ res = res + np.absolute(A_D(wlen, dists[j], fx, fy) * np.fft.fft2(phase) - 
				# ~ (rads_freq[j] - 
				# ~ rads_freq[0] - 
		res = res + np.absolute(A_D(wlen, dists[j], fx, fy) * pyfftw.interfaces.numpy_fft.fft2(phase) - 
				(rads_freq[j] - 
				rads_freq[0] - 					
				delta_D(rads[0], phase, wlen, dists[j], pix_width, fx, fy)))				
	return res
		
def R(rads, phase, dists, delta, beta, wlen, pix_width, fx, fy, alpha):
	"""
	used in mixed approach algorithm to determine regularizing parameter
	alpha
	
	Parameters
	----------
	rads : list of 2D-array
		Elements of the list correspond to projections of the sample
		taken at different distance. One projection per element.
	phase : 2D-array
	dists : list of float
		Object to detector distance (propagation distance) in mm. One 
		distance per element.
	delta : float	
		refractive index decrement
	beta : float	
		absorption index
	wlen : float
		X-ray wavelentgth assumes monochromatic source.	
	pix_width : float
		Pixel width in mm
	fx, fy : ndarray
		Fourier conjugate / spatial frequency of x and y.	
	alpha : float
		regularization factor.
	"""
	
	fc = 1 / np.sqrt(2*wlen*dists[-1])	
	# ~ return np.absolute(np.fft.fft2(phase) - phi_0(delta, beta, np.fft.fft2(rads[0]), fc, fx, fy, pix_width))
	return np.absolute(pyfftw.interfaces.numpy_fft.fft2(phase) - phi_0(delta, beta, pyfftw.interfaces.numpy_fft.fft2(rads[0]), fc, fx, fy, pix_width))
	
