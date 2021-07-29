import numpy as np
from scipy import special
import pyfftw
import time
import scipy
from scipy import ndimage, misc

#~ def OTF(fx, fy, a1, a2, a3, a4, b1, b2, b3, c1, c2):
	#~ b4 = c1 - c2 - b1 - b2 - b3
	#~ return (b1*np.exp(-2*(np.pi*a1)**2*(fx**2+fy**2))+
			#~ b2*np.exp(-2*(np.pi*a2)**2*(fx**2+fy**2))+
			#~ b3*np.exp(-2*(np.pi*a3)**2*(fx**2+fy**2))+
			#~ b4*np.exp(-2*(np.pi*a4)**2*(fx**2+fy**2)))



def OTF(x, a1, a2, a3, a4, b1, b2, b3, c1, c2):
	b4 = c2 - c1 - b1 - b2 - b3
	print(b4)	
	return (b1*np.exp(-1*(np.pi*a1)**2*x)+
			b2*np.exp(-1*(np.pi*a2)**2*x)+
			b3*np.exp(-1*(np.pi*a3)**2*x)+
			b4*np.exp(-1*(np.pi*a4)**2*x))
			
			
#~ c1 = 0.01
#~ c2 = 1.00
#~ a1 = 0.9093
#~ a2 = 2.8537
#~ a3 = 49.3503
#~ a4 = 748.8534
#~ b1 = 0.5884
#~ b2 = 0.3570
#~ b3 = 0.0352
#~ pix_width = 0.65
#~ fx = np.fft.fftfreq(2560,d=pix_width)
#~ fy = np.fft.fftfreq(2160,d=pix_width)
#~ fx, fy = np.meshgrid(fx,fy)

#~ mt = OTF(fx,fy, a1, a2, a3, a4, b1, b2, b3)
#~ import matplotlib.pyplot as plt
#~ #plt.imshow(fx**2+fy**2)
#~ np.savetxt(r'C:\Users\forien1\Desktop\result_2.txt', fx**2+fy**2, fmt='%.2e')
#~ exit()
#~ plt.plot(fy[:,0])
#~ plt.show()
#~ exit()

 
    
def Energy2Wavelength(keV):
	"""
	convert energy in keV to wavelength in mm
	"""    
	h = 6.62607004e-34
	c = 299792458             
	JperkeV = 1.60218e-19
	return 1000*h*c / (JperkeV*keV*1000)
    

def Paganin(rad, wlen, dist, delta, beta, fx, fy, Rm, alpha):    		
	rad_freq = pyfftw.interfaces.numpy_fft.fft2(rad)

	'''from Paganin et al., 2002'''
	#~ alpha=0
	#~ mu = (4 * np.pi * beta) / wlen
	#~ phase = (rad_freq * mu) / (alpha+delta*dist*4*(np.pi**2)*(fx**2+fy**2)*Rm+mu) # 4 * pi^2 not explicit in manuscript
	#~ phase = np.real(pyfftw.interfaces.numpy_fft.ifft2(phase))
	#~ phase = (1/mu)*np.log(phase)
	#~ phase = (2*np.pi*delta/wlen)*phase

	'''from ANKA - Weitkamp et al., 2011'''
	filtre =  1 + (wlen*dist*delta*4*(np.pi**2)*(fx**2+fy**2) / (4*np.pi*beta)) # 4 * pi^2 not explicit in manuscript
	trans_func = np.log(np.real( pyfftw.interfaces.numpy_fft.ifft2( rad_freq / filtre)))
	phase = (delta/(2*beta)) * trans_func
		
	#~ phase = phase *(-wlen)/(2*np.pi*delta)
	return phase    

def multiPaganin(rads, wlen, dists, delta, beta, fx, fy, Rm, alpha):
	"""
	Phase retrieval method based on Contrast Transfer Function. This 
	method relies on linearization of the direct problem, based  on  the
	first  order  Taylor expansion of the transmittance function.
	Found in Yu et al. 2018 and adapted from Cloetens et al. 1999


	Parameters
	----------
	rad : 2D-array
		projection.
	wlen : float
		X-ray wavelentgth assumes monochromatic source.
	dist : float
		Object to detector distance (propagation distance) in mm.
	delta : float    
		refractive index decrement
	beta : float    
		absorption index
	fx, fy : ndarray
		Fourier conjugate / spatial frequency coordinates of x and y.
	alpha : float
		regularization factor.
		
	Return
	------

	phase retrieved projection in real space
	"""    
	numerator = 0
	denominator = 0    
	for j in range(0, len(dists)):    
		rad_freq = pyfftw.interfaces.numpy_fft.fft2(rads[j])	
		taylorExp = 1 + wlen * dists[j] * np.pi * (delta/beta) * (fx**2+fy**2)
		numerator = numerator + taylorExp * (rad_freq)
		denominator = denominator + taylorExp**2 

	numerator = numerator / len(dists)
	denominator = (denominator / len(dists)) + alpha

	phase = np.log(np.real(  pyfftw.interfaces.numpy_fft.ifft2(numerator / denominator) ))	
	phase = (delta/beta) * 0.5 * phase

	
	return phase
    

def sglDstCTF(rad, wlen, dist, delta, beta, fx, fy, Rm, alpha):
	"""
	Phase retrieval method based on Contrast Transfer Function.    This 
	method relies on linearization of the direct problem, based  on  the
	first  order  Taylor expansion of the transmittance function.
	Found in Yu et al. 2018 and adapted from Cloetens et al. 1999


	Parameters
	----------
	rad : 2D-array
		projection.
	wlen : float
		X-ray wavelentgth assumes monochromatic source.
	dist : float
		Object to detector distance (propagation distance) in mm.
	delta : float    
		refractive index decrement
	beta : float    
		absorption index
	fx, fy : ndarray
		Fourier conjugate / spatial frequency coordinates of x and y.
	alpha : float
		regularization factor.
		
	Return
	------

	phase retrieved projection in real space
	"""    
	delta_dirac = np.bitwise_and(fx==0,fy==0).astype(np.double) #Aditya: Discretized Dirac Delta function
	rad_freq = pyfftw.interfaces.numpy_fft.fft2(rad)
	filtre = np.cos(np.pi*wlen*dist*(fx**2+fy**2)) + (delta/beta) * np.sin(np.pi*wlen*dists[d]*(fx**2+fy**2))
	phase = (delta/beta) * 0.5 * ((rad_freq - delta_dirac) / filtre)
	phase = np.real(pyfftw.interfaces.numpy_fft.ifft2(phase))

	return phase


def multiCTF(rads, wlen, dists, delta, beta, fx, fy, Rm, alpha):
	"""



	Parameters
	----------
	rad : 2D-array
		projection.
	wlen : float
		X-ray wavelentgth assumes monochromatic source.
	dist : float
		Object to detector distance (propagation distance) in mm.
	delta : float    
		refractive index decrement
	beta : float    
		absorption index
	fx, fy : ndarray
		Fourier conjugate / spatial frequency coordinates of x and y.
	alpha : float
		regularization factor.
		
	Return
	------

	phase retrieved projection in real space
	"""    
	delta_dirac = np.bitwise_and(fx==0,fy==0).astype(np.double) #Aditya: Discretized Dirac Delta function
	numerator = 0
	denominator = 0
	for j in range(0, len(dists)):    
		rad_freq = pyfftw.interfaces.numpy_fft.fft2(rads[j])
		cos = np.cos(np.pi*wlen*dists[j]*(fx**2+fy**2))
		sin = np.sin(np.pi*wlen*dists[j]*(fx**2+fy**2)) 
		taylorExp = cos*Rm[:,:,j] + (delta/beta) * sin*Rm[:,:,j]
		numerator = numerator + taylorExp * (rad_freq - delta_dirac)
		denominator = denominator + taylorExp**2

	numerator = numerator / len(dists)
	denominator = (denominator / len(dists)) + alpha
	
	phase = numerator / denominator	
	phase = np.real(  pyfftw.interfaces.numpy_fft.ifft2(phase) )
	phase = (delta/beta) * 0.5 * phase
	
	return phase


def CTF(rads, wlen, dists, fx, fy, Rm, alpha):
	"""
	Phase retrieval method based on Contrast Transfer Function.    This 
	method assumes weak absoprtion and slowly varying phase shift.
	Derived from Langer et al., 2008: Quantitative comparison of direct
	phase retrieval algorithms.

	Parameters
	----------
	rads : list of 2D-array
		Elements of the list correspond to projections of the sample
		taken at different distance. One projection per element.
	wlen : float
		X-ray wavelentgth assumes monochromatic source.
	dists : list of float
		Object to detector distance (propagation distance) in mm. One 
		distance per element.
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

	for j in range(0,len(dists)):
		sin = 2*np.sin(np.pi*wlen*dists[j]*(fx**2+fy**2)) * Rm[:,:,j]
		cos = 2*np.cos(np.pi*wlen*dists[j]*(fx**2+fy**2)) * Rm[:,:,j]
		A = A + sin * cos
		B = B + sin * sin
		C = C + cos * cos
		rad_freq = pyfftw.interfaces.numpy_fft.fft2(rads[j])
		E = E + rad_freq * sin
		F = F + rad_freq * cos
	A = A / len(dists)
	B = B / len(dists)
	C = C / len(dists)	
	Delta = B * C - A**2
	
	phase = (C * E - A * F)    * (1 / (2*Delta+alpha)) 
	phase[0,0] = 0. + 0.j
	phase = pyfftw.interfaces.numpy_fft.ifft2(phase)
	phase = np.real(phase)

	return phase


def CTFPurePhase(rads, wlen, dists, delta, beta, fx, fy, Rm, alpha):
	"""
	weak phase approximation from Cloetens et al. 2002


	Parameters
	----------
	rad : 2D-array
		projection.
	wlen : float
		X-ray wavelentgth assumes monochromatic source.
	dist : float
		Object to detector distance (propagation distance) in mm.
	delta : float    
		refractive index decrement
	beta : float    
		absorption index
	fx, fy : ndarray
		Fourier conjugate / spatial frequency coordinates of x and y.
	alpha : float
		regularization factor.
		
	Return
	------

	phase retrieved projection in real space
	"""    

	numerator = 0
	denominator = 0    
	for j in range(0, len(dists)):    
		rad_freq = pyfftw.interfaces.numpy_fft.fft2(rads[j])
		taylorExp = np.sin(np.pi*wlen*dists[j]*(fx**2+fy**2)) 
		numerator = numerator + taylorExp * (rad_freq)
		denominator = denominator + 2*taylorExp**2 

	numerator = numerator / len(dists)
	denominator = (denominator / len(dists)) + alpha

	phase = np.log(np.real(  pyfftw.interfaces.numpy_fft.ifft2(numerator / denominator) ))
	phase = (delta/beta) * 0.5 * phase

	return phase

def CTFPurePhaseWithAbs(rads, wlen, dists, delta, beta, fx, fy, Rm, alpha):
	"""
	weak phase approximation from Cloetens et al. 2002


	Parameters
	----------
	rad : 2D-array
		projection.
	wlen : float
		X-ray wavelentgth assumes monochromatic source.
	dist : float
		Object to detector distance (propagation distance) in mm.
	delta : float    
		refractive index decrement
	beta : float    
		absorption index
	fx, fy : ndarray
		Fourier conjugate / spatial frequency coordinates of x and y.
	alpha : float
		regularization factor.
		
	Return
	------

	phase retrieved projection in real space
	"""    
	argMin = np.argmin(dists)
	numerator = 0
	denominator = 0    
	for j in range(0, len(dists)):    
		rad_freq = pyfftw.interfaces.numpy_fft.fft2(rads[j]/rads[argMin])
		taylorExp = np.sin(np.pi*wlen*dists[j]*(fx**2+fy**2)) 
		numerator = numerator + taylorExp * (rad_freq)
		denominator = denominator + 2*taylorExp**2 

	numerator = numerator / len(dists)
	denominator = (denominator / len(dists)) + alpha

	phase = np.log(np.real(  pyfftw.interfaces.numpy_fft.ifft2(numerator / denominator) ))
	phase = (delta/beta) * 0.5 * phase

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





def TIE(rads,wlen,dists,pix_width,fx,fy,Rm,alpha):		
	"""
	Transport of Intensity Equation
	Derived from Langer et al., 2008: Quantitative comparison of direct
	phase retrieval algorithms.

	Parameters
	----------
	rads : 2D-array
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
	rad0 = rads[0]
	rad1 = rads[1]
	
	rad0[rad0==0] = 1e-6 #avoid division by 0 of some pixel
	res = (rad1-rad0) / (dists[1]-dists[0])
	res = invLaplacian(res,pix_width,fx,fy,alpha)
	res_y, res_x = np.gradient(res)
	res_x = res_x/rad0
	res_y = res_y/rad0
	res = np.gradient(res_x, axis=1) + np.gradient(res_y, axis=0)
	res = res/(pix_width**2) #Aditya: Should divide by pixel width for each gradient
	res = invLaplacian(res,pix_width,fx,fy,alpha)
	res = (-2*np.pi/wlen)*res    
	return res





def WTIE(rads,wlen,dists,pix_width,fx,fy,Rm,alpha):
	"""
	TIE for weak absorption. Similar method to the TIE but combining
	phase retrieval and inverse Radon transform in one step. 
	Derived from Langer et al., 2008: Quantitative comparison of direct
	phase retrieval algorithms.

	Parameters
	----------
	rads : 2D-array
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
	rad0 = rads[0]
	rad1 = rads[1]
	
	rad0 = rad0 + 1e-6 #avoid division by 0
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
	# ~ t0 = time.time()
	res_y, res_x = np.gradient(np.log(I0))    
	res = np.gradient(phase * res_x, axis=1) + np.gradient(phase * res_y, axis=0)    
	res = res/(pix_width**2) #Aditya: Should divide by pixel width for each gradient
	# ~ t1 = time.time()
	# ~ print('here delta_D: '+str(t1-t0))
	
	return np.cos(np.pi*wlen*dist*(fx**2+fy**2)) * (wlen*dist*(1/(2*np.pi))) * pyfftw.interfaces.numpy_fft.fft2(res) #* np.sqrt(fx**2 + fy**2)
	
	# ~ t0 = time.time()
	# ~ res = torch.from_numpy(res)
	# ~ res = torch.fft.fft(torch.fft.fft(res, dim=0),dim=1)
	# ~ res = res.numpy()	
	# ~ t1 = time.time()
	# ~ res = np.cos(np.pi*wlen*dist*(fx**2+fy**2)) * (wlen*dist*(1/(2*np.pi))) * res	
	# ~ print('here delta_D 2: '+str(t1-t0))
	
	# ~ return res
	
	
	# ~ t0 = time.time()
	# ~ sigma_I0filter = 10
	# ~ res = torch.from_numpy(np.log(I0))
	# ~ res = torch.fft.fft(torch.fft.fft(res, dim=0),dim=1)
	# ~ res = res.numpy()	
	# ~ res = scipy.ndimage.fourier_gaussian(res, sigma_I0filter)
	
	# ~ res_x = torch.from_numpy(res*1j*2*np.pi*fx)
	# ~ res_x = torch.fft.ifft(res_x)
	# ~ res_x = res_x.numpy()
	# ~ res_x = res_x.real
	
	# ~ res_y = torch.from_numpy(res*1j*2*np.pi*fy)
	# ~ res_y = torch.fft.ifft(res_y)
	# ~ res_y = res_y.numpy()
	# ~ res_y = res_y.real
	
	# ~ res = phase*res_x + phase*res_y
	# ~ t1 = time.time()
	# ~ print('here delta_D 1: '+str(t1-t0))
		

	# ~ t0 = time.time()
	# ~ res = torch.from_numpy(res)
	# ~ res = torch.fft.fft(torch.fft.fft(res, dim=0),dim=1)
	# ~ res = res.numpy()	
	# ~ t1 = time.time()
	# ~ res = np.cos(np.pi*wlen*dist*(fx**2+fy**2)) * (wlen*dist*(1/(2*np.pi))) * res	
	# ~ print('here delta_D 2: '+str(t1-t0))
	
	# ~ return res #* np.sqrt(fx**2 + fy**2)

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
	
	res = (delta*I0*np.log(I0))/(2*beta)	
	# ~ res = torch.from_numpy(res)
	# ~ res = torch.fft.fft(torch.fft.fft(res, dim=0),dim=1)		
	# ~ res = res.numpy()
	res = pyfftw.interfaces.numpy_fft.fft2(res)
	return special.erfc(fr-fc)* res



def Lcurve(self, dataset, projection):
	
	Lcurve_min = -9
	Lcurve_max = -3
	Lcurve_step = 1
	Lcurve_range = np.arange(Lcurve_min, Lcurve_max+1, Lcurve_step, dtype=float)
	
	alpha_HF = -10

	FID = [0 for x in range(self.ND)]
	for distance in range(self.ND):
		FID[distance] = dataset.GetProjection(projection, distance+1, 'Fourier')              
		if not distance == 0:     
			FID[distance] = FID[distance] - FID[0] #Wouldn't it be better filtered as well?

	FI0_filtered = scipy.ndimage.fourier_gaussian(FID[0], self.sigma_I0filter)
	I0_filtered = np.real(np.fft.ifft2(FI0_filtered))
	dfxI0 = np.real(np.fft.ifft2(2j*np.pi*self.fx * FI0_filtered)) / I0_filtered
	dfyI0 = np.real(np.fft.ifft2(2j*np.pi*self.fy * FI0_filtered)) / I0_filtered

	I0 = np.real(np.fft.ifft2(FID[0]))
	#TODO: refactor doule code with reconstructprojection

	if self.prior == 'forward':
		prior = dataset.GetImage(projection, 'prior', 'Fourier')
		self.delta_beta = 1 #TODO: Necessary?
		print('forward')
	else:
		prior = np.fft.fft2(np.log(I0)*I0/2)

	prior = self.LPfilter * prior

	model_error = np.zeros(Lcurve_range.shape)
	regularisation_error = np.zeros(Lcurve_range.shape)        
	for index in range(Lcurve_range.shape[0]):
		print('Alpha: {}, Delta/Beta: {}'.format(Lcurve_range[index], self.delta_beta))
		self.alpha = np.array([10**Lcurve_range[index], 10**alpha_HF])

		phase = self.ReconstructProjection(dataset, projection) # TODO: ReconstructProjection could return image?
		
		# Propagate with mixed (move to propagator)
		
		phase_dfxI0 = np.fft.fft2(phase*dfxI0)
		phase_dfyI0 = np.fft.fft2(phase*dfyI0)
		
		phasef = np.fft.fft2(phase)
		
		for distance in range(1, self.ND):
			mixed_contrast = 2*self.sinchirp[distance]*phasef + self.coschirp_dfx[distance]*phase_dfxI0 + self.coschirp_dfy[distance]*phase_dfyI0
			model_difference = FID[distance] - mixed_contrast
			model_difference[0, 0] = 0 # Disregard offset (necessary?)
			model_difference_r = np.fft.ifft2(model_difference)
			model_difference_rc = np.real(model_difference_r[self.ny//2:-self.ny//2, self.nx//2:-self.nx//2])
			model_error[index] += np.sum(np.square(model_difference_rc)) / (self.nx*self.ny)
		
		model_error[index] = model_error[index] / (self.ND-1)
		regularisation_difference = phasef - self.delta_beta*prior
		regularisation_difference = np.fft.ifft2(regularisation_difference)
		regularisation_difference = np.real(regularisation_difference[self.ny//2:-self.ny//2, self.nx//2:-self.nx//2])
		regularisation_error[index] = np.sum(np.square(regularisation_difference)) / (self.nx*self.ny)
		print("ME: {} , RE: {}".format(model_error[index], regularisation_error[index]))
	
	model_error_log = np.log10(model_error)
	regularisation_error_log = np.log10(regularisation_error)

#        model_error_log = np.array([-4.2471, -4.2460, -4.2442, -4.2384, -4.2114, -4.1003, -3.8592])
#        regularisation_error_log = np.array([1.477485, 0.790983, 0.313785, -0.039234, -0.403655, -0.896721, -1.472450])
#        LR = np.array([-9, -8, -7, -6, -5, -4, -3])
	
	Loversamp=10
	#LR = np.log10(Lcurve_range)     
	t = np.linspace(0, 1, len(model_error_log))
	ts = np.linspace(0, 1, len(model_error_log)*Loversamp)
	
	M = interpolate.UnivariateSpline(t, model_error_log, s=0)    
	R = interpolate.UnivariateSpline(t, regularisation_error_log, s=0)    
	
	Mp = M.derivative()
	Rp = R.derivative()
	
	Mpp = Mp.derivative()
	Rpp = Rp.derivative()
	
	K = (Mp(ts)*Rpp(ts) - Rp(ts)*Mpp(ts)) / (Rp(ts)**2 + Mp(ts)**2)**1.5 # Langer 2010 eq. 25

	Mts = M(ts)
	Rts = R(ts)
	
#        Kmax = K[Mts.argmin():Rts.argmin()].argmax()
	Kmax = K.argmax()
	Mmin = Mts.argmin()
	
	Lts = np.linspace(Lcurve_range.min(), Lcurve_range.max(), len(Lcurve_range)*Loversamp)
	
	#TODO: Plot function in display

	self.alpha[0]=10**Lts[Kmax]
	dataset.alpha = self.alpha
	
	lcurve_filename=dataset.path+'/'+dataset.name+'_/lcurve.pickle' #TODO: refactor
	with open(lcurve_filename, 'wb') as f:
		pickle.dump([Lts, model_error_log, regularisation_error_log, Mts, Rts, K, Kmax, Mmin], f, pickle.HIGHEST_PROTOCOL)

	dataset.WriteParameterFile()
	self.DisplayLcurve(dataset)




def DisplayLcurve(self, dataset):
	lcurve_filename=dataset.path+'/'+dataset.name+'_/lcurve.pickle' #TODO: refactor
	with open(lcurve_filename, 'rb') as f:
		Lts, model_error_log, regularisation_error_log, Mts, Rts, K, Kmax, Mmin = pickle.load(f)
###
	Loversamp=10
	#LR = np.log10(Lcurve_range)     
	t = np.linspace(0, 1, len(model_error_log))
	ts = np.linspace(0, 1, len(model_error_log)*Loversamp)

	M = interpolate.InterpolatedUnivariateSpline(t, model_error_log, k=4)    
	R = interpolate.InterpolatedUnivariateSpline(t, regularisation_error_log, k=4)    


	Mp = M.derivative()
	Rp = R.derivative()

	Mpp = M.derivative(2)
	Rpp = R.derivative(2)

	K = (Mp(ts)*Rpp(ts) - Rp(ts)*Mpp(ts)) / (Rp(ts)**2 + Mp(ts)**2)**1.5 # Langer 2010 eq. 25

	Mts = M(ts)
	Rts = R(ts)
###
	
	plt.figure()
	plt.plot(Lts, K)
	plt.show()

	plt.figure()
	plt.plot(model_error_log, regularisation_error_log, 'rx', Mts, Rts, 'b-', Mts[Kmax], Rts[Kmax], 'go', Mts[Mmin], Rts[Mmin], 'co')
	plt.show()
   
#        plt.figure()
#        plt.title("Mts")
#        plt.plot(Lts,Mts)
#        plt.show()
#        
#        plt.figure()
#        plt.title("Rts")
#        plt.plot(Lts,Rts)
#        plt.show()
#        
#        plt.figure()
#        plt.title("Mp")
#        plt.plot(Lts,Mp(ts))
#        plt.show()
#
#        plt.figure()
#        plt.title("Mpp")
#        plt.plot(Lts,Mpp(ts))
#       
#        plt.show()
#
#        plt.figure()
#        plt.title("Rp")
#        plt.plot(Lts,Rp(ts))
#        plt.show()
#
#        plt.figure()
#        plt.title("Rpp")
#        plt.plot(Lts,Rpp(ts))
#        plt.show()

	print('Maximum curvature at: {}'.format(Lts[Kmax]))
	print('Minimum model error at: {}'.format(Lts[Mmin]))

	pass




def mixed(rads, wlen, dists, pix_width, delta, beta, fx, fy, Rm, alpha):
	A_D2 = rads[0] * 0.0
	radj_fft_img = []
	radj_fft_img.append(pyfftw.interfaces.numpy_fft.fft2(rads[0]))
	for j in range(1, len(dists)): 
		# ~ img = np.array(rads[j])
		# ~ img = torch.from_numpy(img)			
		# ~ fft_img = torch.fft.fft(torch.fft.fft(img, dim=0),dim=1)
		# ~ fft_img = fft_img.numpy()			
		# ~ radj_fft_img.append(fft_img)
		radj_fft_img.append(pyfftw.interfaces.numpy_fft.fft2(rads[j]))	
		A_D2 = A_D2 + (2 * np.sin(np.pi*wlen*dists[j]*(fx**2+fy**2)))**2
	
	sigma = 10
	I0 = np.real(pyfftw.interfaces.numpy_fft.ifft2(scipy.ndimage.fourier_gaussian(radj_fft_img[0], sigma)))	

	dfxI0 = np.real(pyfftw.interfaces.numpy_fft.ifft2(2j*np.pi*fx * scipy.ndimage.fourier_gaussian(radj_fft_img[0], sigma))) / I0
	dfyI0 = np.real(pyfftw.interfaces.numpy_fft.ifft2(2j*np.pi*fy * scipy.ndimage.fourier_gaussian(radj_fft_img[0], sigma))) / I0

	prior = pyfftw.interfaces.numpy_fft.fft2(np.log(I0)*I0/2)
	
	R = np.sqrt(fx**2 + fy**2)
	LP_cutoff = 0.5
	LP_slope = .5e3
	LPfilter = 1 - 1/(1 + np.exp(-LP_slope * (R-LP_cutoff))) #Logistic filter

	prior = LPfilter * prior
	phase = rads[0] * 0.0
	
	for i in range(4): #minimum of 3 iterations required        
		numerator = rads[0] * 0.0
		
		phase_dfxI0 = pyfftw.interfaces.numpy_fft.fft2(phase*dfxI0)
		phase_dfyI0 = pyfftw.interfaces.numpy_fft.fft2(phase*dfyI0)

		for j in range(1, len(dists)): 
			numerator = numerator + ( (2 * np.sin(np.pi*wlen*dists[j]*(fx**2+fy**2))) * 
							(radj_fft_img[j] - radj_fft_img[0] - 
							np.cos(np.pi*wlen*dists[j]*(fx**2)) * (wlen*dists[j]*(1/(2*np.pi))) * phase_dfxI0 -
							np.cos(np.pi*wlen*dists[j]*(fy**2)) * (wlen*dists[j]*(1/(2*np.pi))) * phase_dfyI0) )
							
							
			
		numerator = numerator / (len(dists)-1)
	  
		phase_n = (numerator + (alpha*(delta/beta)*prior)) / (alpha+A_D2)
		phase_n = np.real(pyfftw.interfaces.numpy_fft.ifft2(phase_n))
		
		# ~ print("Iteration: {} RMS: {}".format(n, np.sqrt(np.sum((phase_n-phase)*(phase_n-phase).conjugate())/(self.nfx*self.nfy))))
		phase = phase_n
			
	return phase

	
	
	
	
def mixedAppr_homo(rads, A_D, phi_0, wlen, dists, pix_width, delta, beta, fx, fy, Rm, alpha):
				  
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
		
	# ~ fc = 1 / np.sqrt(2*wlen*np.amax(dists))        
		
	radj_fft_img = []
	for j in range(0, len(dists)): 
		# ~ img = np.array(rads[j])
		# ~ img = torch.from_numpy(img)			
		# ~ fft_img = torch.fft.fft(torch.fft.fft(img, dim=0),dim=1)
		# ~ fft_img = fft_img.numpy()			
		# ~ radj_fft_img.append(fft_img)
		radj_fft_img.append(pyfftw.interfaces.numpy_fft.fft2(rads[j]))
	
	
	for i in range(4): #minimum of 3 iterations required        
		numerator = rads[0] * 0.0
		denominator = rads[0] * 0.0
		if i == 0:
			phase = rads[0] * 0.0                        				
			
		for j in range(0, len(dists)):    
			# ~ numerator = numerator + ( A_D(wlen, dists[j], fx, fy) *
				# ~ (np.fft.fft2(rads[j]) - np.fft.fft2(rads[0]) -
				# ~ delta_D(rads[0], phase, wlen, dists[j], pix_width, fx, fy)) )								
			
			# ~ t2 = time.time()
			# ~ numerator = numerator + ( A_D[j] *
									# ~ (radj_fft_img[j] - radj_fft_img[0] -
									# ~ delta_D(rads[0], phase, wlen, dists[j], pix_width, fx, fy)) )
			
			
			# ~ t0 = time.time()
			res_y, res_x = np.gradient(np.log(rads[0]))    
			res = np.gradient(phase * res_x, axis=1) + np.gradient(phase * res_y, axis=0)    
			res = res/(pix_width**2) #Aditya: Should divide by pixel width for each gradient
			# ~ t1 = time.time()
			# ~ print('here delta_D: '+str(t1-t0))						
			
			# ~ t0 = time.time()
			# ~ res = torch.from_numpy(res)
			# ~ res = torch.fft.fft(torch.fft.fft(res, dim=0),dim=1)
			# ~ res = res.numpy()	
			res = pyfftw.interfaces.numpy_fft.fft2(res)
			# ~ t1 = time.time()
			delta_D = np.cos(np.pi*wlen*dists[j]*(fx**2+fy**2)) * (wlen*dists[j]*(1/(2*np.pi))) * res	
			# ~ print('here delta_D 2: '+str(t1-t0))
			
			# ~ t0 = time.time()
			numerator = numerator + ( A_D[j] * (radj_fft_img[j] - radj_fft_img[0] - delta_D) )
			# ~ t1 = time.time()
			# ~ print('here delta_D 3: '+str(t1-t0))					
			
			# ~ t3 = time.time()
			# ~ print("here 3:"+str(t3-t2))
			
			denominator = denominator + A_D[j]**2        
			# ~ t4 = time.time()
			# ~ print("here 4:"+str(t4-t3))
										
		
		
		numerator = numerator + alpha * phi_0
		denominator = denominator + alpha
		phase = numerator / denominator        
		phase[0,0] = 0. + 0.j
				
		
		phase = pyfftw.interfaces.numpy_fft.ifft2(phase)
		phase = phase.real
		# ~ phase = torch.from_numpy(phase)		
		# ~ phase = torch.fft.ifft(phase)
		# ~ phase = phase.numpy()
		# ~ phase = phase.real

		

	# ~ phase = phase/rads[0] #Aditya: Divide by I0	
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


