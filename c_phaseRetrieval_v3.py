import numpy as np
from scipy import special
import pyfftw
import time
import scipy
from scipy import ndimage, misc

def OTF(x, a1, a2, a3, a4, b1, b2, b3, c1, c2):
    b4 = c2 - c1 - b1 - b2 - b3
    print(b4)    
    return (b1*np.exp(-1*(np.pi*a1)**2*x)+
            b2*np.exp(-1*(np.pi*a2)**2*x)+
            b3*np.exp(-1*(np.pi*a3)**2*x)+
            b4*np.exp(-1*(np.pi*a4)**2*x))
            
    
def Energy2Wavelength(keV):
    """
    convert energy in keV to wavelength in mm
    """    
    h = 6.62607004e-34
    c = 299792458             
    JperkeV = 1.60218e-19
    return 1000*h*c / (JperkeV*keV*1000)
    

def Paganin(rad, wlen, dist, delta, beta, fx, fy, Rm):            
    rad_freq = pyfftw.interfaces.numpy_fft.fft2(rad)

    '''from Paganin et al., 2002'''
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

