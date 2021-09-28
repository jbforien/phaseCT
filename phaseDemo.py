import sys
sys.path.insert(0, r"your path to phase retrieval directory here")

import c_phaseRetrieval_v3
import findAlphaFit
from LTTserver import LTTserver
from distutils.version import StrictVersion 
import numpy as np
import os
import glob
from natsort import natsorted
import joblib
from joblib import Parallel, delayed
import multiprocessing
import warnings
from PIL import Image
import datetime
import time
import matplotlib.pyplot as plt
import matplotlib
from skimage import io
import pyfftw #requires scipy 1.3.3 (pip install scipy==1.3.3)
from scipy import interpolate
from scipy.optimize import curve_fit
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

#choose type of phase retrieval algorithm to use
'''        CTF - CTFPurePhase - CTFPurePhaseWithAbs - homoCTF            '''
'''        TIE - WTIE                                                    '''
'''        Paganin - multiPaganin                                        '''
'''        mixedAppr_homo                                                '''

phaseRetAlgo = 'homoCTF' 
inputDir = r"main directory path"
pathProj1 = inputDir + os.sep + 'folder_1'
pathProj2 = inputDir + os.sep + 'folder_2'
pathProj3 = inputDir + os.sep + 'folder_3'
pathProj4 = inputDir + os.sep + 'folder_4'
listPathDir = [pathProj1,pathProj2,pathProj3,pathProj4]

#adapt experimental setup parameters
'''
Parameter of the experiment
'''
sod = 21000                                             # source to detector distance in mm
dists = [15,40,86,132]                                  # object to detector distance in mm
energy = 35                                             # in keV
pix_width = 0.65e-3                                     # in mm
delta = 1.90297703E-06  
beta = 3.11117105E-08    
nbCPU = 75                                              # in percent

#correct for detector response?
detectCorr = 0                                            # 1 or  'yes'
c1 = 0.01; c2 = 1.00                                    #edge spread function parameters
a1 = 0.9093; a2 = 2.8537; a3 = 49.3503; a4 = 748.8534    #edge spread function parameters
b1 = 0.5884; b2 = 0.3570; b3 = 0.0352                    #edge spread function parameters

#correct for tranverse coherence?
transCoh = 0
sigmaH = 30e-3                                             # horizontal source size in mm
sigmaV = 120e-3                                             # vertical source size in mm

#apply low-/high-frequency filter?
filterr = 0

#find alpha?
alpha = 50
findAlpha = 0                                            # 1 or 0
alpha_range = np.asarray([10**i for i in range(1,20,2)])/1e10

findAlpha_x1 = 0                 
findAlpha_x2 = -1
findAlpha_y1 = 0
findAlpha_y2 = -1

########################################################################
################# NO NEED TO CHANGE BELOW THESE LINES ##################
########################################################################
'''
Create dictionary with parameters
'''

paramList = {'phase retrieval algorithm ': phaseRetAlgo,
             'source object distance': sod,
             'object detector distance': dists,
             'energy': energy,
             'pixel size': pix_width,
             'delta': delta,
             'beta': beta,
             'delta beta ratio': delta/beta,
             'dataset path': listPathDir,
             'detector response corr': detectCorr,
             'a1, b1, c1, a2, b2, c2, a3, b3, a4': [a1, b1, c1, a2, b2, c2, a3, b3, a4],
             'tranverse coherence corr': transCoh,
             'tranverse coherence horizontal sigma': sigmaH,
             'tranverse coherence vertical sigma': sigmaV,
             'low and high frequency filter corr': filterr,
             'regularization parameter alpha': alpha,
             'alpha ROI x1 y1 x2 y2': [findAlpha_x1, findAlpha_y1, findAlpha_x2, findAlpha_y2]}
    
now = datetime.datetime.now()
now = now.strftime('%y%m%d_%H%M%S')

dists = np.asarray(dists)
mag =(sod+dists)/sod
maxMag = np.amax(mag)
dists = (sod*dists) / (sod+dists)
pix_width = pix_width/maxMag
wlen = c_phaseRetrieval_v3.Energy2Wavelength(energy)

totalNbrCores = multiprocessing.cpu_count()                # find the number of cpu available
nbCores = int(totalNbrCores*nbCPU/100)                     # define the number of cpu that will be used

newParamListValue = {'date time': now,
                     'magnification': list(mag),
                     'maximum magnification': maxMag,
                     'corrected distances':list(dists),
                     'corrected pixel size':pix_width,
                     'wavelength': wlen,
                     'available cpu': totalNbrCores,
                     'used cpu': nbCores}
paramList.update(newParamListValue)


'''
Check that listPathDir is list
'''
if type(phaseRetAlgo) == 'str':
    sys.exit("Abort script - Variable listPathDir needs to be a list, add [ ]")
if phaseRetAlgo == 'Paganin':
    if len(listPathDir)>1:
        sys.exit("Abort script - Choose only one distance path in listPathDir")
    if len(dists)!=1:
        sys.exit("Abort script - Choose only onde distance in dists")
elif phaseRetAlgo == 'TIE' or phaseRetAlgo == 'WTIE':
    if len(listPathDir)!=2:
        sys.exit("Abort script - Choose two folders path in listPathDir")
    if len(dists)!=2:
        sys.exit("Abort script - Choose two distances in dists")    

'''
Check that listPathDir as same nbr of variable than dists
'''
if len(listPathDir) != len(dists):
    sys.exit("Abort script - Number of input folder and distance must be equal")


'''
Create array with path of projections
'''
print("Create list of projections")
listFile = []
for i in range(0,len(listPathDir)): #loop through each folder    
    listFile_temp = glob.glob(listPathDir[i]+os.sep+"*.tif")    
    listFile_temp = natsorted(listFile_temp) #natural sorting of the files        
    listFile.append(listFile_temp)
listFile = np.stack(listFile,axis=0).T
# ~ listFile = listFile[::200,:]    #use this to consider only a subset of projections


'''
determine padding to add so that projections are padded to the next power of 2
'''
print("Determine padding")
dummy_img = np.asarray(Image.open(listFile[0,0]))
pad_x = int( (np.power(2,np.ceil(np.log2(dummy_img.shape[1]))) - dummy_img.shape[1]) / 2 )
pad_y = int( (np.power(2,np.ceil(np.log2(dummy_img.shape[0]))) - dummy_img.shape[0]) / 2 )
if (dummy_img.shape[1] % 2) == 0:
    add_x = 0
else:
    add_x = 1
if (dummy_img.shape[0] % 2) == 0:
    add_y = 0
else:
    add_y = 1
dummy_img = np.pad(dummy_img, ((pad_y,pad_y+add_y),(pad_x,pad_x+add_x)), 'edge')

'''
Create complex conjugate
'''
dummy_img_fft = pyfftw.interfaces.numpy_fft.fft2(dummy_img)
print("Creating complex conjugate")

fx = np.fft.fftfreq(dummy_img_fft.shape[1],d=pix_width)
fy = np.fft.fftfreq(dummy_img_fft.shape[0],d=pix_width)

fx, fy = np.meshgrid(fx,fy)


'''
Calculate the contribution of degree of coherence
    based from P.Cloetens PhD thesis eq. 4.16
'''
degCoh = np.ones((dummy_img_fft.shape[0], dummy_img_fft.shape[1], len(dists)))
if transCoh == 1:
    print("Calculate degree of coherence")    
    for i in range(0,len(dists)):
        degCoh[:,:,i] = (np.exp(((-2*(np.pi*(sigmaH/2.35/sod)*dists[i])**2)*fx**2) + 
                        (-2*(np.pi*(sigmaV/2.35/sod)*dists[i])**2)*fy**2) )


'''
Calculate the contribution of the detector
    based from P.Cloetens PhD thesis eq. 3.43
'''
OptTrnFunc = np.ones((dummy_img_fft.shape[0], dummy_img_fft.shape[1]))
if detectCorr == 1:
    print("Calculate optical transfer function of the detector")
    f = fx**2+fy**2
    f = 0.5*(f/np.amax(f))
    OptTrnFunc = c_phaseRetrieval_v3.OTF(f, a1, a2, a3, a4, b1, b2, b3, c1, c2)
    OptTrnFunc[0,0] = 1    


'''
Combine contribution of degree of coherence and detector
'''
Rm = np.zeros((dummy_img_fft.shape[0], dummy_img_fft.shape[1], len(dists)))
for i in range(0,len(dists)):
    Rm[:,:,i] = degCoh[:,:,i] * OptTrnFunc


'''
Create filter for high and low frequencies
'''
fltr = np.zeros((dummy_img.shape[0], dummy_img.shape[1]))
if filterr == 1:
    print("Calculate filter for low- and high-frequencies")
    low = 1e-5
    high = 2e-5
    
    
'''
phase-retrieval algorithm:
CTF                    (rads, wlen, dists, fx, fy, Rm alpha)
CTFPurePhase        (rads, wlen, dists, delta, beta, fx, fy, Rm, alpha)
CTFPurePhaseWithAbs    (rads, wlen, dists, delta, beta, fx, fy, Rm, alpha)
homoCTF            (rads, wlen, dists, delta, beta, fx, fy, Rm alpha)
TIE                    (rads, wlen, dists, pix_width, fx, fy, Rm, alpha)
WTIE                (rads, wlen, dists, pix_width, fx, fy, Rm, alpha)
Paganin                (rad,  wlen, dist,  delta, beta, fx, fy, Rm)
multiPaganin        (rads, wlen, dists, delta, beta, fx, fy, Rm, alpha)
'''
def phaseRetrieval_task(projsPath, wlen, dists, pix_width, fx, fy, Rm, alpha, outPath=0, rtrnRslt=0):
    projs = []
    for dist in range(len(dists)):
        img = np.exp(-1*np.asarray(Image.open(projsPath[dist])))
        #~ img = np.asarray(Image.open(projsPath[dist]))
        img = np.pad(img, ((pad_y,pad_y+add_y),(pad_x,pad_x+add_x)), 'edge')
        projs.append(img)
    
    
    if phaseRetAlgo == 'CTF':
        phase = c_phaseRetrieval_v3.CTF(projs, wlen, dists, fx, fy, Rm, alpha)
    elif phaseRetAlgo == 'CTFPurePhase':
        phase = c_phaseRetrieval_v3.CTFPurePhase(projs, wlen, dists, delta, beta, fx, fy, Rm, alpha)
    if phaseRetAlgo == 'CTFPurePhaseWithAbs':
        phase = c_phaseRetrieval_v3.CTFPurePhaseWithAbs(projs, wlen, dists, delta, beta, fx, fy, Rm, alpha)
    elif phaseRetAlgo == 'CTFPurePhaseWithAbs':
        phase = c_phaseRetrieval_v3.CTFPurePhaseWithAbs(projs, wlen, dists, delta, beta, fx, fy, Rm, alpha)
    if phaseRetAlgo == 'homoCTF':
        phase = c_phaseRetrieval_v3.homoCTF(projs, wlen, dists, delta, beta, fx, fy, Rm, alpha)
    elif phaseRetAlgo == 'TIE':
        phase = c_phaseRetrieval_v3.TIE(projs, wlen, dists, pix_width, fx, fy, Rm, alpha)
    elif phaseRetAlgo == 'WTIE':
        phase = c_phaseRetrieval_v3.WTIE(projs, wlen, dists, pix_width, fx, fy, Rm, alpha)
    elif phaseRetAlgo == 'Paganin':
        phase = c_phaseRetrieval_v3.Paganin(projs[0],  wlen, dists[0],  delta, beta, fx, fy, Rm[:,:,0])
    elif phaseRetAlgo == 'multiPaganin':
        phase = c_phaseRetrieval_v3.multiPaganin(projs, wlen, dists, delta, beta, fx, fy, Rm, alpha)
    elif phaseRetAlgo == 'mixedAppr_homo':
        phase = c_phaseRetrieval_v3.mixedAppr_homo(projs, wlen, dists, pix_width, delta, beta, fx, fy, Rm, alpha)

    phase = phase[pad_y:-pad_y-add_y,pad_x:-pad_x-add_x]        
    
    if outPath != 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            io.imsave(outPath+'.tif', phase.astype('float32'))    
    
    if rtrnRslt != 0:
        return phase

'''
Find Alpha
'''
from skimage.filters.rank import entropy
from skimage.morphology import disk
if findAlpha == 1:
    print("Find Alpha")    
    alpha_list = []
    fwhm_list = []
    for proj in range(0, listFile.shape[0], 250):
        
        r = []    
        r = (Parallel(n_jobs=nbCores)(delayed(phaseRetrieval_task)(listFile[proj],
                wlen, dists, pix_width, fx, fy, Rm, alpha, 0, 1) for alpha in alpha_range))                
        r = np.stack(r,axis=0)

        io.imsave(inputDir + os.sep + now + '_' + phaseRetAlgo 
                + '_alphaStack_' + str(proj) + '.tif', r.astype('float32'))
                
        stdDev = []
        entr = []
        for i in range(r.shape[0]):
            stdDev.append(np.nanstd(r[i,findAlpha_y1:findAlpha_y2,findAlpha_x1:findAlpha_x2]))

        alphaTmp = findAlphaFit.SymSigmoid(stdDev, alpha_range)
        alpha_list.append(alphaTmp)
            
    alpha_list = np.stack(alpha_list, axis=0)
    alpha = np.mean(alpha_list)
    print(alpha)



'''
create output folder and run algorithm
'''
print("Phase-retrieval")
if phaseRetAlgo == 'Paganin':
    ouputDir = inputDir + os.sep + now + "_phase_" + phaseRetAlgo + "_deltabeta_"    + str(np.int(delta/beta)) #create folder where phase projections will be saved
else:
    ouputDir = inputDir + os.sep + now + "_phase_" + phaseRetAlgo + "_deltabeta_"    + str(np.int(delta/beta)) + "_alpha_" + str('{:0.2e}'.format(alpha))    #create folder where phase projections will be saved

ouputDir = ouputDir.replace(".","p")
ouputDir = ouputDir.replace("+","p")
ouputDir = ouputDir.replace("-","m")
os.mkdir(ouputDir)
outPath     = [(str(ouputDir)+os.sep+"PHASE_"+str(n).zfill(4)) for n in range(listFile.shape[0])]
Parallel(n_jobs=nbCores)(delayed(phaseRetrieval_task)(listFile[proj], 
                         wlen, dists, pix_width, fx, fy, Rm, alpha, outPath[proj], 0) 
                         for proj in tqdm(range(0, listFile.shape[0])))

'''
write parameter file
'''
with open(ouputDir+os.sep+'retrievalParam.txt', 'w') as f: 
    for key, value in paramList.items(): 
        f.write('%s: %s\n' % (key, value))
