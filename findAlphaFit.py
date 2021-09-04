import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


''' Symmetric sigmoid function '''
def SymSigmoidFunc(x, *p0):
    return  (p0[0] * np.exp(-(x-p0[1])/p0[2])) / (np.exp(-(x-p0[1])/p0[2]) + 1)  + p0[3]

''' Find Alpha '''
def SymSigmoid(stdevArr, alphaArr):
    ''' determine initil guess'''
    stdevArrGrad = np.gradient(stdevArr)
    argMinStdevGrad = np.argmin(stdevArrGrad)  #estimate for SymSigmoidFunc center
    amplitude = np.amax(stdevArr) - np.amin(stdevArr) #estimate for SymSigmoidFunc amplitude
    
    ''' perform fitting in linear scale'''
    # p0 is the initial guess for the fitting coefficients (amplitude, center, stdDev, baseline)
    p0 = np.asarray([amplitude, argMinStdevGrad, 1., np.amin(stdevArr)])
    alphaArr_lin = np.asarray([np.log10(i) for i in alphaArr]) #transform to linear scale
    pFit, cov = curve_fit(SymSigmoidFunc, alphaArr_lin, stdevArr, p0=p0, maxfev=6000) #perform fitting
    
    ''' 10**pFit[1]               '''    # x-result in log space
    ''' SymSigmoidFunc(pFit[1]    '''    # y-result
    
    ''' plotting results in linear scale'''
    #~ plt.scatter(alphaArr_lin, stdevArr, marker='o', alpha=0.7, c ='red')    
    #~ plt.scatter(alphaArr_lin, SymSigmoidFunc(alphaArr_lin, *pFit), c ='black', s=10)
    #~ plt.xlabel('alpha - linear scale (-)')
    #~ plt.ylabel('stdev (-)')
    #~ plt.show()
    #~ plt.close()

    ''' convert linear scale fitting results to log-scale'''
    AccuracyFactor = 10.
    xInterpoSigLin = np.arange(np.amin(alphaArr_lin),np.amax(alphaArr_lin),1/AccuracyFactor)
    yfitSigmoid = SymSigmoidFunc(xInterpoSigLin,*[pFit[0],pFit[1],pFit[2],pFit[3]])
    xInterpoSigLog = 10**xInterpoSigLin #np.logspace(2, 15, num=len(xfitSigmoid))/ 1e8
    
    print("alpha found at: "+str(10**pFit[1]))
    
    '''plotting results in log-scale'''
    plt.scatter(alphaArr, stdevArr, marker='o', alpha=0.7, c ='red')    
    plt.plot(xInterpoSigLog, yfitSigmoid, '-', c ='black', label = 'fit')    
    plt.scatter(10**pFit[1], SymSigmoidFunc(pFit[1], *pFit), c ='black')
    plt.scatter(10**(pFit[1]+4*pFit[2]), SymSigmoidFunc(pFit[1]+4*pFit[2], *pFit), c ='blue')
    plt.xlabel('alpha (-)')
    plt.ylabel('stdev - gradient (-)')
    plt.xscale("log")
    plt.show()
    plt.close()
    
    return 10**pFit[1]
