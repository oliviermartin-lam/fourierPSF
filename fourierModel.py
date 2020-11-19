#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep  1 16:31:39 2020

@author: omartin
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:00:44 2018

@author: omartin
"""
import numpy as np
import matplotlib as mpl

import numpy.fft as fft
import matplotlib.pyplot as plt
import scipy.special as spc
import os.path as ospath

import math
import time
import os
import sys

from astropy.io import fits
from configparser import ConfigParser
from distutils.spawn import find_executable

import FourierUtils
from telescope import telescope
from atmosphere import atmosphere
from source import source

   
#%% DISPLAY FEATURES
mpl.rcParams['font.size'] = 16

if find_executable('tex'): 
    usetex = True
else:
    usetex = False

plt.rcParams.update({
    "text.usetex": usetex,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
 
#%%
            
def demoMavisPSD():
    # Instantiate the FourierModel class
    t0 = time.time()
    if sys.platform[0:3] == 'win':
        fao = fourierModel(os.getcwd()+"\parFile\mavisParams.ini",calcPSF=False,verbose=True,display=False,getErrorBreakDown=False)
    else:
        fao = fourierModel(os.getcwd()+"/parFile/mavisParams.ini",calcPSF=False,verbose=True,display=False,getErrorBreakDown=False)
    PSD = fao.powerSpectrumDensity()
    ttot = time.time() - t0
    print("Total calculation time - {:d} PSD (s)\t : {:f} ".format(fao.nSrc,ttot))
    return PSD

def demoMavisPSF():
    if sys.platform[0:3] == 'win':
        fao = fourierModel(os.getcwd()+"\parFile\mavisParams.ini",calcPSF=True,verbose=True,display=True,getErrorBreakDown=False)
    else:
        fao = fourierModel(os.getcwd()+"/parFile/mavisParams.ini",calcPSF=True,verbose=True,display=True,getErrorBreakDown=False)
    return fao

def demoHarmoniPSF():
    if sys.platform[0:3] == 'win':
        fao = fourierModel(os.getcwd()+"\parFile\harmoniParams.ini",calcPSF=True,verbose=True,display=True,\
                       getErrorBreakDown=False,getPSFMetrics=True,displayContour=True)    
    else:
        fao = fourierModel(os.getcwd()+"/parFile/harmoniParams.ini",calcPSF=True,verbose=True,display=True,\
                       getErrorBreakDown=False,getFWHM=True,getEncircledEnergy=True,getEnsquaredEnergy=False,displayContour=True)    
    return fao

def demoHarmoniSCAOPSF():
    if sys.platform[0:3] == 'win':
        fao = fourierModel(os.getcwd()+"\parFile\harmoniSCAOParams.ini",calcPSF=True,verbose=True,display=True,\
                       getErrorBreakDown=False,getPSFMetrics=True,displayContour=True)    
    else:
        fao = fourierModel(os.getcwd()+"/parFile/harmoniSCAOParams.ini",calcPSF=True,verbose=True,display=True,\
                       getErrorBreakDown=True,getFWHM=False,getEncircledEnergy=False,getEnsquaredEnergy=False,displayContour=False)    
    return fao
    
class fourierModel:
    """ Fourier class gathering the PSD calculation for PSF reconstruction. 
    """
    
    # DEPENDANT VARIABLES DEFINITION
    @property
    def kc(self):
        """Cut-of frequency"""
        if self.kcExt.any():
            return self.kcExt
        else:
            return 1/(2*max(self.pitchs_dm.min(),self.pitchs_wfs.min()))
    
    @property
    def kcDM(self):
        """DM cut-of frequency"""
        return 1/(2*self.pitchs_dm)
    
    @property
    def kcInMas(self):
        """DM cut-of frequency"""
        radian2mas = 180*3600*1e3/np.pi
        return self.kc*self.atm.wvl*radian2mas;
    
    @property
    def nTimes(self):
        """"""
        #return int(np.round(max([self.fovInPixel,2*self.resAO])/self.resAO))
        return min(2,math.ceil(self.fovInPixel/self.resAO/2))

    # CONTRUCTOR
    def __init__(self,file,calcPSF=True,verbose=False,display=True,aoFilter='circle',\
                 getErrorBreakDown=False,getFWHM=False,getEnsquaredEnergy=False,getEncircledEnergy=False\
                 ,displayContour=False, extraPSFsDirections=[],kcExt=[]):
        
        tstart = time.time()
        # PARSING INPUTS
        self.verbose = verbose
        self.status = 0
        self.file   = file  
        self.display = display
        self.getErrorBreakDown = getErrorBreakDown
        self.getPSFmetrics = getFWHM or getEnsquaredEnergy or getEncircledEnergy
        self.calcPSF = calcPSF
        self.kcExt   = np.array(kcExt)
        
        # GRAB PARAMETERS
        self.status = self.parameters(self.file,extraPSFsDirections=extraPSFsDirections)        
        
        if self.status:
            # DEFINE THE FREQUENCY VECTORS WITHIN THE AO CORRECTION BAND
            kx = self.resAO*self.PSDstep*fft.fftshift(fft.fftfreq(self.resAO)) + 1e-10
            ky = self.resAO*self.PSDstep*fft.fftshift(fft.fftfreq(self.resAO)) + 1e-10
            self.kx,self.ky = np.meshgrid(kx,ky)
            self.kxy        = np.hypot(self.kx,self.ky)    
            
            # DEFINE THE PISTON FILTER FOR LOW-ORDER FREQUENCIES
            self.pistonFilterIn_ = FourierUtils.pistonFilter(self.tel.D,self.kxy)
            
            # DEFINE THE FREQUENCY DOMAIN OVER THE FULL PSD DOMAIN
            kxExt           = self.fovInPixel*self.PSDstep*fft.fftshift(fft.fftfreq(self.fovInPixel))
            kyExt           = self.fovInPixel*self.PSDstep*fft.fftshift(fft.fftfreq(self.fovInPixel))
            self.kxExt,self.kyExt = np.meshgrid(kxExt,kyExt)
            self.kExtxy     = np.hypot(self.kxExt,self.kyExt)           
            
            # DEFINE THE AO CORRECTION and PSF HALO  REGIONS
            if aoFilter == 'circle':
                self.mskIn_  = self.kxy  <= self.kc      
                self.mskOut_ = np.hypot(self.kxExt,self.kyExt) > self.kc
            else:
                self.mskIn_  = (abs(self.kx) <= self.kc) | (abs(self.ky) <= self.kc)    
                self.mskOut_ = (abs(self.kxExt)>self.kc) | (abs(self.kyExt)>self.kc)         
            
            # DEFINE NOISE AND ATMOSPHERE PSD
            self.Wn   = np.mean(self.noiseVariance)/(2*self.kc)**2
            self.Wphi = self.atm.spectrum(self.kxy);
            
            # DEFINE THE OTF TELESCOPE
            P                   = np.zeros((self.fovInPixel,self.fovInPixel))
            id1                 = np.floor(self.fovInPixel/2 - self.tel.resolution/2).astype(int)
            id2                 = np.floor(self.fovInPixel/2 + self.tel.resolution/2).astype(int)
            P[id1:id2,id1:id2]  = self.tel.pupil
            if self.path_static != []:
                obj = fits.open(self.path_static)
                im  = obj[0].data
                obj.close()
                im[np.where(im!=im)] = 0
                im              = FourierUtils.interpolateSupport(im,self.resolution,kind='linear')
                self.staticMap  = np.zeros((self.fovInPixel,self.fovInPixel))
                self.staticMap[id1:id2,id1:id2]  = im
                P               = P*np.exp(-2*complex(0,1)*np.pi*self.staticMap/self.wvlSrc)
            
            self.otfTel         = np.real(fft.fftshift(FourierUtils.fftCorrel(P,P)))
            self.otfTel         = self.otfTel/self.otfTel.max()
        
           
            # DEFINE THE RECONSTRUCTOR
            wvl                 = self.wvlRef
            self.atm.wvl        = wvl
            self.atm_mod.wvl    = wvl
            if self.nGs <2:
                self.reconstructionFilter()
            else:
                self.finalReconstructor()
            
            # DEFINE THE CONTROLLER
            self.controller()
            
            
            # INIT OUTPUTS
            self.PSF = []
            self.PSD = []
            self.SR  = []
            self.FWHM= []
            self.EncE= []
            self.EnsqE= []
 
            if calcPSF:
                self.getPSF(verbose=verbose,getErrorBreakDown=getErrorBreakDown,\
                            getFWHM=getFWHM,getEnsquaredEnergy=getEnsquaredEnergy,getEncircledEnergy=getEncircledEnergy)
                if display:
                    self.displayResults(displayContour=displayContour)
            else:
                self.psd = self.powerSpectrumDensity()
                
            self.t_init = 1000*(time.time()  - tstart)
            
            if verbose:
                self.displayExecutionTime()
            
          
    def __repr__(self):
        s = "Fourier Model class "
        if self.status == 1:
            s = s + "instantiated"
        else:
            s = s + "not instantiated"
        
        #self.displayResults()
        
        return s

    def parameters(self,file,extraPSFsDirections=[]):
                    
        tstart = time.time() 
    
        # verify if the file exists
        if ospath.isfile(file) == False:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The .ini file does not exist\n')
            return 0
        
        # open the .ini file
        config = ConfigParser()
        config.optionxform = str
        config.read(file)
        
        #%% Telescope
        self.D              = eval(config['telescope']['TelescopeDiameter'])
        self.zenith_angle   = eval(config['telescope']['zenithAngle'])
        self.obsRatio       = eval(config['telescope']['obscurationRatio'])
        self.resolution     = eval(config['telescope']['resolution'])
        self.path_pupil     = eval(config['telescope']['path_pupil'])
        self.path_static    = eval(config['telescope']['path_static'])
        
        #%% Atmosphere
        rad2arcsec          = 3600*180/np.pi 
        rad2mas             = 1e3*rad2arcsec
        self.wvlAtm         = eval(config['atmosphere']['atmosphereWavelength']) 
        self.r0             = 0.976*self.wvlAtm/eval(config['atmosphere']['seeing'])*rad2arcsec
        self.L0             = eval(config['atmosphere']['L0']) 
        self.weights        = np.array(eval(config['atmosphere']['Cn2Weights']) )
        self.heights        = np.array(eval(config['atmosphere']['Cn2Heights']) )
        self.wSpeed         = np.array(eval(config['atmosphere']['wSpeed']) )
        self.wDir           = np.array(eval(config['atmosphere']['wDir']) )
        self.nLayersReconstructed = eval(config['atmosphere']['nLayersReconstructed'])
        #-----  verification
        if len(self.weights) == len(self.heights) == len(self.wSpeed) == len(self.wDir):
            self.nbLayers = len(self.weights)
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of atmospheric layers is not consistent in the parameters file\n')
            return 0
        
        #%% PSF directions
        self.nSrc           = len(np.array(eval(config['PSF_DIRECTIONS']['ScienceZenith'])))
        self.wvlSrc         = np.unique(np.array(eval(config['PSF_DIRECTIONS']['ScienceWavelength'])))
        self.nWvl           = self.wvlSrc.size
        self.wvlRef         = self.wvlSrc.min()
        self.zenithSrc      = np.array(np.array(eval(config['PSF_DIRECTIONS']['ScienceZenith'])))
        self.azimuthSrc     = np.array(np.array(eval(config['PSF_DIRECTIONS']['ScienceAzimuth'])))
        
        # INCLUDE THE ADDITIONAL PSF EVALUATIONS
        if np.any(extraPSFsDirections):
            self.nExtraSrc = len(extraPSFsDirections)
            tmp = np.zeros(self.nSrc + self.nExtraSrc)
            tmp[0:self.nSrc-1] = self.zenithSrc
            self.zenithSrc = tmp
            tmp = np.zeros(self.nSrc + self.nExtraSrc)
            tmp[0:self.nSrc-1] = self.azimuthSrc
            self.azimuthSrc = tmp
            
            for j in range(self.nExtraSrc):
                self.zenithSrc[self.nSrc+j] = extraPSFsDirections[j][0]
                self.azimuthSrc[self.nSrc+j] = extraPSFsDirections[j][1]
            
            self.nSrc = self.nSrc + self.nExtraSrc
          
            
        # ----- verification
        self.src = []
        if len(self.zenithSrc) == len(self.azimuthSrc):
            self.nSrc = len(self.zenithSrc)
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of scientific sources is not consistent in the parameters file\n')
            return 0
        
        #%% Guide stars
        self.nGs            = len(eval(config['GUIDESTARS_HO']['GuideStarZenith_HO']))
        self.zenithGs       = np.array(eval(config['GUIDESTARS_HO']['GuideStarZenith_HO']))
        self.azimuthGs      = np.array(eval(config['GUIDESTARS_HO']['GuideStarAzimuth_HO']))
        self.heightGs       = eval(config['GUIDESTARS_HO']['GuideStarHeight_HO'])
        # ----- verification
        if len(self.zenithGs) == len(self.azimuthGs):
            self.nGs = len(self.zenithGs)
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of guide stars for high-order sensing is not consistent in the parameters file\n')
            return 0
        
        #%% WFS parameters
        self.loopGain       = eval(config['SENSOR_HO']['loopGain_HO'])
        self.samplingTime   = 1/eval(config['SENSOR_HO']['SensorFrameRate_HO'])
        # Note : so far, the WFSs have all the same exposure time
        self.latency        = eval(config['SENSOR_HO']['loopDelaySteps_HO'])*self.samplingTime
        self.wvlGs          = eval(config['SENSOR_HO']['SensingWavelength_HO'])
        self.nLenslet_HO    = eval(config['SENSOR_HO']['nLenslet_HO'])
        self.nph_HO         = eval(config['SENSOR_HO']['nph_HO'])
        self.pixel_Scale_HO = eval(config['SENSOR_HO']['pixel_scale_HO'])
        self.sigmaRON_HO    = eval(config['SENSOR_HO']['sigmaRON_HO'])
        # Note : so far, the WFSs have all the same subaperture size
        self.pitchs_wfs     = self.D/self.nLenslet_HO * np.ones(self.nGs)
        
        # Calculate the noise variance
        if config.has_option('SENSOR_HO', 'noiseVariance_HO') and np.isscalar( eval(config['SENSOR_HO']['noiseVariance_HO'])):
            self.noiseVariance  = eval(config['SENSOR_HO']['noiseVariance_HO'])
            self.noiseVariance = self.noiseVariance * np.ones(self.nGs)    
        else:
            self.Npix_per_subap_HO = int(self.resolution/self.nLenslet_HO)
            if self.pixel_Scale_HO > 1: # put the value in arcsec
                    self.pixel_Scale_HO = self.pixel_Scale_HO/1e3
            self.ND             = self.wvlGs/self.pitchs_wfs*rad2arcsec/self.pixel_Scale_HO #spot FWHM in pixels and without turbulence
            varRON              = np.pi**2/3*(self.sigmaRON_HO /self.nph_HO)**2*(self.Npix_per_subap_HO**2/self.ND)**2
            if varRON.any() > 3:
                print('The read-out noise variance is very high (%.1f >3 rd^2), there is certainly smth wrong with your inputs, set to 0'%(varRON))
                varRON = 0
                
            self.NT             = self.wvlGs/self.r0*(self.wvlGs/self.wvlAtm)**1.2 * rad2arcsec/self.pixel_Scale_HO
            varShot             = np.pi**2/(2*self.nph_HO)*(self.NT/self.ND)**2
            if varShot.any() > 3:
                print('The shot noise variance is very high (%.1f >3 rd^2), there is certainly smth wrong with your inputs, set to 0'%(varShot))
                varShot = 0
            self.noiseVariance  = (self.wvlGs/self.wvlRef)**2 * (varRON + varShot)
        
        #%% DM parameters
        self.h_dm           = np.array(eval(config['DM']['DmHeights']))
        self.pitchs_dm      = np.array(eval(config['DM']['DmPitchs']))
        self.zenithOpt      = np.array(eval(config['DM']['OptimizationZenith']))
        self.azimuthOpt     = np.array(eval(config['DM']['OptimizationAzimuth']))
        self.weightOpt      = np.array(eval(config['DM']['OptimizationWeight']))
        self.weightOpt      = self.weightOpt/self.weightOpt.sum()
        self.condmax_tomo   = eval(config['DM']['OptimizationConditioning'])
        self.condmax_popt   = eval(config['DM']['OptimizationConditioning'])
        
        #%% Sampling and field of view
        self.psf_FoV        = eval(config['PSF_DIRECTIONS']['psf_FoV'])
        lonD                = rad2mas*self.wvlRef/self.D
        self.psInMas        = eval(config['PSF_DIRECTIONS']['psInMas'])
        if self.psInMas == 0:
            self.psInMas    = lonD/2
          
        self.samp       = lonD/self.psInMas/2
        if self.samp > 0.98 and self.samp < 1.02:
            self.samp   = 1.0
            
        if self.samp >=1:
            self.fovInPixel = int(max(self.psf_FoV,round(self.resolution*self.samp*2)))
        else:
            self.fovInPixel = int(max(self.psf_FoV,round(2*self.resolution/self.samp)))
        
        if self.verbose:
            print('.Field of view:\t\t%4.2f arcsec\n.Pixel scale:\t\t%4.2f mas\n.Over-sampling:\t\t%4.2f'%(self.fovInPixel*self.psInMas/1e3,self.psInMas,self.samp))
            print('\n-------------------------------------------\n')
        
        self.PSDstep  = self.psInMas/self.wvlRef/rad2mas
        #self.resAO    = int(1/np.min(self.pitchs_dm)/self.PSDstep)
        self.resAO    = int(2*self.kc/self.PSDstep)
        
        #%% instantiating sub-classes
        
        # Telescope
        self.tel = telescope(self.D,self.zenith_angle,self.obsRatio,self.resolution,self.path_pupil)
        
        # Strechning factor (LGS case)      
        self.r0       = self.r0*self.tel.airmass**(-3/5)
        self.heights  = self.heights*self.tel.airmass
        self.heightGs = self.heightGs*self.tel.airmass # LGS height
        if self.heightGs > 0:
            self.heights = self.heights/(1 - self.heights/self.heightGs)
                    
        # Model atmosphere
        self.r0_mod         = self.r0
        self.L0_mod         = self.L0
        
        if self.nLayersReconstructed < len(self.weights):
            self.weights_mod,self.heights_mod = FourierUtils.eqLayers(self.weights,self.heights,self.nLayersReconstructed)
            self.wSpeed_mod = np.linspace(min(self.wSpeed),max(self.wSpeed),num=self.nLayersReconstructed)
            self.wDir_mod   = np.linspace(min(self.wDir),max(self.wDir),num=self.nLayersReconstructed)
        else:
            self.weights_mod    = self.weights
            self.heights_mod    = self.heights
            self.wSpeed_mod     = self.wSpeed
            self.wDir_mod       = self.wDir
            
        # Atmosphere
        self.atm = atmosphere(self.wvlAtm,self.r0,self.weights,self.heights,self.wSpeed,self.wDir,self.L0)
        self.atm_mod = atmosphere(self.wvlAtm,self.r0_mod,self.weights_mod,self.heights_mod,self.wSpeed_mod,self.wDir_mod,self.L0_mod)
        self.atm.wvl = self.wvlRef
        self.atm_mod.wvl = self.wvlRef
        
        # Scientific Sources
        self.src = [source(0,0,0) for k in range(self.nSrc)]  
        for n in range(self.nSrc):
            self.src[n] = source(self.wvlRef,self.zenithSrc[n],self.azimuthSrc[n],0,n+1,"SCIENTIFIC STAR",verbose=True)
                   
        # Guide stars
        #self.gs = []
        self.gs = [source(0,0,0) for k in range(self.nGs)]  
        for n in range(self.nGs):
            self.gs[n] = source(self.wvlGs,self.zenithGs[n],self.azimuthGs[n],self.heightGs,n+1,"GUIDE STAR",verbose=True)
            
        self.t_getParam = 1000*(time.time() - tstart)
        
        return 1
    
#%% RECONSTRUCTOR DEFINITION    
    def reconstructionFilter(self,MV=0):
        """
        """          
        tstart = time.time()
        # reconstructor derivation
        i           = complex(0,1)
        d           = self.pitchs_wfs[0]   
        Sx          = 2*i*np.pi*self.kx*d
        Sy          = 2*i*np.pi*self.ky*d                        
        Av          = np.sinc(d*self.kx)*np.sinc(d*self.ky)*np.exp(i*np.pi*d*(self.kx+self.ky))        
        self.SxAv   = Sx*Av
        self.SyAv   = Sy*Av
        gPSD        = abs(self.SxAv)**2 + abs(self.SyAv)**2 + MV*self.Wn/self.Wphi
        self.Rx     = np.conj(self.SxAv)/gPSD
        self.Ry     = np.conj(self.SyAv)/gPSD
                
        # Manage NAN value if any   
        self.Rx[np.isnan(self.Rx)] = 0
        self.Ry[np.isnan(self.Ry)] = 0
            
        # Set central point (i.e. kx=0,ky=0) to zero
        N = int(np.ceil((self.kx.shape[0]-1)/2))
        self.Rx[N,N] = 0
        self.Ry[N,N] = 0
        self.t_reconstructor = 1000*(time.time()  - tstart)
        
    def tomographicReconstructor(self):
        
        tstart  = time.time()
        nK      = self.resAO
        nL      = len(self.heights)
        nL_mod  = len(self.heights_mod)
        nGs     = self.nGs
        i       = complex(0,1)
        d       = self.pitchs_wfs   #sub-aperture size      
        
         # WFS operator and projection matrices
        M     = np.zeros([nK,nK,nGs,nGs],dtype=complex)
        P     = np.zeros([nK,nK,nGs,nL_mod],dtype=complex)         
        for j in range(nGs):
            M[:,:,j,j] = 2*i*np.pi*self.kxy*np.sinc(d[j]*self.kx)*np.sinc(d[j]*self.ky)
            for n in range(nL_mod):
                P[:,:,j,n] = np.exp(i*2*np.pi*self.heights_mod[n]*(self.kx*self.gs[j].direction[0] + self.ky*self.gs[j].direction[1]))
        self.M = M
        MP = np.matmul(self.M,P)
        MP_t = np.conj(MP.transpose(0,1,3,2))
        
        # Noise covariance matrix
        self.Cb = np.ones((nK,nK,nGs,nGs))*np.diag(self.noiseVariance)
        
        # Atmospheric PSD with the true atmosphere
        self.Cphi   = np.zeros([nK,nK,nL,nL],dtype=complex)
        cte         = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2./(2.*np.pi**(11/3)))
        kernel      = self.atm.r0**(-5/3)*cte*(self.kxy**2 + 1/self.atm.L0**2)**(-11/6)*self.pistonFilterIn_
        self.Cphi   = kernel.repeat(nL**2,axis=1).reshape((nK,nK,nL,nL))*np.diag(self.weights)
        
        # Atmospheric PSD with the modelled atmosphere
        if nL_mod == nL:
            self.Cphi_mod = self.Cphi
        else:
            self.Cphi_mod = kernel.repeat(nL_mod**2,axis=1).reshape((nK,nK,nL_mod,nL_mod))*np.diag(self.weights_mod)
        to_inv  = np.matmul(np.matmul(MP,self.Cphi_mod),MP_t) + self.Cb 
        
        # Wtomo
        inv = np.linalg.pinv(to_inv,rcond=1/self.condmax_tomo)
        Wtomo = np.matmul(np.matmul(self.Cphi_mod,MP_t),inv)        
        self.t_tomo = 1000*(time.time() - tstart)
        
        return Wtomo
 
    def optimalProjector(self):
        
        tstart = time.time()
        nDm     = len(self.h_dm)
        nDir    = (len(self.zenithOpt))
        nL      = len(self.heights_mod)
        nK      = self.resAO
        i       = complex(0,1)
        
        mat1    = np.zeros([nK,nK,nDm,nL],dtype=complex)
        to_inv  = np.zeros([nK,nK,nDm,nDm],dtype=complex)
        theta_x = self.zenithOpt/206264.8 * np.cos(self.azimuthOpt*np.pi/180)
        theta_y = self.zenithOpt/206264.8 * np.sin(self.azimuthOpt*np.pi/180)
        
        for d_o in range(nDir):                 #loop on optimization directions
            Pdm = np.zeros([nK,nK,1,nDm],dtype=complex)
            Pl  = np.zeros([nK,nK,1,nL],dtype=complex)
            fx  = theta_x[d_o]*self.kx
            fy  = theta_y[d_o]*self.ky
            for j in range(nDm):                # loop on DM
                index   = self.kxy <= self.kcDM[j]
                Pdm[index,0,j] = np.exp(i*2*np.pi*self.h_dm[j]*(fx[index]+fy[index]))
            Pdm_t = np.conj(Pdm.transpose(0,1,3,2))
            for l in range(nL):                 #loop on atmosphere layers
                Pl[:,:,0,l] = np.exp(i*2*np.pi*self.heights_mod[l]*(fx + fy))
                
            mat1   += np.matmul(Pdm_t,Pl)*self.weightOpt[d_o]
            to_inv += np.matmul(Pdm_t,Pdm)*self.weightOpt[d_o]
            
        # Popt
        mat2 = np.linalg.pinv(to_inv,rcond=1/self.condmax_popt)
        Popt = np.matmul(mat2,mat1)
        
        self.t_opt = 1000*(time.time() - tstart)
        return Popt
 
    def finalReconstructor(self):
        tstart  = time.time()
        
        self.Wtomo  = self.tomographicReconstructor()
        self.Popt   = self.optimalProjector()
        self.W      = np.matmul(self.Popt,self.Wtomo)
        
        # Computation of the Pbeta^DM matrix
        nDm     = len(self.h_dm)
        nK      = self.resAO
        i       = complex(0,1)
        nK      = self.resAO
        nH      = self.nbLayers
        Hs      = self.heights
        d       = self.pitchs_dm[0]
            
        self.PbetaDM = []
        for s in range(self.nSrc):
            fx = self.src[s].direction[0]*self.kx
            fy = self.src[s].direction[1]*self.ky
            PbetaDM = np.zeros([nK,nK,1,nDm],dtype=complex)
            for j in range(nDm): #loop on DMs
                index               = self.kxy <= self.kcDM[j]
                PbetaDM[index,0,j]  = np.exp(2*i*np.pi*self.h_dm[j]*(fx[index] + fy[index]))
            self.PbetaDM.append(PbetaDM)
        
        # Computation of the Malpha matrix
        self.wDir_x  = np.cos(self.wDir*np.pi/180)
        self.wDir_y  = np.sin(self.wDir*np.pi/180)
        self.MPalphaL = np.zeros([nK,nK,self.nGs,nH],dtype=complex)
        for h in range(nH):
            www = np.sinc(self.samplingTime*self.wSpeed[h]*(self.wDir_x[h]*self.kx + self.wDir_y[h]*self.ky))
            for g in range(self.nGs):
                Alpha = [self.gs[g].direction[0],self.gs[g].direction[1]]
                fx = Alpha[0]*self.kx
                fy = Alpha[1]*self.ky
                self.MPalphaL[self.mskIn_,g,h] = www[self.mskIn_]*2*i*np.pi*self.kxy[self.mskIn_]*np.sinc(d*self.kx[self.mskIn_])*\
                    np.sinc(d*self.ky[self.mskIn_])*np.exp(i*2*np.pi*Hs[h]*(fx[self.mskIn_]+fy[self.mskIn_]))
            
        self.Walpha = np.matmul(self.W,self.MPalphaL)
        self.t_finalReconstructor = 1000*(time.time() - tstart)
        
#%% CONTROLLER DEFINITION
    def  controller(self,nTh=1,nF=500):
        """
        """
        tstart  = time.time()
        i           = complex(0,1)
        vx          = self.atm.wSpeed*np.cos(self.atm.wDir*np.pi/180)
        vy          = self.atm.wSpeed*np.sin(self.atm.wDir*np.pi/180)   
        nPts        = self.resAO
        thetaWind   = np.linspace(0, 2*np.pi-2*np.pi/nTh,nTh)
        costh       = np.cos(thetaWind)
        weights     = self.atm.weights
        Ts          = self.samplingTime
        td          = self.latency        
        delay       = np.floor(td/Ts)
                   
        # Instantiation
        h1          = np.zeros((nPts,nPts))
        h2          = np.zeros((nPts,nPts))
        hn          = np.zeros((nPts,nPts))
        h1buf       = np.zeros((nPts,nPts,nTh))*(1+i)
        h2buf       = np.zeros((nPts,nPts,nTh))
        hnbuf       = np.zeros((nPts,nPts,nTh))
        
        # Get the noise propagation factor
        #import pdb
        #pdb.set_trace()
        f      = np.logspace(-2,np.log10(0.5/Ts),nF)
        z      = np.exp(-2*i*np.pi*f*Ts)
        self.hInt   = self.loopGain/(1.0 - z**(-1.0))
        self.rtfInt = 1.0/(1+self.hInt*z**(-delay))
        self.atfInt = self.hInt*z**(-delay)*self.rtfInt
        
        if self.loopGain == 0:
            self.ntfInt = 1
        else:
            self.ntfInt = self.atfInt/z
                
        self.noiseGain = np.trapz(abs(self.ntfInt)**2,f)*2*Ts
        
        
        # Get transfer functions                                        
        for l in range(self.atm.nL):
            for iTheta in range(nTh):
                fi      = -vx[l]*self.kx*costh[iTheta] - vy[l]*self.ky*costh[iTheta]
                idx     = abs(fi) <1e-7;
                z       = np.exp(-2*i*np.pi*fi*Ts)
                fi[idx] = 1e-8*np.sign(fi[idx])
                hInt    = self.loopGain/(1-z**(-1))
                rtfInt  = 1/(1+hInt*z**(-delay))
                atfInt  = hInt*z**(-delay)*rtfInt
                
                # ao transfer function
                MAG               = abs(atfInt)                
                MAG[fi == 0]      = 1
                PH                = np.angle(atfInt)                
                h2buf[:,:,iTheta] = abs(MAG*np.exp(i*PH))**2
                h1buf[:,:,iTheta] = MAG*np.exp(i*PH)
                # noise transfer function
                if self.loopGain == 0:
                    ntfInt = 1
                else:
                    ntfInt = atfInt/z
                MAG = ntfInt           
                MAG[fi == 0] = 1
                PH  = np.angle(ntfInt)  
                hnbuf[:,:,iTheta] = abs(MAG*np.exp(i*PH))**2
                
            h1 = h1 + weights[l]*np.sum(h1buf,axis=2)/nTh
            h2 = h2 + weights[l]*np.sum(h2buf,axis=2)/nTh
            hn = hn + weights[l]*np.sum(hnbuf,axis=2)/nTh
        
        self.h1 = h1
        self.h2 = h2
        self.hn = hn
        
        if self.display:
            plt.figure()
            plt.semilogx(f,10*np.log10(abs(self.rtfInt)**2),label='Rejection transfer function')
            plt.semilogx(f,10*np.log10(abs(self.ntfInt)**2),label='Noise transfer function')
            plt.semilogx(f,10*np.log10(abs(self.atfInt)**2),label='Aliasing transfer function')
            plt.xlabel('Temporal frequency (Hz)')
            plt.ylabel('Magnitude (dB)')
            plt.legend()
            
        self.t_controller = 1000*(time.time() - tstart)
      
 #%% PSD DEFINTIONS   
    def fittingPSD(self):
        """ FITTINGPSD Fitting error power spectrum density """                 
        tstart  = time.time()
        #Instantiate the function output
        psd                 = np.zeros((self.fovInPixel,self.fovInPixel))
        psd[self.mskOut_]   = self.atm.spectrum(self.kExtxy[self.mskOut_])
        self.t_fittingPSD = 1000*(time.time() - tstart)
        return psd
        
        
    def aliasingPSD(self):
        """
        """
        tstart  = time.time()
        psd = np.zeros((self.resAO,self.resAO))
        i  = complex(0,1)
        d  = self.pitchs_wfs[0]
        T  = self.samplingTime
        td = self.latency        
        vx = self.atm.wSpeed*np.cos(self.atm.wDir*np.pi/180)
        vy = self.atm.wSpeed*np.sin(self.atm.wDir*np.pi/180)
        weights = self.atm.weights  
        w = 2*i*np.pi*d;

        if hasattr(self, 'Rx') == False:
            self.reconstructionFilter()
        Rx = self.Rx*w
        Ry = self.Ry*w
        
        if self.loopGain == 0:
            tf = 1
        else:
            tf = self.h1
            
        
        for mi in range(-self.nTimes,self.nTimes):
            for ni in range(-self.nTimes,self.nTimes):
                if (mi!=0) | (ni!=0):
                    km   = self.kx - mi/d
                    kn   = self.ky - ni/d
                    PR   = FourierUtils.pistonFilter(self.tel.D,np.hypot(km,kn),fm=mi/d,fn=ni/d)
                    W_mn = (km**2 + kn**2 + 1/self.atm.L0**2)**(-11/6)     
                    Q    = (Rx*km + Ry*kn) * (np.sinc(d*km)*np.sinc(d*kn))
                    avr  = 0
                        
                    for l in range(self.atm.nL):
                        avr = avr + weights[l]* (np.sinc(km*vx[l]*T)*np.sinc(kn*vy[l]*T)
                        *np.exp(2*i*np.pi*km*vx[l]*td)*np.exp(2*i*np.pi*kn*vy[l]*td)*tf)
                                                          
                    psd = psd + PR*W_mn * abs(Q*avr)**2
        
        self.t_aliasingPSD = 1000*(time.time() - tstart)
        return self.mskIn_*psd*self.atm.r0**(-5/3)*0.0229 
    
    def noisePSD(self):
        """NOISEPSD Noise error power spectrum density
        """
        tstart  = time.time()
        if self.noiseVariance[0] > 0:
            if self.nGs < 2:        
                psd = np.zeros((self.resAO,self.resAO),dtype=complex)
                psd = self.noiseVariance/(2*self.kc)**2*(abs(self.Rx)**2 + abs(self.Ry**2))
                psd = self.mskIn_ * psd*self.pistonFilterIn_
            else:  
                psd = np.zeros((self.resAO,self.resAO,self.nSrc),dtype=complex)
                #PbetaDM = self.PbetaDMj
                for j in range(self.nSrc):
                    PW      = np.matmul(self.PbetaDM[j],self.W)
                    PW_t    = np.conj(PW.transpose(0,1,3,2))
                    tmp     = np.matmul(PW,np.matmul(self.Cb,PW_t))
                    psd[:,:,j] = self.mskIn_ * tmp[:,:,0,0]*self.pistonFilterIn_
        
        self.t_noisePSD = 1000*(time.time() - tstart)
        return psd*self.noiseGain
    
    def servoLagPSD(self):
        """ SERVOLAGPSD Servo-lag power spectrum density
        """
        tstart  = time.time()    
        psd = np.zeros((self.resAO,self.resAO))    
        if hasattr(self, 'Rx') == False:
            self.reconstructionFilter()

        F = self.Rx*self.SxAv+ self.Ry*self.SyAv     
        Watm = self.mskIn_ * self.Wphi * self.pistonFilterIn_       
        if (self.loopGain == 0):
            psd = abs(1-F)**2*Watm
        else:
            psd = (1 + abs(F)**2*self.h2 - 2*np.real(F*self.h1))*Watm
        
        self.t_servoLagPSD = 1000*(time.time() - tstart)
        return psd
    
    def spatioTemporalPSD(self):
        """%% SPATIOTEMPORALPSD Power spectrum density including reconstruction, field variations and temporal effects
        """
        tstart  = time.time()   
        psd = np.zeros((self.resAO,self.resAO,self.nSrc),dtype=complex)        
        i   = complex(0,1)
        nH  = self.nbLayers
        Hs  = self.heights
        Ws  = self.atm.weights
        nK  = self.resAO
        deltaT  = self.latency+self.samplingTime
        
        Watm = self.mskIn_ * self.Wphi * self.pistonFilterIn_      
        F = self.Rx*self.SxAv + self.Ry*self.SyAv
        
        for s in range(self.nSrc):
            if self.nGs < 2:  
                th  = self.src[s].direction - self.gs[0].direction
                if any(th):
                    A = np.zeros((self.resAO,self.resAO))
                    for l in range(self.atm.nL):                
                        A   = A + Ws[l]*np.exp(2*i*np.pi*Hs[l]*(self.kx*th[0] + self.ky*th[1]))            
                else:
                    A = np.ones((self.resAO,self.resAO))
          
                if (self.loopGain == 0):  
                    psd[:,:,s] = abs(1-F)**2*Watm
                else:
                    psd[:,:,s] = (1 + abs(F)**2*self.h2- 2*np.real(F*self.h1*A))*Watm                   
            else:    
                # tomographic case
                Beta = [self.src[s].direction[0],self.src[s].direction[1]]
                PbetaL = np.zeros([nK,nK,1,nH],dtype=complex)
                fx = Beta[0]*self.kx
                fy = Beta[1]*self.ky
                for j in range(nH):
                    PbetaL[self.mskIn_,0,j] = np.exp(i*2*np.pi*( Hs[j]*\
                          (fx[self.mskIn_]+fy[self.mskIn_]) -  deltaT*self.wSpeed[j]\
                          *(self.wDir_x[j]*self.kx[self.mskIn_] + self.wDir_y[j]*self.ky[self.mskIn_]) ))
  
                proj    = PbetaL - np.matmul(self.PbetaDM[s],self.Walpha)            
                proj_t  = np.conj(proj.transpose(0,1,3,2))
                tmp     = np.matmul(proj,np.matmul(self.Cphi,proj_t))
                psd[:,:,s] = tmp[:,:,0,0]*self.pistonFilterIn_
        self.t_spatioTemporalPSD = 1000*(time.time() - tstart)
        return psd
    
    def anisoplanatismPSD(self):
        """%% ANISOPLANATISMPSD Anisoplanatism power spectrum density
        """
        tstart  = time.time()
        psd = np.zeros((self.resAO,self.resAO,self.nSrc))
        Hs = self.atm.heights
        Ws = self.atm.weights
        Watm = self.mskIn_ * self.Wphi * self.pistonFilterIn_       
        
        for s in range(self.nSrc):
            th  = self.src[s].direction - self.gs[0].direction
            if any(th):
                A = np.zeros((self.resAO,self.resAO))
                for l in range(self.atm.nL):
                    A   = A + 2*Ws[l]*(1 - np.cos(2*np.pi*Hs[l]*(self.kx*th[0] + self.ky*th[1])))             
                psd[:,:,s] = A*Watm
        self.t_anisoplanatismPSD = 1000*(time.time() - tstart)
        return np.real(psd)
    
    def tomographyPSD(self):
        """%% TOMOGRAPHYPSD Tomographic error power spectrum density
        """
        tstart  = time.time()
        nK      = self.resAO
        psd     = np.zeros((nK,nK))
        deltaT  = self.latency + self.samplingTime
        nH      = self.nbLayers
        Hs      = self.heights
        i       = complex(0,1)
        d       = self.pitchs_dm[0]
        wDir_x  = np.cos(self.wDir*np.pi/180)
        wDir_y  = np.sin(self.wDir*np.pi/180)
        s       = 0
        Beta = [self.src[s].direction[0],self.src[s].direction[1]]
            
        MPalphaL = np.zeros([nK,nK,self.nGs,nH],dtype=complex)
        for h in range(nH):
            www = np.sinc(self.samplingTime*self.wSpeed[h]*(wDir_x[h]*self.kx + wDir_y[h]*self.ky))
            for g in range(self.nGs):
                Alpha = [self.gs[g].direction[0],self.gs[g].direction[1]]
                fx = Alpha[0]*self.kx
                fy = Alpha[1]*self.ky
                MPalphaL[self.mskIn_,g,h] = www[self.mskIn_]*2*i*np.pi*self.kxy[self.mskIn_]*np.sinc(d*self.kx[self.mskIn_])\
                *np.sinc(d*self.ky[self.mskIn_])*np.exp(i*2*np.pi*Hs[h]*(fx[self.mskIn_]+fy[self.mskIn_]))
            
        PbetaL = np.zeros([nK,nK,1,nH],dtype=complex)
        fx = Beta[0]*self.kx
        fy = Beta[1]*self.ky
        for j in range(nH):
            PbetaL[self.mskIn_,0,j] = np.exp(i*2*np.pi*( Hs[j]*\
                  (fx[self.mskIn_]+fy[self.mskIn_]) -  \
                  deltaT*self.wSpeed[j]*(wDir_x[j]*self.kx[self.mskIn_] + wDir_y[j]*self.ky[self.mskIn_]) ))
            
        W       = self.W
        Cphi    = self.Cphi # PSD obtained from the true atmosphere
            
        # this calculation is not ok !!
        proj    = PbetaL - np.matmul(W,MPalphaL)           
        proj_t  = np.conj(proj.transpose(0,1,3,2))
        psd     = np.matmul(proj,np.matmul(Cphi,proj_t))
        psd     = psd[:,:,0,0]
        self.t_tomographyPSD = 1000*(time.time() - tstart)
        return psd*self.pistonFilterIn_
    
    def powerSpectrumDensity(self):
        """ POWER SPECTRUM DENSITY AO system power spectrum density
        """
        tstart  = time.time()
        # Manage the wavelength/sampling
        wvl             = self.wvlRef
        self.atm.wvl    = wvl
        self.atm_mod.wvl= wvl
        dk              = 2*self.kc/self.resAO
        psd = np.zeros((self.fovInPixel,self.fovInPixel,self.nSrc))
        # AO correction area
        id1 = np.floor(self.fovInPixel/2 - self.resAO/2).astype(int)
        id2 = np.floor(self.fovInPixel/2 + self.resAO/2).astype(int)
        
        # Noise
        self.psdNoise           = np.real(self.noisePSD())       
        if self.nGs == 1:
            psd[id1:id2,id1:id2,:]  = np.repeat(self.psdNoise[:, :, np.newaxis], self.nSrc, axis=2)
        else:
            psd[id1:id2,id1:id2,:]  = self.psdNoise
            
        # Aliasing
        self.psdAlias           = np.real(self.aliasingPSD())
        psd[id1:id2,id1:id2,:]  = psd[id1:id2,id1:id2,:] + np.repeat(self.psdAlias[:, :, np.newaxis], self.nSrc, axis=2)
        
        
        # Add the noise and spatioTemporal PSD
        self.psdSpatioTemporal  = np.real(self.spatioTemporalPSD())
        psd[id1:id2,id1:id2,:]  = psd[id1:id2,id1:id2,:] + self.psdSpatioTemporal
       
        # Fitting
        self.psdFit = np.real(self.fittingPSD())
        psd = psd + np.repeat(self.psdFit[:, :, np.newaxis], self.nSrc, axis=2)
        
        # Return the 3D PSD array in nm^2.m^2
        self.t_powerSpectrumDensity = 1000*(time.time() - tstart)
        return psd * (dk * wvl*1e9/2/np.pi)**2
    
    def errorBreakDown(self):
        """
        """        
        tstart  = time.time()
        # Derives wavefront error
        rad2nm      = (2*self.kc/self.resAO) * self.wvlRef*1e9/2/np.pi
        self.wfeFit = np.sqrt(self.psdFit.sum()) * rad2nm
        self.wfeAl  = np.sqrt(self.psdAlias.sum()) * rad2nm
        self.wfeN   = np.sqrt(self.psdNoise.sum(axis=(0,1)))* rad2nm
        self.wfeST  = np.sqrt(self.psdSpatioTemporal.sum(axis=(0,1)))* rad2nm
        self.wfeTot = np.sqrt(self.wfeFit**2 + self.wfeAl**2 + self.wfeST**2 + self.wfeN**2)
        self.SRmar  = 100*np.exp(-(self.wfeTot*2*np.pi*1e-9/self.wvlSrc)**2)
        
        # Print
        if self.verbose == True:
            print('\n_____ ERROR BREAKDOWN  ON-AXIS_____')
            print('------------------------------------------')
            idCenter = self.zenithSrc.argmin()
            print('.Image Strehl at %4.2fmicron:\t%4.2f%s'%(self.wvlRef*1e6,self.SR[idCenter,0],'%'))
            print('.Maréchal Strehl at %4.2fmicron:\t%4.2f%s'%(self.atm.wvl*1e6,self.SRmar[idCenter],'%'))
            print('.Residual wavefront error:\t%4.2fnm'%self.wfeTot[idCenter])
            print('.Fitting error:\t\t\t%4.2fnm'%self.wfeFit)
            print('.Aliasing error:\t\t%4.2fnm'%self.wfeAl)
            if self.nGs == 1:
                print('.Noise error:\t\t\t%4.2fnm'%self.wfeN)
            else:
                print('.Noise error:\t\t\t%4.2fnm'%self.wfeN[idCenter])
            print('.Spatio-temporal error:\t\t%4.2fnm'%self.wfeST[idCenter])
            print('-------------------------------------------')
            self.psdS = self.servoLagPSD()
            self.wfeS = np.sqrt(self.psdS.sum()) * rad2nm
            print('.Sole servoLag error:\t\t%4.2fnm'%self.wfeS)
            print('-------------------------------------------')            
            if self.nGs == 1:
                self.psdAni = self.anisoplanatismPSD()
                self.wfeAni = np.sqrt(self.psdAni.sum(axis=(0,1))) * rad2nm
                print('.Sole anisoplanatism error:\t%4.2fnm'%self.wfeAni[idCenter])
            else:
                self.wfeTomo = np.sqrt(self.wfeST**2 - self.wfeS**2)
                print('.Sole tomographic error:\t%4.2fnm'%self.wfeTomo[idCenter])
            print('-------------------------------------------')
        self.t_errorBreakDown = 1000*(time.time() - tstart)
        
    def getPSF(self,verbose=False,fftphasor=False,getErrorBreakDown=False,getFWHM=False,getEncircledEnergy=False,getEnsquaredEnergy=False,displayContour=False):
        """
        """
        
        if not self.status:
            print("The fourier Model class must be instantiated first\n")
            return 0,0
        
        # GET THE AO RESIDUAL PSD/SF
        self.PSD   = self.powerSpectrumDensity() 
        
        tstart  = time.time()
        self.PSF = np.zeros((self.fovInPixel,self.fovInPixel,self.nSrc,self.nWvl))
        self.SR  = np.zeros((self.nSrc,self.nWvl))
        
        # DEFINE THE FFT PHASOR AND MULTIPLY TO THE TELESCOPE OTF
        if fftphasor:
             # FOURIER PHASOR
             uu =  fft.fftshift(fft.fftfreq(self.fovInPixel))  
             ux,uy = np.meshgrid(uu,uu)
             self.fftPhasor = np.exp(-complex(0,1)*np.pi*(ux + uy))
        else:
             self.fftPhasor = 1
        kernel = self.otfTel * self.fftPhasor      
        kernel = np.repeat(kernel[:,:,np.newaxis],self.nSrc,axis=2)     
        S     = self.otfTel.sum()
        
        
                
        # GET THE AO RESIDUAL PHASE STRUCTURE FUNCTION    
        #t1 = time.time()
        cov = fft.fftshift(fft.fftn(fft.fftshift(self.PSD,axes=(0,1)),axes=(0,1)),axes=(0,1))
        sf  = (2*cov.max(axis=(0,1)) - cov - np.conj(cov))

        
        # LOOP ON WAVELENGTHS   
        for j in range(self.nWvl):
            # GET THE AO RESIDUAL OTF
            otfTot      = fft.fftshift(np.exp(-0.5*sf*(2*np.pi*1e-9/self.wvlSrc[j])**2) * kernel,axes=(0,1))
            self.SR[:,j]= 1e2*np.abs(otfTot).sum(axis=(0,1))/S
            
            # GET THE FINAL PSF
            psf = np.real(fft.fftshift(fft.ifftn(otfTot,axes=(0,1)),axes = (0,1)))
            if self.samp <1:
                self.PSF[:,:,:,j] = FourierUtils.interpolateSupport(psf,round(self.resolution*2*self.samp).astype('int'))
            else:
                self.PSF[:,:,:,j] = psf           

        # GET THE WAVE FRONT ERROR BREAKDOWN
        if getErrorBreakDown == True:
            self.errorBreakDown()                
        
        # GET METRICS
        if getFWHM == True or getEnsquaredEnergy==True or getEncircledEnergy==True:
            self.getPsfMetrics(getEnsquaredEnergy=getEnsquaredEnergy,getEncircledEnergy=getEncircledEnergy,getFWHM=getFWHM)
        
        self.t_getPSF = 1000*(time.time() - tstart)
        
    def getPsfMetrics(self,getEnsquaredEnergy=False,getEncircledEnergy=False,getFWHM=False):
        tstart  = time.time()
        self.FWHM = np.zeros((2,self.nSrc,self.nWvl))
        if getEnsquaredEnergy==True:
            self.EnsqE   = np.zeros((int(self.fovInPixel/2)+1,self.nSrc,self.nWvl))
        if getEncircledEnergy==True:
            rr,radialprofile = FourierUtils.radial_profile(self.PSF[:,:,0,0])
            self.EncE   = np.zeros((len(radialprofile),self.nSrc,self.nWvl))
        for n in range(self.nSrc):
            for j in range(self.nWvl):
                if getFWHM == True:
                    self.FWHM[:,n,j]  = FourierUtils.getFWHM(self.PSF[:,:,n,j],self.psInMas,rebin=1,method='contour',nargout=2)
                if getEnsquaredEnergy == True:
                    self.EnsqE[:,n,j] = 1e2*FourierUtils.getEnsquaredEnergy(self.PSF[:,:,n,j])
                if getEncircledEnergy == True:
                    self.EncE[:,n,j]  = 1e2*FourierUtils.getEncircledEnergy(self.PSF[:,:,n,j])
                        
        self.t_getPsfMetrics = 1000*(time.time() - tstart)
                
    def displayResults(self,eeRadiusInMas=75,displayContour=False):
        """
        """
        tstart  = time.time()
        deg2rad = np.pi/180
        # GEOMETRY
        plt.figure()
        plt.polar(self.azimuthSrc*deg2rad,self.zenithSrc,'ro',markersize=7,label='PSF evaluation (arcsec)')
        plt.polar(self.azimuthGs*deg2rad,self.zenithGs,'bs',markersize=7,label='GS position')
        plt.polar(self.azimuthOpt*deg2rad,self.zenithOpt,'kx',markersize=10,label='Optimization directions')
        plt.legend(bbox_to_anchor=(1.05, 1))
           
        # PSFs
        if np.any(self.PSF):   
            nmin = self.zenithSrc.argmin()
            nmax = self.zenithSrc.argmax()
            plt.figure()
            if self.PSF.shape[2] >1 and self.PSF.shape[3] == 1:             
                plt.title("PSFs at {:.1f} and {:.1f} arcsec from center".format(self.zenithSrc[nmin],self.zenithSrc[nmax]))
                P = np.concatenate((self.PSF[:,:,nmin,0],self.PSF[:,:,nmax,0]),axis=1)
            elif self.PSF.shape[2] >1 and self.PSF.shape[3] >1:
                plt.title("PSFs at {:.0f} and {:.0f} arcsec from center\n - Top: {:.0f}nm - Bottom:{:.0f} nm".format(self.zenithSrc[0],self.zenithSrc[-1],1e9*self.wvlSrc[0],1e9*self.wvlSrc[-1]))
                P1 = np.concatenate((self.PSF[:,:,nmin,0],self.PSF[:,:,nmax,0]),axis=1)
                P2 = np.concatenate((self.PSF[:,:,nmin,-1],self.PSF[:,:,nmax,-1]),axis=1)
                P  = np.concatenate((P1,P2),axis=0)
            else:
                plt.title('PSF')
                P = self.PSF[:,:,nmin,0]
            plt.imshow(np.log10(np.abs(P)))
        
           
        if displayContour == True and np.any(self.SR) and self.SR.size > 1:
            self.displayPsfMetricsContours(eeRadiusInMas=eeRadiusInMas)
        else:
            # STREHL-RATIO
            if np.any(self.SR) and self.SR.size > 1:
                plt.figure()
                plt.plot(self.zenithSrc,self.SR[:,0],'bo',markersize=10)
                plt.xlabel("Off-axis distance")
                plt.ylabel("Strehl-ratio at {:.1f} nm (percents)".format(self.wvlSrc[0]*1e9))
                plt.show()
  
            # FWHM
            if np.any(self.FWHM) and self.FWHM.size > 1:
                plt.figure()
                plt.plot(self.zenithSrc,np.hypot(self.FWHM[0,:,0],self.FWHM[1,:,0]),'bo',markersize=10)
                plt.xlabel("Off-axis distance")
                plt.ylabel("Geometrical-mean FWHM at {:.1f} nm (mas)".format(self.wvlSrc[0]*1e9))
                plt.show()
         
            # Ensquared energy
            if np.any(self.EnsqE):
                nntrue      = eeRadiusInMas/self.psInMas
                nn2         = int(nntrue)
                EEmin       = self.EnsqE[nn2,:,0]
                EEmax       = self.EnsqE[nn2+1,:,0]
                EEtrue      = (nntrue - nn2)*EEmax + (nn2+1-nntrue)*EEmin
                plt.figure()
                plt.plot(self.zenithSrc,EEtrue,'bo',markersize=10)
                plt.xlabel("Off-axis distance")
                plt.ylabel("{:f}-mas-side Ensquared energy at {:.1f} nm (percents)".format(eeRadiusInMas,self.wvlSrc[0]*1e9))
                plt.show()

            if np.any(self.EncE):
                nntrue      = eeRadiusInMas/self.psInMas
                nn2         = int(nntrue)
                EEmin       = self.EncE[nn2,:,0]
                EEmax       = self.EncE[nn2+1,:,0]
                EEtrue      = (nntrue - nn2)*EEmax + (nn2+1-nntrue)*EEmin
                plt.figure()
                plt.plot(self.zenithSrc,EEtrue,'bo',markersize=10)
                plt.xlabel("Off-axis distance")
                plt.ylabel("{:f}-mas-diameter Encircled energy at {:.1f} nm (percents)".format(eeRadiusInMas*2,self.wvlSrc[0]*1e9))
                plt.show()
        
        self.t_displayResults = 1000*(time.time() - tstart)
            
    def displayPsfMetricsContours(self,eeRadiusInMas=75):

        tstart  = time.time()
        # Polar to cartesian
        x = self.zenithSrc * np.cos(np.pi/180*self.azimuthSrc)
        y = self.zenithSrc * np.sin(np.pi/180*self.azimuthSrc)
    

        nn          = int(np.sqrt(self.SR.shape[0]))
        if nn**2 == self.SR.shape[0]:
            nIntervals  = nn
            X           = np.reshape(x,(nn,nn))    
            Y           = np.reshape(y,(nn,nn))
        
            # Strehl-ratio
            SR = np.reshape(self.SR[:,0],(nn,nn))
            plt.figure()
            contours = plt.contour(X, Y, SR, nIntervals, colors='black')
            plt.clabel(contours, inline=True,fmt='%1.1f')
            plt.contourf(X,Y,SR)
            plt.title("Strehl-ratio at {:.1f} nm (percents)".format(self.wvlSrc[0]*1e9))
            plt.colorbar()
        
            # FWHM
            if np.any(self.FWHM) and self.FWHM.size > 1:
                FWHM = np.reshape(np.hypot(self.FWHM[0,:,0],self.FWHM[1,:,0]),(nn,nn))
                plt.figure()
                contours = plt.contour(X, Y, FWHM, nIntervals, colors='black')
                plt.clabel(contours, inline=True,fmt='%1.1f')
                plt.contourf(X,Y,FWHM)
                plt.title("Geometrical-mean FWHM at {:.1f} nm (mas)".format(self.wvlSrc[0]*1e9))
                plt.colorbar()
        
            # Ensquared Enery
            if np.any(self.EnsqE) and self.EnsqE.shape[1] > 1:
                nntrue      = eeRadiusInMas/self.psInMas
                nn2         = int(nntrue)
                EEmin       = self.EnsqE[nn2,:,0]
                EEmax       = self.EnsqE[nn2+1,:,0]
                EEtrue      = (nntrue - nn2)*EEmax + (nn2+1-nntrue)*EEmin
                EE          = np.reshape(EEtrue,(nn,nn))
                plt.figure()
                contours = plt.contour(X, Y, EE, nIntervals, colors='black')
                plt.clabel(contours, inline=True,fmt='%1.1f')
                plt.contourf(X,Y,EE)
                plt.title("{:.1f}-mas-side Ensquared energy at {:.1f} nm (percents)".format(eeRadiusInMas*2,self.wvlSrc[0]*1e9))
                plt.colorbar()
            
            # Encircled Enery
            if np.any(self.EncE) and self.EncE.shape[1] > 1:
                nntrue      = eeRadiusInMas/self.psInMas
                nn2         = int(nntrue)
                EEmin       = self.EncE[nn2,:,0]
                EEmax       = self.EncE[nn2+1,:,0]
                EEtrue      = (nntrue - nn2)*EEmax + (nn2+1-nntrue)*EEmin
                EE          = np.reshape(EEtrue,(nn,nn))
                plt.figure()
                contours = plt.contour(X, Y, EE, nIntervals, colors='black')
                plt.clabel(contours, inline=True,fmt='%1.1f')
                plt.contourf(X,Y,EE)
                plt.title("{:.1f}-mas-diameter Encircled energy at {:.1f} nm (percents)".format(eeRadiusInMas*2,self.wvlSrc[0]*1e9))
                plt.colorbar()
        else:
            print('You must define a square grid for PSF evaluations directions - no contours plots avalaible')
            
        self.t_displayPsfMetricsContours = 1000*(time.time() - tstart)
    
    def displayExecutionTime(self):
        
        # total
        print("Required time for total calculation (ms)\t : {:f}".format(self.t_init))
        print("Required time for getting parameters (ms)\t : {:f}".format(self.t_getParam))
        # Reconstructors
        if self.nGs == 1:
            print("Required time for reconstructors init (ms)\t : {:f}".format(self.t_reconstructor))
        else:
            print("Required time for optimization init (ms)\t : {:f}".format(self.t_finalReconstructor))
            print("Required time for tomography init (ms)\t : {:f}".format(self.t_tomo))
            print("Required time for optimization init (ms)\t : {:f}".format(self.t_opt))
        # Controller
        print("Required time for controller instantiation (ms)\t : {:f}".format(self.t_controller))
        # PSD
        print("Required time for final PSD calculation (ms)\t : {:f}".format(self.t_powerSpectrumDensity))
        print("Required time for fitting PSD calculation (ms)\t : {:f}".format(self.t_fittingPSD))
        print("Required time for aliasing PSD calculation (ms)\t : {:f}".format(self.t_aliasingPSD))
        print("Required time for noise PSD calculation (ms)\t : {:f}".format(self.t_noisePSD))
        print("Required time for ST PSD calculation (ms)\t : {:f}".format(self.t_spatioTemporalPSD))
        if self.calcPSF:
            print("Required time for all PSFs calculation (ms)\t : {:f}".format(self.t_getPSF))
        # Error breakdown
        if self.getErrorBreakDown:
            print("Required time for error calculation (ms)\t : {:f}".format(self.t_errorBreakDown))
            
        # PSF metrics
        if self.getPSFmetrics:
            print("Required time for get PSF metrics (ms)\t\t : {:f}".format(self.t_getPsfMetrics))
        
        # Display
        if self.display:
            print("Required time for displaying figures (ms)\t : {:f}".format(self.t_displayResults))