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
import numpy.fft as fft
import matplotlib.pyplot as plt
import math
import scipy.special as spc
import time
import os.path as ospath
from configparser import ConfigParser

import FourierUtils
from telescope import telescope
from atmosphere import atmosphere
from source import source

    
class fourierModel:
    """ Fourier class gathering the PSD calculation for PSF reconstruction. 
    """
    
    # DEPENDANT VARIABLES DEFINITION
    @property
    def kc(self):
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
        return math.ceil(self.fovInPixel/self.resAO)

    # CONTRUCTOR
    def __init__(self,file,calcPSF=True,verbose=False,display=True,aoFilter='circle',getErrorBreakDown=False):
    
        # PARSING INPUTS
        self.verbose = verbose
        self.status = 0
        self.file   = file  
        self.status = self.parameters(self.file)        
        
        if self.status:
            start = time.time()
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
                self.mskIn_  = self.kxy  <= self.kc[0]      
                self.mskOut_ = np.hypot(self.kxExt,self.kyExt) > self.kc[0]
            else:
                self.mskIn_  = (abs(self.kx) <= self.kc[0]) | (abs(self.ky) <= self.kc[0])    
                self.mskOut_ = (abs(self.kxExt)>self.kc[0]) | (abs(self.kyExt)>self.kc[0])         
            
            # DEFINE NOISE AND ATMOSPHERE PSD
            self.Wn   = np.mean(self.noiseVariance)/(2*self.kc[0])**2
            self.Wphi = self.atm.spectrum(self.kxy);
            
            # DEFINE THE OTF TELESCOPE
            P                   = np.zeros((self.fovInPixel,self.fovInPixel))
            id1                 = np.floor(self.fovInPixel/2 - self.tel.resolution/2).astype(int)
            id2                 = np.floor(self.fovInPixel/2 + self.tel.resolution/2).astype(int)
            P[id1:id2,id1:id2]  = self.tel.pupil
            self.otfTel         = np.real(fft.fftshift(FourierUtils.fftCorrel(P,P)))
            self.otfTel         = self.otfTel/self.otfTel.max()
        
        
            if self.verbose:
                self.elapsed_time_init = (time.time() - start) 
                print("Required time for initialization (s)\t : {:f}".format(self.elapsed_time_init))
          
            if calcPSF:
                PSF,PSD,SR,FWHM = self.getPSF(verbose=verbose,getErrorBreakDown=getErrorBreakDown)
                if display:
                    self.displayResults()
                

          
    def __repr__(self):
        s = "Fourier Model class "
        if self.status == 1:
            s = s + "instantiated"
        else:
            s = s + "not instantiated"
        
        self.displayResults()
        
        return s
    
    def parameters(self,file):
                    
        start = time.time() 
        
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
        self.wvlSrc         = np.array(eval(config['PSF_DIRECTIONS']['ScienceWavelength']))*np.ones(self.nSrc)
        self.zenithSrc      = np.array(np.array(eval(config['PSF_DIRECTIONS']['ScienceZenith'])))
        self.azimuthSrc     = np.array(np.array(eval(config['PSF_DIRECTIONS']['ScienceAzimuth'])))
        # ----- verification
        self.src = []
        if len(self.zenithSrc) == len(self.azimuthSrc):
            self.nSrc = len(self.zenithSrc)
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of scientific sources is not consistent in the parameters file\n')
            return 0
        
        #%% Guide stars
        self.nGs            = len(eval(config['GUIDESTARTS_HO']['GuideStarZenith_HO']))
        self.zenithGs       = np.array(eval(config['GUIDESTARTS_HO']['GuideStarZenith_HO']))
        self.azimuthGs      = np.array(eval(config['GUIDESTARTS_HO']['GuideStarAzimuth_HO']))
        self.heightGs       = eval(config['GUIDESTARTS_HO']['GuideStarHeight_HO'])
        # ----- verification
        if len(self.zenithGs) == len(self.azimuthGs):
            self.nGs = len(self.zenithGs)
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of guide stars for high-order sensing is not consistent in the parameters file\n')
            return 0
        
        #%% WFS parameters
        self.wvlGs          = eval(config['SENSOR_HO']['SensingWavelength_HO'])*np.ones(self.nGs)
        self.nLenslet_HO    = eval(config['SENSOR_HO']['nLenslet_HO'])
        self.pitchs_wfs     = self.D/self.nLenslet_HO * np.ones(self.nGs)
        self.nph_HO         = eval(config['SENSOR_HO']['nph_HO'])
        self.pixel_Scale_HO = eval(config['SENSOR_HO']['pixel_scale_HO'])
        self.sigmaRON_HO    = eval(config['SENSOR_HO']['sigmaRON_HO'])
        self.Npix_per_subap_HO = eval(config['SENSOR_HO']['Npix_per_subap_HO'])
        self.ND             = self.wvlGs[0]/self.pitchs_wfs[0]*rad2mas/self.pixel_Scale_HO #spot FWHM in pixels and without turbulence
        varRON              = np.pi**2/3*(self.sigmaRON_HO /self.nph_HO)**2*(self.Npix_per_subap_HO**2/self.ND)**2
        self.NT             = self.wvlGs[0]/self.r0*(self.wvlGs[0]/self.wvlAtm)**1.2 * rad2mas/self.pixel_Scale_HO
        varShot             = np.pi**2/(2*self.nph_HO)*(self.NT/self.ND)**2
        self.noiseVariance  = (varRON + varShot) *np.ones(self.nGs)
        self.loopGain       = eval(config['SENSOR_HO']['loopGain_HO'])
        self.samplingTime   = 1/eval(config['SENSOR_HO']['SensorFrameRate_HO'])
        self.latency        = eval(config['SENSOR_HO']['loopDelaySteps_HO'])*self.samplingTime
        
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
        lonD  = rad2mas*self.wvlSrc[0]/ self.D
        self.psInMas = eval(config['PSF_DIRECTIONS']['psInMas'])
        if self.psInMas == 0:
            self.psInMas    = lonD/2
            #fovInPixel = 2*self.resolution
       #else:
            #fovInPixel = round(psf_FoV*1e3/self.psInMas)

        self.samp       = lonD/self.psInMas/2
        if self.samp >=1:
            self.fovInPixel = round(self.resolution*self.samp*2).astype('int')
        else:
            self.fovInPixel = round(2*self.resolution/self.samp).astype('int')
        
        if self.verbose:
            print('.Field of view:\t\t%4.2f arcsec\n.Pixel scale:\t\t%4.2f mas\n.Over-sampling:\t\t%4.2f'%(self.fovInPixel*self.psInMas/1e3,self.psInMas,self.samp))
            print('\n-------------------------------------------\n')
        
        self.PSDstep  = self.psInMas/1e3/self.wvlSrc[0]/rad2arcsec
        self.resAO    = int(1/np.min(self.pitchs_dm)/self.PSDstep)

        #%% instantiating sub-classes
        
        # Telescope
        self.tel = telescope(self.D,self.zenith_angle,self.obsRatio,self.resolution,self.path_pupil)
        
        # Strechning factor (LGS case)        
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
        self.atm = atmosphere(self.wvlAtm,self.r0*self.tel.airmass**(-3/5),self.weights,self.heights,self.wSpeed,self.wDir,self.L0)
        self.atm_mod = atmosphere(self.wvlAtm,self.r0_mod*self.tel.airmass**(-3/5),self.weights_mod,self.heights_mod,self.wSpeed_mod,self.wDir_mod,self.L0_mod)
        
        # Scientific Sources
        self.src = [source(0,0,0) for k in range(self.nSrc)]  
        for n in range(self.nSrc):
            self.src[n] = source(self.wvlSrc[n],self.zenithSrc[n],self.azimuthSrc[n],0,n+1,"SCIENTIFIC STAR",verbose=True)
                   
        # Guide stars
        #self.gs = []
        self.gs = [source(0,0,0) for k in range(self.nGs)]  
        for n in range(self.nGs):
            self.gs[n] = source(self.wvlGs[n],self.zenithGs[n],self.azimuthGs[n],self.heightGs,n+1,"GUIDE STAR",verbose=True)
        if len(self.pitchs_wfs) == 1:
            self.pitchs_wfs = self.pitchs_wfs * np.ones(self.nGs)
        if len(self.noiseVariance) == 1:
            self.noiseVariance = self.noiseVariance * np.ones(self.nGs)    
        
        self.tinit = (time.time() - start) 
        print("Required time for grabbing param. (s)\t : {:f}".format(self.tinit))
        
        return 1
    
#%% RECONSTRUCTOR DEFINITION    
    def reconstructionFilter(self):
        """
        """          
       
        # reconstructor derivation
        i    = complex(0,1)
        d    = self.pitchs_dm[0]   
        Sx   = 2*i*np.pi*self.kx*d
        Sy   = 2*i*np.pi*self.ky*d                        
        Av   = np.sinc(d*self.kx)*np.sinc(d*self.ky)*np.exp(i*np.pi*d*(self.kx+self.ky))        
        self.SxAv = Sx*Av
        self.SyAv = Sy*Av
        gPSD = abs(self.SxAv)**2 + abs(self.SyAv)**2 + self.Wn/self.Wphi
        self.Rx = np.conjugate(self.SxAv)/gPSD
        self.Ry = np.conjugate(self.SyAv)/gPSD
                
        # Manage NAN value if any   
        self.Rx[np.isnan(self.Rx)] = 0
        self.Ry[np.isnan(self.Ry)] = 0
            
        # Set central point (i.e. kx=0,ky=0) to zero
        N = int(np.ceil((self.kx.shape[0]-1)/2))
        self.Rx[N,N] = 0
        self.Ry[N,N] = 0

    def tomographicReconstructor(self):
        
        tstart = time.time()
        nK      = self.resAO
        nL      = len(self.heights)
        nL_mod  = len(self.heights_mod)
        nGs     = self.nGs
        Alpha = np.zeros([2,nGs])
        for j in range(nGs):
            Alpha[0,j] = self.gs[j].direction[0]
            Alpha[1,j] = self.gs[j].direction[1]
            
        # WFS operator matrix
        i    = complex(0,1)
        d    = self.pitchs_wfs   #sub-aperture size      
        M    = np.zeros([nK,nK,nGs,nGs],dtype=complex)
        for j in np.arange(0,nGs):
            M[:,:,j,j] = 2*i*np.pi*self.kxy*np.sinc(d[j]*self.kx)*np.sinc(d[j]*self.ky)
        self.M = M
        # Projection matrix
        P    = np.zeros([nK,nK,nGs,nL_mod],dtype=complex)
        
        for n in range(nL_mod):
            for j in range(nGs):
                fx = self.kx*Alpha[0,j]
                fy = self.ky*Alpha[1,j]
                P[:,:,j,n] = np.exp(i*2*np.pi*self.heights_mod[n]*(fx + fy))
                
        MP = np.matmul(M,P)
        MP_t = np.conj(MP.transpose(0,1,3,2))
        
        # Noise covariance matrix
        Cb = np.zeros([nK,nK,nGs,nGs],dtype=complex)
        for j in range(nGs):
            Cb[:,:,j,j]  = self.noiseVariance[j]
        self.Cb = Cb
        
        # Atmospheric PSD with the true atmosphere
        Cphi = np.zeros([nK,nK,nL,nL],dtype=complex)
        cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2./(2.*np.pi**(11/3)))
        if nL == 1:
            Cphi[:,:,0,0] = self.atm.r0**(-5/3)*cte*(self.kxy**2 + 1/self.atm.L0**2)**(-11/6)*self.pistonFilterIn_
        else:
            for j in range(nL):
                Cphi[:,:,j,j] = self.atm.layer[j].weight * self.atm.r0**(-5/3)*cte*(self.kxy**2 + 1/self.atm.L0**2)**(-11/6)*self.pistonFilterIn_
        self.Cphi = Cphi
        
        # Atmospheric PSD with the modelled atmosphere
        Cphi = np.zeros([nK,nK,nL_mod,nL_mod],dtype=complex)
        if nL_mod == 1:
            Cphi[:,:,0,0] = self.atm_mod.r0**(-5/3)*cte*(self.kxy**2 + 1/self.atm_mod.L0**2)**(-11/6)*self.pistonFilterIn_
        else:
            for j in range(nL_mod):
                Cphi[:,:,j,j] = self.atm_mod.layer[j].weight * self.atm_mod.r0**(-5/3)*cte*(self.kxy**2 + 1/self.atm_mod.L0**2)**(-11/6)*self.pistonFilterIn_
        self.Cphi_mod = Cphi
        
        to_inv  = np.matmul(np.matmul(MP,self.Cphi_mod),MP_t) + self.Cb 
        inv     = np.zeros(to_inv.shape,dtype=complex)
        
        u,s,vh      = np.linalg.svd(to_inv)
        for x in range(to_inv.shape[0]):
            for y in range(to_inv.shape[1]):
                slim         = np.amax(s[x,y])/self.condmax_tomo
                rank         = np.sum(s[x,y] > slim)
                uu           = u[x, y, :, :rank]
                uu          /= s[x,y, :rank]
                inv[x,y,:,:] = np.transpose(np.conj(np.dot(uu, np.asarray(vh[x,y,:rank]) ) ) )
        Wtomo = np.matmul(np.matmul(self.Cphi_mod,MP_t),inv)
        
        self.ttomo = time.time() - tstart
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
                index   = self.kxy <= self.kc[j]
                Pdm[index,0,j] = np.exp(i*2*np.pi*self.h_dm[j]*(fx[index]+fy[index]))
            Pdm_t = np.conj(Pdm.transpose(0,1,3,2))
            for l in range(nL):                 #loop on atmosphere layers
                Pl[:,:,0,l] = np.exp(i*2*np.pi*self.heights_mod[l]*(fx + fy))
                
            mat1   += np.matmul(Pdm_t,Pl)*self.weightOpt[d_o]
            to_inv += np.matmul(Pdm_t,Pdm)*self.weightOpt[d_o]
            
        mat2 = np.zeros(to_inv.shape,dtype=complex)
        
        u,s,vh      = np.linalg.svd(to_inv)
        for x in range(to_inv.shape[0]):
            for y in range(to_inv.shape[1]):
                slim        = np.amax(s[x,y])/self.condmax_tomo
                rank        = np.sum(s[x,y] > slim)
                uu          = u[x, y, :, :rank]
                uu          /= s[x,y, :rank]
                mat2[x,y,:,:] = np.transpose(np.conj(np.dot(uu, np.asarray(vh[x,y,:rank]) ) ) )
        
        Popt = np.matmul(mat2,mat1)
        
        self.topt = time.time() - tstart
        return Popt
 
    def finalReconstructor(self):
        self.Wtomo  = self.tomographicReconstructor()
        self.Popt   = self.optimalProjector()
        self.W      = np.matmul(self.Popt,self.Wtomo)
        
        # Computation of the Pbeta^DM matrix
        nDm     = len(self.h_dm)
        nK      = self.resAO

        self.PbetaDM = []
        for s in range(self.nSrc):
            fx = self.src[s].direction[0]*self.kx
            fy = self.src[s].direction[1]*self.ky
            PbetaDM = np.zeros([nK,nK,1,nDm],dtype=complex)
            for j in range(nDm): #loop on DMs
                index   = self.kxy <= self.kc[j]
                PbetaDM[index,0,j] = np.exp(complex(0,1)*2*np.pi*self.h_dm[j]*(fx[index] + fy[index]))
            self.PbetaDM.append(PbetaDM)
        
        
#%% CONTROLLER DEFINITION
    def  controller(self,nTh=10,nF=500):
        """
        """
        
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
        f      = np.logspace(-2,np.log10(0.5/1e-3),nF)
        z      = np.exp(-2*i*np.pi*f*Ts)
        hInt   = self.loopGain/(1-z**(-1))
        rtfInt = 1/(1+hInt*z**(-delay))
        atfInt = hInt*z**(-delay)*rtfInt
        
        if self.loopGain == 0:
            ntfInt = 1
        else:
            ntfInt = atfInt/z
                
        self.noiseGain = np.trapz( abs(ntfInt)**2,f)*2*Ts
        
        
        # Get transfer functions                                        
        for l in np.arange(0,self.atm.nL):
            for iTheta in np.arange(0,nTh):
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
                MAG = abs(ntfInt)                
                MAG[fi == 0] = 1
                PH  = np.angle(ntfInt)  
                hnbuf[:,:,iTheta] = abs(MAG*np.exp(i*PH))**2
                
            h1 = h1 + weights[l]*np.sum(h1buf,axis=2)/nTh
            h2 = h2 + weights[l]*np.sum(h2buf,axis=2)/nTh
            hn = hn + weights[l]*np.sum(hnbuf,axis=2)/nTh
        
        self.h1 = h1
        self.h2 = h2
        self.hn = hn
                
        
 #%% PSD DEFINTIONS      
    def fittingPSD(self):
        """ FITTINGPSD Fitting error power spectrum density """                 
        #Instantiate the function output
        psd                 = np.zeros((self.fovInPixel,self.fovInPixel))
        psd[self.mskOut_]   = self.atm.spectrum(self.kExtxy[self.mskOut_])
        return psd
    
    def aliasingPSD(self):
        """
        """
        psd = np.zeros((self.resAO,self.resAO))
        tmp = psd
        if self.nGs < 2:
            i  = complex(0,1)
            d  = self.pitchs_dm[0]
            T  = self.samplingTime
            td = self.latency        
            vx = self.atm.wSpeed*np.cos(self.atm.wDir*np.pi/180)
            vy = self.atm.wSpeed*np.sin(self.atm.wDir*np.pi/180)
            Rx = self.Rx
            Ry = self.Ry
        
            if self.loopGain == 0:
                tf = 1
            else:
                tf = self.h1
                    
            weights = self.atm.weights
        
            w = 2*i*np.pi*d;
            for mi in np.arange(-self.nTimes,self.nTimes+1):
                for ni in np.arange(-self.nTimes,self.nTimes+1):
                    if (mi!=0) | (ni!=0):
                        km = self.kx - mi/d
                        kn = self.ky - ni/d
                        PR = FourierUtils.pistonFilter(self.tel.D,np.hypot(km,kn),fm=mi/d,fn=ni/d)
                        W_mn = (abs(km**2 + kn**2) + 1/self.atm.L0**2)**(-11/6)     
                        Q = (Rx*w*km + Ry*w*kn) * (np.sinc(d*km)*np.sinc(d*kn))
                        avr = 0
                        
                        for l in np.arange(0,self.atm.nL):
                            avr = avr + weights[l]* (np.sinc(km*vx[l]*T)*np.sinc(kn*vy[l]*T)
                            *np.exp(2*i*np.pi*km*vx[l]*td)*np.exp(2*i*np.pi*kn*vy[l]*td)*tf)
                                                          
                        tmp = tmp + PR*W_mn * abs(Q*avr)**2
               
            tmp[np.isnan(tmp)] = 0        
            psd[self.mskIn_]         = tmp[self.mskIn_]
        
        return psd*self.atm.r0**(-5/3)*0.0229 
            
    def noisePSD(self):
        """NOISEPSD Noise error power spectrum density
        """
        psd = np.zeros((self.resAO,self.resAO),dtype=complex)
        if self.noiseVariance[0] > 0:
            if self.nGs < 2:        
                psd[self.mskIn_] = self.noiseVariance/(2*self.kc)**2*(abs(self.Rx[self.mskIn_])**2 + abs(self.Ry[self.mskIn_]**2));    
            else:  
                nK      = self.resAO
                W       = self.W
                PbetaDM = self.PbetaDMj
                Cb      = self.Cb
                PW      = np.matmul(PbetaDM,W)
                PW_t    = np.conj(PW.transpose(0,1,3,2))
                
                #### TO BE OPTIMIZED !!!! ############
                for x in range(nK):
                    for y in range(nK):
                        if self.mskIn_[x,y] == True:
                            psd[x,y] = np.dot(PW[x,y,:,:],np.dot(Cb[x,y,:,:],PW_t[x,y,:,:]))
                #### TO BE OPTIMIZED !!!! ############
        return psd*self.pistonFilterIn_*self.noiseGain
    
    def servoLagPSD(self):
        """ SERVOLAGPSD Servo-lag power spectrum density
        """
            
        psd = np.zeros((self.resAO,self.resAO))    
        if hasattr(self, 'Rx') == False:
            self.reconstructionFilter()

        F = (self.Rx[self.mskIn_]*self.SxAv[self.mskIn_] + self.Ry[self.mskIn_]*self.SyAv[self.mskIn_])        
        if (self.loopGain == 0):
            psd[self.mskIn_] = abs(1-F)**2*self.Wphi[self.mskIn_]
        else:
            psd[self.mskIn_] = (1 + abs(F)**2*self.h2[self.mskIn_] - 2*np.real(F*self.h1[self.mskIn_]))*self.Wphi[self.mskIn_]
            
        return psd*self.pistonFilterIn_
    
    def spatioTemporalPSD(self):
        """%% SPATIOTEMPORALPSD Power spectrum density including reconstruction, field variations and temporal effects
        """
           
        psd = np.zeros((self.resAO,self.resAO))        
        if self.nGs < 2:
            heights = self.atm.heights
            weights = self.atm.weights
            A       = 0*self.kx
            if sum(sum(abs(self.srcj.direction - self.gs[0].direction)))!=0:
                th  = self.srcj.direction - self.gs[0].direction
                for l in np.arange(0,self.atm.nL):                
                    red = 2*np.pi*heights[l]*(self.kx*th[0] + self.ky*th[1])
                    A   = A + weights[l]*np.exp(complex(0,1)*red)            
            else:
                A = np.ones(self.resAO)
        
            F = (self.Rx[self.mskIn_]*self.SxAv[self.mskIn_] + self.Ry[self.mskIn_]*self.SyAv[self.mskIn_])

            
            if (self.loopGain == 0):  
                psd[self.mskIn_] = abs(1-F)**2*A*self.Wphi[self.mskIn_]
            else:
                psd[self.mskIn_] = (1 + abs(F)**2*self.h2[self.mskIn_] 
                - 2*np.real(F*self.h1[self.mskIn_]*A[self.mskIn_]))*self.Wphi[self.mskIn_]
                                
        else:
            
            deltaT  = self.latency+self.samplingTime
            nK      = self.resAO
            nH      = self.nbLayers
            Hs      = self.heights
            i       = complex(0,1)
            d       = self.pitchs_dm[0]
            wDir_x  = np.cos(self.wDir*np.pi/180)
            wDir_y  = np.sin(self.wDir*np.pi/180)
            
            Beta = [self.srcj.direction[0],self.srcj.direction[1]]
            
            MPalphaL = np.zeros([nK,nK,self.nGs,nH],dtype=complex)
            for h in range(nH):
                www = np.sinc(self.samplingTime*self.wSpeed[h]*(wDir_x[h]*self.kx + wDir_y[h]*self.ky))
                for g in range(self.nGs):
                    Alpha = [self.gs[g].direction[0],self.gs[g].direction[1]]
                    fx = Alpha[0]*self.kx
                    fy = Alpha[1]*self.ky
                    MPalphaL[self.mskIn_,g,h] = www[self.mskIn_]*2*i*np.pi*self.kxy[self.mskIn_]*np.sinc(d*self.kx[self.mskIn_])*\
                    np.sinc(d*self.ky[self.mskIn_])*np.exp(i*2*np.pi*Hs[h]*(fx[self.mskIn_]+fy[self.mskIn_]))
            
            PbetaL = np.zeros([nK,nK,1,nH],dtype=complex)
            fx = Beta[0]*self.kx
            fy = Beta[1]*self.ky
            for j in range(nH):
                PbetaL[self.mskIn_,0,j] = np.exp(i*2*np.pi*( Hs[j]*\
                      (fx[self.mskIn_]+fy[self.mskIn_]) -  deltaT*self.wSpeed[j]\
                      *(wDir_x[j]*self.kx[self.mskIn_] + wDir_y[j]*self.ky[self.mskIn_]) ))
            
            PbetaDM = self.PbetaDMj
            W       = self.W
            Cphi    = self.Cphi # PSD obtained from the true atmosphere
            
            proj = PbetaL - np.matmul(PbetaDM,np.matmul(W,MPalphaL))            
            proj_t  = np.conj(proj.transpose(0,1,3,2))
            psd     = np.matmul(proj,np.matmul(Cphi,proj_t))
            psd     = psd[:,:,0,0]
        return psd*self.pistonFilterIn_
    
    def anisoplanatismPSD(self):
        """%% ANISOPLANATISMPSD Anisoplanatism power spectrum density
        """
        
        psd = np.zeros((self.resAO,self.resAO))
        heights = self.atm.heights
        weights = self.atm.weights
        A       = 0*self.kx
        if sum(sum(abs(self.srcj.direction - self.gs[0].direction)))!=0:
            th  = self.srcj.direction - self.gs[0].direction
            for l in np.arange(0,self.atm.nL):
                red = 2*np.pi*heights[l]*(self.kx*th[0] + self.ky*th[1])
                A   = A + 2*weights[l]*( 1 - np.cos(red) )     
        else:
            A = np.ones(self.resAO)
        
        psd[self.mskIn_] = A[self.mskIn_]*self.Wphi[self.mskIn_]
        
        return psd*self.pistonFilterIn_
    
    def tomographyPSD(self):
        """%% TOMOGRAPHYPSD Tomographic error power spectrum density
        """
        nK      = self.resAO
        psd     = np.zeros((nK,nK))
        deltaT  = self.latency + self.samplingTime
        nH      = self.nbLayers
        Hs      = self.heights
        i       = complex(0,1)
        d       = self.pitchs_dm[0]
        wDir_x  = np.cos(self.wDir*np.pi/180)
        wDir_y  = np.sin(self.wDir*np.pi/180)
            
        Beta = [self.srcj.direction[0],self.srcj.direction[1]]
            
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
        proj = PbetaL - np.matmul(W,MPalphaL)           
        proj_t = np.conj(proj.transpose(0,1,3,2))
        psd = np.matmul(proj,np.matmul(Cphi,proj_t))
        psd = psd[:,:,0,0]

        return psd*self.pistonFilterIn_
        
    def powerSpectrumDensity(self):
        """ POWER SPECTRUM DENSITY AO system power spectrum density
        """
        # COmputation of the extended spatial frequencies domain
        psd     = np.zeros((self.fovInPixel,self.fovInPixel),dtype=complex)
        # Summing PSDs
        id1 = np.floor(self.fovInPixel/2 - self.resAO/2).astype(int)
        id2 = np.floor(self.fovInPixel/2 + self.resAO/2).astype(int)
        psd[id1:id2,id1:id2]     = self.noisePSD() + self.aliasingPSD() + self.spatioTemporalPSD()
        
        return np.real(psd + self.fittingPSD())
    
    def errorBreakDown(self,iSrc=0):
        """
        """
        # Constants
        self.srcj   = self.src[iSrc]
        wvl         = self.srcj.wvl
        self.atm.wvl= wvl
        self.atm_mod.wvl = wvl
        rad2nm      = 1e9*wvl/2/np.pi        
        self.PbetaDMj = self.PbetaDM[iSrc]
       
        # Get PSDs
        psdFit = self.fittingPSD()
        psdAl  = self.aliasingPSD()
        psdN   = self.noisePSD()
        psdST  = self.spatioTemporalPSD()
        
        
        # Derives wavefront error
        self.wfeFit = np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdFit,self.kxExt),self.kxExt)))
        self.wfeAl  = np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdAl,self.kx),self.kx)))
        self.wfeN   = np.real(np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdN,self.kx),self.kx))))
        self.wfeST  = np.real(np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdST,self.kx),self.kx))))   
        self.wfeTot = np.sqrt(self.wfeFit**2 + self.wfeAl**2 + self.wfeST**2 + self.wfeN**2)
        strehl      = 100*np.exp(-(self.wfeTot/rad2nm)**2)
        
        # Print
        if self.verbose == True:
            print('\n_____ ERROR BREAKDOWN SCIENTIFIC SOURCE ',iSrc+1,' _____')
            print('------------------------------------------')
            print('.Strehl-ratio at %4.2fmicron:\t%4.2f%s'%(wvl*1e6,strehl,'%'))
            print('.Residual wavefront error:\t%4.2fnm'%self.wfeTot)
            print('.Fitting error:\t\t\t%4.2fnm'%self.wfeFit)
            print('.Aliasing error:\t\t%4.2fnm'%self.wfeAl)
            print('.Noise error:\t\t\t%4.2fnm'%self.wfeN)
            print('.Spatio-temporal error:\t\t%4.2fnm'%self.wfeST)
            print('-------------------------------------------')
            psdS   = self.servoLagPSD()
            self.wfeS   = np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdS,self.kx),self.kx)))
            print('.Sole servoLag error:\t\t%4.2fnm'%self.wfeS)
            print('-------------------------------------------')            
            if self.nGs == 1:
                psdAni = self.anisoplanatismPSD()
                self.wfeAni = np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdAni,self.kx),self.kx)))
                print('.Sole anisoplanatism error:\t%4.2fnm'%self.wfeAni)
            else:
                #psdTomo = self.tomographyPSD()
                #self.wfeTomo = np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdTomo,self.kx),self.kx)))
                self.wfeTomo = np.sqrt(self.wfeST**2 - self.wfeS)
                print('.Sole tomographic error:\t%4.2fnm'%self.wfeTomo)
            print('-------------------------------------------')
            
        return strehl
    
    def getPSF(self,verbose=False,fftphasor=True,getErrorBreakDown=False):
        """
        """
        start0 = time.time()
        
        if not self.status:
            print("The fourier Model class must be instantiated first\n")
            return 0,0
        
        # GET CONSTANTS
        dk          = 2*self.kc[0]/self.resAO
        
       
        # INITIALIZING OUTPUTS
        self.SR  = []
        self.PSF = []
        self.PSD = []
        self.FWHM= []
       
        # DEFINE THE FFT PHASOR
        if fftphasor:
             # FOURIER PHASOR
             uu =  fft.fftshift(fft.fftfreq(self.fovInPixel))  
             ux,uy = np.meshgrid(uu,uu)
             self.fftPhasor = np.exp(-complex(0,1)*np.pi*(ux + uy))
        else:
             self.fftPhasor = 1
              
        for j in range(self.nSrc):
            # UPDATE THE ATMOSPHERE WAVELENGTH
            self.srcj    = self.src[j]
            wvl          = self.srcj.wvl
            self.atm.wvl = wvl
            self.atm_mod.wvl = wvl
            
            # DEFINE THE RECONSTRUCTOR
            t1 = time.time()
            if self.nGs <2:
                self.reconstructionFilter()
            else:
                self.finalReconstructor()
                self.PbetaDMj = self.PbetaDM[j]
            self.trec = time.time() - t1
            
            # DEFINE THE CONTROLLER
            t1 = time.time()
            self.controller()
            self.tcont = time.time() - t1

            # GET THE AO RESIDUAL PSD/OTF
            t1 = time.time()
            psd   = self.powerSpectrumDensity()        
            self.PSD.append(psd)
            
            self.otfAO = fft.fftshift(FourierUtils.psd2otf(psd,dk))
            self.otfAO = self.otfAO/self.otfAO.max()
            self.totf  = time.time() - t1
            self.otfTot= self.otfAO * self.otfTel * self.fftPhasor
            
            # GET THE WAVE FRONT ERROR BREAKDOWN
            t1 = time.time()
            if getErrorBreakDown == True:
                strehl = self.errorBreakDown(iSrc=j)
            else:
                strehl = np.sum(np.abs(self.otfTot))/self.otfTel.sum()
                
            self.SR.append(strehl)
            self.terr = time.time() - t1
            
            # GET THE FINAL PSF
            t1 = time.time()
            psf = FourierUtils.otf2psf(self.otfTot)
            if self.samp <1:
                psf = FourierUtils.interpolateSupport(psf,round(self.resolution*2*self.samp))
                      
            self.PSF.append(psf)
            self.tpsf = time.time() - t1

            # GET THE FWHM
            
        self.tcalc = (time.time() - start0) 
        if self.verbose == True:
            print("Required time for total calculation (s)\t : {:f}".format(self.tcalc))
            print("Required time for calculating a PSF (s)\t : {:f}".format(self.tcalc/self.nSrc))
            print("Required time for reconstructor init (s) : {:f}".format(self.trec))
            print("Required time for tomography init (s)\t : {:f}".format(self.ttomo))
            print("Required time for optimization init (s)\t : {:f}".format(self.topt))
            print("Required time for controller init (s)\t : {:f}".format(self.tcont))
            print("Required time for otf calculation (s)\t : {:f}".format(self.totf))
            print("Required time for error calculation (s)\t : {:f}".format(self.terr))
            print("Required time for psf calculation (s)\t : {:f}".format(self.tpsf))

        self.PSF = np.array(self.PSF)
        self.PSD = np.array(self.PSD)
        self.SR  = np.array(self.SR)
        self.FWHM  = np.array(self.FWHM)
        
        return self.PSF,self.PSD,self.SR,self.FWHM
        
    def displayResults(self):
        """
        """
        # GEOMETRY
        deg2rad = np.pi/180
        plt.figure()
        plt.polar(self.azimuthSrc*deg2rad,self.zenithSrc,'ro',markersize=7,label='PSF evaluation')
        plt.polar(self.azimuthGs*deg2rad,self.zenithGs,'bs',markersize=7,label='GS position')
        plt.polar(self.azimuthOpt*deg2rad,self.zenithOpt,'kx',markersize=10,label='Optimization directions')
        plt.legend(bbox_to_anchor=(1.05, 1))
           
        # PSFs
        if np.any(self.PSF):   
            plt.figure()
            if self.PSF.shape[0] >1:
                plt.title("PSFs at {:.1f} and {:.1f} arcsec from center".format(self.zenithSrc[0],self.zenithSrc[-1]))
                P = np.concatenate((self.PSF[0,:,:],self.PSF[-1,:,:]),axis=1)
            else:
                plt.title('PSF')
                P = self.PSF[0]
            plt.imshow(np.log10(P))
        
        # STREHL-RATIO
        if np.any(self.SR):
            plt.figure()
            plt.plot(self.zenithSrc,self.SR,'bo',markersize=10)
            plt.xlabel("Off-axis distance")
            plt.ylabel("Strehl-ratio (%)")
            plt.show()
        
def demo():
    # Instantiate the FourierModel class
    path = '/home/omartin/Projects/fourierPSF/parFile/'
    fao = fourierModel(path+"mavisParams.ini",calcPSF=True,verbose=True,display=True,getErrorBreakDown=True)


    return fao

fao = demo()