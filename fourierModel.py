#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:00:44 2018

@author: omartin
"""
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import sys
import re
import scipy.special as spc
import time

import FourierUtils
from telescope import telescope
from atmosphere import atmosphere
from source import source

import pdb

def fprintf(stream, format_spec, *args):
    stream.write(format_spec % args)
    
class fourierModel:
    """ Fourier class gathering the PSD calculation for PSF reconstruction. 
    """
    
    # DEPENDANT VARIABLES DEFINITION
    @property
    def kc(self):
        """DM cut-of frequency"""
        return 0.5*(self.nActuator-1)/self.tel.D;
    
    @property
    def kcInMas(self):
        """DM cut-of frequency"""
        radian2mas = 180*3600*1e3/np.pi
        return self.kc*self.atm.wvl*radian2mas;
    
    @property
    def nTimes(self):
        """"""
        fovInPixel = int((np.ceil(2e3*self.fovInArcsec/self.psInMas))/2)
        fovInPixel   = max([fovInPixel,2*self.resAO])
        return int(np.round(fovInPixel/self.resAO))
    
    # CONTRUCTOR
    def __init__(self,file):
    
        # PARSING INPUTS
        self.status = 0
        self.file         = file  
        self.status = self.parameters(self.file)        
        
        if self.status:
            # DEFINE THE FREQUENCY VECTORS WITHIN THE AO CORRECTION BAND
            kx = 2*self.kc*fft.fftshift(fft.fftfreq(self.resAO)) + 1e-10   
            ky = 2*self.kc*fft.fftshift(fft.fftfreq(self.resAO)) + 1e-10
            self.kx,self.ky = np.meshgrid(kx,ky)
        
    def __repr__(self):
        s = "Fourier Model class"
        if self.status == 1:
            s = s + "instantiated"
        else:
            s = s + "not instantiated"
            
        return s
    
    def parameters(self,file):
                    
        if file.find(".py") > 0:
            # run the .py file
            runfile(file)
            
            # Telescope
            self.D              = D
            self.zenith_angle   = zenith_angle
            self.obsRatio       = obsRatio
            self.resolution     = resolution
            self.path_pupil     = path_pupil
            
            # Atmosphere
            self.wvlAtm         = wvlAtm*1e-9
            self.r0             = r0
            self.L0             = L0
            self.weights        = np.array(weights)
            self.heights        = np.array(heights)
            self.wSpeed         = np.array(wSpeed)
            self.wDir           = np.array(wDir)
            
            if len(self.weights) == len(self.heights) == len(self.wDir) == len(self.wSpeed):
                self.nbLayers = len(self.weights)
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('Please enter all the parameters of the different layers')
                print('\n')
                return 0
            
            # Scientific sources
            self.wvlSrc         = np.array(wvlSrc)
            self.zenithSrc      = np.array(zenithSrc)
            self.azimuthSrc     = np.array(azimuthSrc)
            
            # Guide stars
            self.wvlGs          = np.array(wvlGs)
            self.zenithGs       = np.array(zenithGs)
            self.azimuthGs      = np.array(azimuthGs)
            self.heightGs       = heightGs
            
            # AO parameters
            self.nActuator      = nActuator
            self.noiseVariance  = noiseVariance
            self.loopGain       = loopGain
            self.samplingTime   = samplingTime*1e-3
            self.latency        = latency*1e-3
            self.resAO          = resAO
            self.psInMas        = psInMas
            self.fovInArcsec    = fovInArcsec
            
            # Optimization
            self.theta_x        = np.array(theta_x)/206264.8
            self.theta_y        = np.array(theta_y)/206264.8
            self.theta_w        = theta_w/np.sum(theta_w)
            self.h_dm           = h_dm
            self.condmax        = condmax

        elif file.find(".txt") > 0:
            # READ THE .TXT FILE
            fichier = open(file,"r")
            values = []
            src = 0
            
            self.weights = [] ; self.heights = [] ; self.wSpeed = [] ; self.wDir = []
            self.theta_x = [] ; self.h_dm = [] ; self.h_recons = [] ; self.theta_y = [] ; self.theta_w = []
            self.wvlSrc = [] ; self.zenithSrc = [] ; self.azimuthSrc = [] ; self.heightSrc = []
            self.wvlGs = [] ; self.zenithGs = [] ; self.azimuthGs = [] ; self.heightGs = []
            
            for line in fichier:
                keys = line.split(';')
                if re.search("path_pupil",keys[0])!=None:
                    keys = line.split('"')
                    self.path_pupil = keys[1]
                elif re.search("SCIENTIFIC SOURCE",keys[0])!=None:
                    src = 1
                elif re.search("GUIDE STAR",keys[0])!=None:
                    src = 2
                elif re.search("weights",keys[0])!=None:
                    for i in range(len(keys)-2):
                        self.weights.append(keys[i+1])
                        values.append(0)
                elif re.search("heights",keys[0])!=None:
                    for i in range(len(keys)-2):
                        self.heights.append(keys[i+1])
                        values.append(0)
                elif re.search("wSpeed",keys[0])!=None:
                    for i in range(len(keys)-2):
                        self.wSpeed.append(keys[i+1])
                        values.append(0)
                elif re.search("wDir",keys[0])!=None:
                    for i in range(len(keys)-2):
                        self.wDir.append(keys[i+1])
                        values.append(0)
                elif re.search("theta_x",keys[0])!=None:
                    for i in range(len(keys)-2):
                        self.theta_x.append(keys[i+1])
                elif re.search("theta_y",keys[0])!=None:
                    for i in range(len(keys)-2):
                        self.theta_y.append(keys[i+1])
                elif re.search("theta_w",keys[0])!=None:
                    for i in range(len(keys)-2):
                        self.theta_w.append(keys[i+1])
                elif re.search("h_dm",keys[0])!=None:
                    for i in range(len(keys)-2):
                        self.h_dm.append(keys[i+1])
                elif re.search("h_recons",keys[0])!=None:
                    for i in range(len(keys)-2):
                        self.h_recons.append(keys[i+1])
                elif re.search("wvl",keys[0])!=None:
                    if src == 1 :
                        for i in range(len(keys)-2):
                            self.wvlSrc.append(keys[i+1])
                    elif src==2:
                        for i in range(len(keys)-2):
                            self.wvlGs.append(keys[i+1])
                elif re.search("zenith",keys[0])!=None:
                    if src == 1 :
                        for i in range(len(keys)-2):
                            self.zenithSrc.append(keys[i+1])
                    else:
                        for i in range(len(keys)-2):
                            self.zenithGs.append(keys[i+1])
                elif re.search("azimuth",keys[0])!=None:
                    if src == 1 :
                        for i in range(len(keys)-2):
                            self.azimuthSrc.append(keys[i+1])
                    elif src == 2:
                        for i in range(len(keys)-2):
                            self.azimuthGs.append(keys[i+1])
                elif re.search("height",keys[0])!=None:
                    if src == 1 :
                        for i in range(len(keys)-2):
                            self.heightSrc.append(keys[i+1])
                    else:
                        for i in range(len(keys)-2):
                            self.heightGs.append(keys[i+1])
                elif len(keys) == 3 :
                    values.append(keys[1])
                    
            values          = list(map(float,values))
            
            if len(self.weights) == len(self.heights) == len(self.wDir) == len(self.wSpeed):
                self.nbLayers = len(self.weights)
            else:
                print('%%%%%%%% ERROR %%%%%%%%')
                print('Please enter all the parameters of the different layers')
                print('\n')
                return 0
            
            # Telescope
            self.D              = values[3+4*self.nbLayers]
            self.zenith_angle   = values[4+4*self.nbLayers]
            self.obsRatio       = values[5+4*self.nbLayers]
            self.resolution     = values[6+4*self.nbLayers]
            
            #Atmosphere
            self.r0             = np.array(values[0])
            self.L0             = values[1]
            self.wvlAtm         = values[2]*1e-9
            self.heights        = np.array(list(map(float,self.heights)))
            self.weights        = np.array(list(map(float,self.weights)))
            self.weights        = self.weights/np.sum(self.weights)
            self.wSpeed         = np.array(list(map(float,self.wSpeed)))
            self.wDir           = np.array(list(map(float,self.wDir)))
            
            # Scientific sources
            self.wvlSrc         = np.array(list(map(float,self.wvlSrc)))
            self.zenithSrc      = np.array( list(map(float,self.zenithSrc)))
            self.azimuthSrc     = np.array(list(map(float,self.azimuthSrc)))
            #self.heightSrc  = np.array(list(map(float,self.heightSrc)))
            
            # Guide stars
            self.wvlGs          = np.array(list(map(float,self.wvlGs)))
            self.zenithGs       = np.array(list(map(float,self.zenithGs)))
            self.azimuthGs      = np.array(list(map(float,self.azimuthGs)))
            self.heightGs       = np.array(list(map(float,self.heightGs)))
            
            # AO parameters
            self.nActuator      = values[7+4*self.nbLayers]
            self.noiseVariance  = values[8+4*self.nbLayers]
            self.loopGain       = values[9+4*self.nbLayers]
            self.samplingTime   = values[10+4*self.nbLayers]*10e-4
            self.latency        = values[11+4*self.nbLayers]*10e-4
            self.resAO          = int(values[12+4*self.nbLayers])
            self.psInMas        = values[13+4*self.nbLayers]
            self.fovInArcsec    = values[14+4*self.nbLayers]
            
            # Optimization
            self.theta_x        = np.array(list(map(float,self.theta_x)))/206264.8
            self.theta_y        = np.array(list(map(float,self.theta_y)))/206264.8
            self.theta_w        = np.array(list(map(float,self.theta_w)))
            self.theta_w        = self.theta_w/np.sum(self.theta_w)
            self.h_dm           = np.array(list(map(float,self.h_dm)))
            self.h_recons       = np.array(list(map(float,self.h_recons)))
            self.condmax        = values[-1]
            
            fichier.close()
            
        
        #%% instantiating sub-classes
        
        # Telescope
        self.tel = telescope(self.D,self.zenith_angle,self.obsRatio,self.resolution,self.path_pupil)
        
        # Strechning factor (LGS case)
        if len(self.weights) == len(self.heights) == len(self.wDir) == len(self.wSpeed):
            self.nbLayers = len(self.weights)
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of atmospheric layers is not consistent in the parameters file\n')
            return 0
            
        self.heights  = self.heights*self.tel.airmass
        self.heightGs = self.heightGs*self.tel.airmass # LGS height
        if self.heightGs > 0:
            self.heights = self.heights/(1 - self.heights/self.heightGs)

        # True atmosphere
        self.atm = atmosphere(self.wvlAtm,self.r0*self.tel.airmass**(-3/5),self.weights,self.heights,self.wSpeed,self.wDir,self.L0)
        self.h_recons = self.heights

        # Scientific Sources
        self.src = []
        if len(self.wvlSrc) == len(self.zenithSrc) == len(self.azimuthSrc):
            self.nSrc = len(self.wvlSrc)
            for n in range(self.nSrc):
                self.src.append(source(self.wvlSrc[n]*1e-9,self.zenithSrc[n],self.azimuthSrc[n],0,n+1,"SCIENTIFIC STAR",verbose=True))
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of scientific sources is not consistent in the parameters file\n')
            return 0
            
        # Guide stars
        self.gs = []
        if len(self.wvlGs) == len(self.zenithGs) == len(self.azimuthGs):
            self.nGs = len(self.wvlGs)
            for n in range(self.nGs):
                self.gs.append(source(self.wvlGs[n]*1e-9,self.zenithGs[n],self.azimuthGs[n],self.heightGs,n+1,"GUIDE STAR",verbose=True))
        else:
            print('%%%%%%%% ERROR %%%%%%%%')
            print('The number of guide stars is not consistent in the parameters file\n')
            return 0
        
        return 1
    
#%% RECONSTRUCTOR DEFINITION    
    def reconstructionFilter(self,kx,ky):
        """
        """          
        # Get spectrums    
        Wn   = self.noiseVariance/(2*self.kc)**2
        k    = np.hypot(kx,ky)           
        Wphi = self.atm.spectrum(k);
            
        # reconstructor derivation
        i    = complex(0,1)
        d    = self.tel.D/(self.nActuator-1)     
        Sx   = 2*i*np.pi*kx*d
        Sy   = 2*i*np.pi*ky*d                        
        Av   = np.sinc(d*kx)*np.sinc(d*ky)*np.exp(i*np.pi*d*(kx+ky))        
        self.SxAv = Sx*Av
        self.SyAv = Sy*Av
        gPSD = abs(self.SxAv)**2 + abs(self.SyAv)**2 + Wn/Wphi
        self.Rx = np.conjugate(self.SxAv)/gPSD
        self.Ry = np.conjugate(self.SyAv)/gPSD
                
        # Manage NAN value if any   
        self.Rx[np.isnan(self.Rx)] = 0
        self.Ry[np.isnan(self.Ry)] = 0
            
        # Set central point (i.e. kx=0,ky=0) to zero
        N = int(np.ceil((kx.shape[0]-1)/2))
        self.Rx[N,N] = 0
        self.Ry[N,N] = 0

    def tomographicReconstructor(self,kx,ky):
        k    = np.hypot(kx,ky)     
        nK   = len(k[0,:])
        nL = len(self.h_recons)
        nGs = self.nGs
        Alpha = np.zeros([2,nGs])
        for j in range(nGs):
            Alpha[0,j] = self.gs[j].direction[0]
            Alpha[1,j] = self.gs[j].direction[1]
            
        # Measure matrix
        i    = complex(0,1)
        d    = self.tel.D/(self.nActuator-1)   #sub-aperture size
        index  = k <= self.kc
        
        M    = np.zeros([nK,nK,nGs,nGs],dtype=complex)
        for j in np.arange(0,nGs):
            M[index,j,j] = 2*i*np.pi*k[index]*np.sinc(d*kx[index])*np.sinc(d*ky[index])
        self.M = M
        # Projection matrix
        P    = np.zeros([nK,nK,nGs,nL],dtype=complex)
        
        for n in range(nL):
            for j in range(nGs):
                fx = kx*Alpha[0,j]
                fy = ky*Alpha[1,j]
                P[index,j,n] = np.exp(i*2*np.pi*self.h_recons[n]*(fx[index]+fy[index]))
                
        MP = np.matmul(M,P)
        MP_t = np.conj(MP.transpose(0,1,3,2))
        
        Cb = np.zeros([nK,nK,nGs,nGs],dtype=complex)
        for j in range(nGs):
            Cb[:,:,j,j]     = self.noiseVariance
        self.Cb = Cb
        
        Cphi = np.zeros ([nK,nK,nL,nL],dtype=complex)
        cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2./(2.*np.pi**(11/3)))
        for j in range(nL):
            Cphi[:,:,j,j] = self.atm.layer[j].weight * self.atm.r0**(-5/3)*cte*(k**2 + 1/self.atm.L0**2)**(-11/6)\
            *FourierUtils.pistonFilter(self.tel.D,k)
        self.Cphi = Cphi
                
        to_inv = np.matmul(np.matmul(MP,Cphi),MP_t) + self.Cb 
        inv = np.zeros(to_inv.shape,dtype=complex)
        
        for x in range(to_inv.shape[0]):
            for y in range(to_inv.shape[1]):
                if index[x,y] == True : 
                    u,s,vh = np.linalg.svd(to_inv[x,y,:,:])
                    slim = np.max(s)/self.condmax
                    rank = np.sum(s > slim)
                    u = u[:, :rank]
                    u /= s[:rank]
                    inv[x,y,:,:] =  np.transpose(np.conjugate(np.dot(u, vh[:rank])))
                
        Wtomo = np.matmul(np.matmul(Cphi,MP_t),inv)
        
        return Wtomo
        
    def optimalProjector(self,kx,ky):
        nDm = len(self.h_dm)
        nDir = (len(self.theta_x))
        nL = len(self.h_recons)
        nK = len(kx[0,:])
        i    = complex(0,1)
        index  = np.hypot(kx,ky) <= self.kc
        
        mat1 = np.zeros([nK,nK,nDm,nL],dtype=complex)
        to_inv = np.zeros([nK,nK,nDm,nDm],dtype=complex)
        for d_o in range(nDir):                 #loop on optimization directions
            Pdm = np.zeros([nK,nK,1,nDm],dtype=complex)
            Pl = np.zeros([nK,nK,1,nL],dtype=complex)
            fx = self.theta_x[d_o]*kx
            fy = self.theta_y[d_o]*ky
            for j in range(nDm):                # loop on DM
                Pdm[index,0,j] = np.exp(i*2*np.pi*self.h_dm[j]*(fx[index]+fy[index]))
            Pdm_t = np.conj(Pdm.transpose(0,1,3,2))
            for j in range(nL):                 #loop on atmosphere layers
                Pl[index,0,j] = np.exp(i*2*np.pi*self.h_recons[j]*(fx[index]+fy[index]))
                
            mat1   += np.matmul(Pdm_t,Pl)*self.theta_w[d_o]
            to_inv += np.matmul(Pdm_t,Pdm)*self.theta_w[d_o]
            
        mat2 = np.zeros(to_inv.shape,dtype=complex)
        
        for x in range(to_inv.shape[0]):
            for y in range(to_inv.shape[1]):
                if index[x,y] == True :
                    u,s,vh = np.linalg.svd(to_inv[x,y,:,:])
                    slim = np.max(s)/self.condmax
                    rank = np.sum(s > slim)
                    u = u[:, :rank]
                    u /= s[:rank]
                    mat2[x,y,:,:] =  np.transpose(np.conjugate(np.dot(u, vh[:rank])))
        
        Popt = np.matmul(mat2,mat1)
        
        return Popt
    
    def finalReconstructor(self,kx,ky):
        self.Wtomo = self.tomographicReconstructor(kx,ky)
        self.Popt = self.optimalProjector(kx,ky)
        self.W = np.matmul(self.Popt,self.Wtomo)
        
        # Computation of the Pbeta^DM matrix
        k = np.hypot(kx,ky)
        index  = k <= self.kc 
        nDm = len(self.h_dm)
        nK = len(k[0,:])

        self.PbetaDM = []
        for s in range(self.nSrc):
            fx = self.src[s].direction[0]*kx
            fy = self.src[s].direction[1]*ky
            PbetaDM = np.zeros([nK,nK,1,nDm],dtype=complex)
            for j in range(nDm): #loop on DMs
                PbetaDM[index,0,j] = np.exp(complex(0,1)*2*np.pi*self.h_dm[j]*(fx[index] + fy[index]))
            self.PbetaDM.append(PbetaDM)
        
        
#%% CONTROLLER DEFINITION
    def  controller(self,kx,ky,nTh=10):
        """
        """
        
        i  = complex(0,1)
        vx          = self.atm.wSpeed*np.cos(self.atm.wDir*np.pi/180)
        vy          = self.atm.wSpeed*np.sin(self.atm.wDir*np.pi/180)   
        nPts        = kx.shape[0]
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
        nF     = 500   
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
                fi      = -vx[l]*kx*costh[iTheta] - vy[l]*ky*costh[iTheta]
                idx     = abs(fi) <1e-7;
                z       = np.exp(-2*i*np.pi*fi*Ts)
                fi[idx] = 1e-8*np.sign(fi[idx])
                hInt    = self.loopGain/(1-z**(-1))
                rtfInt  = 1/(1+hInt*z**(-delay))
                atfInt  = hInt*z**(-delay)*rtfInt
                
                # ao transfer function
                MAG = abs(atfInt)                
                MAG[fi == 0] = 1
                PH  = np.angle(atfInt)                
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
    def fittingPSD(self,kx,ky,aoFilter='circle'):
        """ FITTINGPSD Fitting error power spectrum density """                 
        #Instantiate the function output
        resExt      = kx.shape[0]*self.nTimes 
        kxExt       = 2*self.nTimes*self.kc*fft.fftshift(fft.fftfreq(resExt))    
        kyExt       = 2*self.nTimes*self.kc*fft.fftshift(fft.fftfreq(resExt))            
        kxExt,kyExt = np.meshgrid(kxExt,kyExt)        
        psd         = np.zeros((resExt,resExt))
            
        # Define the correction area
        if aoFilter == 'square':
            index  = (abs(kxExt)>self.kc) | (abs(kyExt)>self.kc)            
        elif aoFilter == 'circle':
            index  = np.hypot(kxExt,kyExt) > self.kc
            
        kExt       = np.hypot(kxExt[index],kyExt[index])
        psd[index] = self.atm.spectrum(kExt)
        
        return psd*FourierUtils.pistonFilter(self.tel.D,np.hypot(kxExt,kyExt))
    
    def aliasingPSD(self,kx,ky,aoFilter='circle'):
        """
        """
        kc = self.kc
        psd = np.zeros(kx.shape)
        tmp = psd
        if aoFilter == 'square':
            index  = (abs(kx) <=kc) | (abs(ky) <= kc)               
        elif aoFilter == 'circle':
            index  = np.hypot(kx,ky) <= self.kc    
        if self.nGs < 2:
            i  = complex(0,1)
            k  = np.hypot(kx,ky)
            d  = self.tel.D/(self.nActuator-1)
            T  = self.samplingTime
            td = self.latency        
            vx = self.atm.wSpeed*np.cos(self.atm.wDir*np.pi/180)
            vy = self.atm.wSpeed*np.sin(self.atm.wDir*np.pi/180)
            Rx = self.Rx
            Ry = self.Ry
        
        
            if self.loopGain == 0:
                tf = 1
            else:
                tf = self.h1#np.reshape(self.h1[index],(side,side))
                    
            weights = self.atm.weights
        
            w = 2*i*np.pi*d;
            for mi in np.arange(-self.nTimes,self.nTimes+1):
                for ni in np.arange(-self.nTimes,self.nTimes+1):
                    if (mi!=0) | (ni!=0):
                        km = kx - mi/d
                        kn = ky - ni/d
                        PR = FourierUtils.pistonFilter(self.tel.D,k,fm=mi/d,fn=ni/d)
                        W_mn = (abs(km**2 + kn**2) + 1/self.atm.L0**2)**(-11/6)     
                        Q = (Rx*w*km + Ry*w*kn) * (np.sinc(d*km)*np.sinc(d*kn))
                        avr = 0
                        
                        for l in np.arange(0,self.atm.nL):
                            avr = avr + weights[l]* (np.sinc(km*vx[l]*T)*np.sinc(kn*vy[l]*T)
                            *np.exp(2*i*np.pi*km*vx[l]*td)*np.exp(2*i*np.pi*kn*vy[l]*td)*tf)
                                                          
                        tmp = tmp + PR*W_mn * abs(Q*avr)**2
               
            tmp[np.isnan(tmp)] = 0        
            psd[index]         = tmp[index]
        
        return psd*self.atm.r0**(-5/3)*0.0229 
            
    def noisePSD(self,kx,ky,aoFilter='circle'):
        """NOISEPSD Noise error power spectrum density
        """
        kc = self.kc;
        psd = np.zeros(kx.shape,dtype=complex)
        if self.noiseVariance > 0:
            if aoFilter == 'square':
                index  = (abs(kx) <=kc) | (abs(ky) <= kc)               
            elif aoFilter == 'circle':
                index  = np.hypot(kx,ky) <= self.kc  
            if self.nGs < 2:        
                psd[index] = self.noiseVariance/(2*self.kc)**2*(abs(self.Rx[index])**2 + abs(self.Ry[index]**2));
                
            else:  
                nK   = len(np.hypot(kx,ky)[0,:])
                
                W = self.W
                PbetaDM = self.PbetaDMj
                Cb = self.Cb
                
                PW = np.matmul(PbetaDM,W)
                PW_t = np.conj(PW.transpose(0,1,3,2))
                
                for x in range(nK):
                    for y in range(nK):
                        if index[x,y] == True:
                            psd[x,y] = np.dot(PW[x,y,:,:],np.dot(Cb[x,y,:,:],PW_t[x,y,:,:]))

        return psd*FourierUtils.pistonFilter(self.tel.D,np.hypot(kx,ky))*self.noiseGain
    
    def servoLagPSD(self,kx,ky,aoFilter='circle'):
        """ SERVOLAGPSD Servo-lag power spectrum density
        """
            
        kc = self.kc
        psd = np.zeros(kx.shape)
        if aoFilter == 'square':
            index  = (abs(kx) <=kc) | (abs(ky) <= kc)               
        elif aoFilter == 'circle':
            index  = np.hypot(kx,ky) <= self.kc     
            
        if self.nGs == 1:        
            F = (self.Rx[index]*self.SxAv[index] + self.Ry[index]*self.SyAv[index])
            Wphi = self.atm.spectrum(np.hypot(kx,ky))
        
            if (self.loopGain == 0):
                psd[index] = abs(1-F)**2*Wphi[index]
            else:
                psd[index] = (1 + abs(F)**2*self.h2[index] - 
                   2*np.real(F*self.h1[index]))*Wphi[index]
            
        return psd*FourierUtils.pistonFilter(self.tel.D,np.hypot(kx,ky))
    
    def spatioTemporalPSD(self,kx,ky,iSrc=0,aoFilter='circle'): #def spatioTemporalPSD
        """%% ANISOSERVOLAGPSD Anisoplanatism + Servo-lag power spectrum density
        """
           
        kc = self.kc
        psd = np.zeros(kx.shape)
        k = np.hypot(kx,ky)
        if aoFilter == 'square':
            index  = (abs(kx) <=kc) | (abs(ky) <= kc)               
        elif aoFilter == 'circle':
            index  = np.hypot(kx,ky) <= self.kc       
        
        if self.nGs < 2:
            heights = self.atm.heights
            weights = self.atm.weights
            A       = 0*kx
            if sum(sum(abs(self.srcj.direction - self.gs[0].direction)))!=0:
                th  = self.srcj.direction - self.gs[0].direction
                for l in np.arange(0,self.atm.nL):                
                    red = 2*np.pi*heights[l]*(kx*th[0] + ky*th[1])
                    A   = A + weights[l]*np.exp(complex(0,1)*red)            
            else:
                A = np.ones(kx.shape)
        
            F = (self.Rx[index]*self.SxAv[index] + self.Ry[index]*self.SyAv[index])
            Wphi = self.atm.spectrum(np.hypot(kx,ky))   
            
            if (self.loopGain == 0):  
                psd[index] = abs(1-F)**2*A*Wphi[index]
            else:
                psd[index] = (1 + abs(F)**2*self.h2[index] 
                - 2*np.real(F*self.h1[index]*A[index]))*Wphi[index]
                                
        else:
            
            deltaT = self.latency+self.samplingTime
            nK = len(k[0,:])
            nH = self.nbLayers
            Hs = self.heights
            i    = complex(0,1)
            d    = self.tel.D/(self.nActuator-1)
            wDir_x = np.cos(self.wDir*np.pi/180)
            wDir_y = np.sin(self.wDir*np.pi/180)
            
            Beta = [self.srcj.direction[0],self.srcj.direction[1]]
            
            MPalphaL = np.zeros([nK,nK,self.nGs,nH],dtype=complex)
            for h in range(nH):
                www = np.sinc(self.samplingTime*self.wSpeed[h]*(wDir_x[h]*kx + wDir_y[h]*ky))
                for g in range(self.nGs):
                    Alpha = [self.gs[g].direction[0],self.gs[g].direction[1]]
                    fx = Alpha[0]*kx
                    fy = Alpha[1]*ky
                    MPalphaL[index,g,h] = www[index]*2*i*np.pi*k[index]*np.sinc(d*kx[index])*np.sinc(d*ky[index])*np.exp(i*2*np.pi*Hs[h]*(fx[index]+fy[index]))
            
            PbetaL = np.zeros([nK,nK,1,nH],dtype=complex)
            fx = Beta[0]*kx
            fy = Beta[1]*ky
            for j in range(nH):
                PbetaL[index,0,j] = np.exp(i*2*np.pi*( Hs[j]*(fx[index]+fy[index]) -  deltaT*self.wSpeed[j]*(wDir_x[j]*kx[index] + wDir_y[j]*ky[index]) ))
            
            PbetaDM = self.PbetaDMj
            W = self.W
            Cphi = self.Cphi
            
            proj = np.zeros([nK,nK,1,nH],dtype=complex)
            for x in range(nK):
                for y in range(nK):
                    if index[x,y] == True:
                        proj[x,y] = PbetaL[x,y]- np.dot(PbetaDM[x,y,:,:],np.dot(W[x,y,:,:],MPalphaL[x,y,:,:]))
            proj_t = np.conj(proj.transpose(0,1,3,2))
            
            psd = np.matmul(proj,np.matmul(Cphi,proj_t))
            psd = psd.reshape(nK,nK)

        return psd*FourierUtils.pistonFilter(self.tel.D,k)
    
    def anisoplanatismPSD(self,kx,ky,iSrc=0,aoFilter='circle'):
        """%% ANISOPLANATISMPSD Anisoplanatism power spectrum density
        """
        
        kc = self.kc
        psd = np.zeros(kx.shape)
        if aoFilter == 'square':
            index  = (abs(kx) <=kc) | (abs(ky) <= kc)               
        elif aoFilter == 'circle':
            index  = np.hypot(kx,ky) <= self.kc        
        
        heights = self.atm.heights
        weights = self.atm.weights
        A       = 0*kx
        if sum(sum(abs(self.srcj.direction - self.gs[0].direction)))!=0:
            th  = self.srcj.direction - self.gs[0].direction
            for l in np.arange(0,self.atm.nL):
                red = 2*np.pi*heights[l]*(kx*th[0] + ky*th[1])
                A   = A + 2*weights[l]*( 1 - np.cos(red) )     
        else:
            A = np.zeros(kx.shape)
        
        Wphi       = self.atm.spectrum(np.hypot(kx,ky))   
        psd[index] = A[index]*Wphi[index]
        
        return psd*FourierUtils.pistonFilter(self.tel.D,np.hypot(kx,ky))
    
    def powerSpectrumDensity(self,kx,ky,iSrc=0,aoFilter='circle'):
        """ POWER SPECTRUM DENSITY AO system power spectrum density
        """
        kc          = self.kc 
        resExt      = kx.shape[0]*self.nTimes 
        kxExt       = 2*self.nTimes*kc*fft.fftshift(fft.fftfreq(resExt))    
        kyExt       = 2*self.nTimes*kc*fft.fftshift(fft.fftfreq(resExt))            
        kxExt,kyExt = np.meshgrid(kxExt,kyExt)        
        psd         = np.zeros((resExt,resExt),dtype=complex)
                      
        index  = (abs(kxExt) <= kc) & (abs(kyExt) <= kc) 
        # Sums PSDs
        noise   = self.noisePSD(kx,ky,aoFilter=aoFilter)
        alias   = self.aliasingPSD(kx,ky,aoFilter=aoFilter)
        spatio  = self.spatioTemporalPSD(kx,ky,iSrc=iSrc,aoFilter=aoFilter)
        tmp     = noise + alias + spatio
        
        psd[np.where(index)] = tmp.ravel()
        return np.real(psd + self.fittingPSD(kx,ky,aoFilter=aoFilter))
    
    def errorBreakDown(self,iSrc=0,aoFilter='circle',verbose=True):
        """
        """
        # Constants
        wvl    = self.src[iSrc].wvl
        rad2nm = 1e9*wvl/2/np.pi        
        kx     = self.kx
        ky     = self.ky
        # DEFINE THE FREQUENCY VECTORS ACROSS ALL SPATIAL FREQUENCIES
        self.resExt = self.resAO*self.nTimes
        kxExt       = 2*self.nTimes*self.kc*fft.fftshift(fft.fftfreq(self.resExt))    
        kyExt       = 2*self.nTimes*self.kc*fft.fftshift(fft.fftfreq(self.resExt))            
        kxExt,kyExt = np.meshgrid(kxExt,kyExt)
        
        # Get PSDs
        psdFit = self.fittingPSD(kx,ky,aoFilter=aoFilter)
        psdAl  = self.aliasingPSD(kx,ky,aoFilter=aoFilter)
        psdN   = self.noisePSD(kx,ky,aoFilter=aoFilter)
        psdST  = self.spatioTemporalPSD(kx,ky,iSrc=iSrc,aoFilter=aoFilter)
        psdS   = self.servoLagPSD(kx,ky,aoFilter=aoFilter)
        psdAni = self.anisoplanatismPSD(kx,ky,iSrc=iSrc,aoFilter=aoFilter)
        
        # Derives wavefront error
        self.wfeFit = np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdFit,kxExt),kxExt)))
        self.wfeAl  = np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdAl,kx),kx)))
        self.wfeN   = np.real(np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdN,kx),kx))))
        self.wfeST  = np.real(np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdST,kx),kx))))
        self.wfeS   = np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdS,kx),kx)))
        self.wfeAni = np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdAni,kx),kx)))
        self.wfeTot = np.sqrt(self.wfeFit**2 + self.wfeAl**2 + self.wfeST**2 + self.wfeN**2)
        strehl      = 100*np.exp(-(self.wfeTot/rad2nm)**2)
        
        # Print
        if verbose == True:
            print('\n_____ ERROR BREAKDOWN SCIENTIFIC SOURCE ',iSrc+1,' _____')
            print('------------------------------------------')
            fprintf(sys.stdout,'.Strehl-ratio at %4.2fmicron:\t%4.2f%%\n',wvl*1e6,strehl)
            fprintf(sys.stdout,'.Residual wavefront error:\t%4.2fnm\n',self.wfeTot)
            fprintf(sys.stdout,'.Fitting error:\t\t\t%4.2fnm\n',self.wfeFit)
            fprintf(sys.stdout,'.Aliasing error:\t\t%4.2fnm\n',self.wfeAl)
            fprintf(sys.stdout,'.Noise error:\t\t\t%4.2fnm\n',self.wfeN)
            fprintf(sys.stdout,'.Aniso+servoLag error:\t\t%4.2fnm\n',self.wfeST)
            print('-------------------------------------------')
            fprintf(sys.stdout,'.Sole anisoplanatism error:\t%4.2fnm\n',self.wfeAni)
            fprintf(sys.stdout,'.Sole servoLag error:\t\t%4.2fnm\n',self.wfeS)
            print('-------------------------------------------')
            
        return strehl
    
    def getPSF(self,aoFilter='circle',nyquistSampling=False,verbose=False):
        """
        """
        if not self.status:
            print("The fourier Model class must be instantiated first\n")
            return 0,0
        
        # GET CONSTANTS
        psInMas     = self.psInMas
        fovInArcsec = self.fovInArcsec
        dk          = 2*self.kc/self.resAO
        
        # TELESCOPE OTF AT NYQUIST-SAMPLING
        otfTel = FourierUtils.pupil2otf(self.tel.pupil,0*self.tel.pupil,2)
        
        # INITIALIZING OUTPUTS
        self.SR  = []
        self.PSF = []
        self.PSD = []
        
       
        for j in range(self.nSrc):
            # UPDATE THE ATMOSPHERE WAVELENGTH
            self.srcj    = self.src[j]
            wvl          = self.srcj.wvl
            self.atm.wvl = wvl
            
            # CALCULATING THE PSF SAMPLING        
            lonD  = (1e3*180*3600/np.pi*wvl/self.tel.D)
            if nyquistSampling == True:
                nqSmpl = 1
                psInMas= lonD/2
            else:
                nqSmpl= lonD/psInMas/2
                
            fovInPixel = int((np.ceil(2e3*fovInArcsec/psInMas))/2)
            fovInPixel = max([fovInPixel,2*self.resAO])
            if verbose:
                fprintf(sys.stdout,'.Field of view:\t\t%4.2f arcsec\n.Pixel scale:\t\t%4.2f mas\n.Nyquist sampling:\t%4.2f',fovInPixel*psInMas/1e3,psInMas,nqSmpl)
                print('\n-------------------------------------------\n')
                
            # DEFINE THE RECONSTRUCTOR
            if self.nGs <2:
                self.reconstructionFilter(self.kx,self.ky)
            else:
                self.finalReconstructor(self.kx,self.ky)
                self.PbetaDMj = self.PbetaDM[j]
            
            # DEFINE THE CONTROLLER
            self.controller(self.kx,self.ky)
            
            # GET THE AO RESIDUAL PSD/OTF     
            psd   = self.powerSpectrumDensity(self.kx,self.ky,iSrc=j,aoFilter=aoFilter)        
            self.PSD.append(psd)
            psd   = FourierUtils.enlargeSupport(psd,2)
            
            otfAO = fft.fftshift(FourierUtils.psd2otf(psd,dk))
            otfAO = FourierUtils.interpolateSupport(otfAO,2*self.tel.resolution)
            otfAO = otfAO/otfAO.max()
            
            # GET THE WAVE FRONT ERROR BREAKDOWN
            strehl = self.errorBreakDown(iSrc=j,verbose=verbose)
            self.SR.append(strehl)
            
            # GET THE FINAL PSF
            self.PSF.append(FourierUtils.otfShannon2psf(otfAO * otfTel,nqSmpl,fovInPixel))
        
        return self.PSF,self.PSD
        
    
def demo():
    # Instantiate the FourierModel class
    start = time.time()
    fao = fourierModel("_parFile_/Parameters.py")
    elapsed_time_init = (time.time() - start) 
    print("Required time for initialization (s) : {:f}".format(elapsed_time_init))
    
    if fao.status:
        # Calculate the PSDs/PSFs
        start = time.time()
        PSF,PSD = fao.getPSF()
        elapsed_time_calc = (time.time() - start) 
        print("Required time for calculation (s) : {:f}".format(elapsed_time_calc))
        print("Required time for calculation/PSF (s) : {:f}".format(elapsed_time_calc/fao.nSrc))
        
        # Display the 2D PSFs/PSDs
        #for j in range(fao.nSrc):
        plt.figure()
        plt.title("PSF on-axis")
        plt.imshow(np.log10(PSF[0]))
        
        plt.figure()
        plt.title("PSF off-axis")
        plt.imshow(np.log10(PSF[fao.nSrc-1]))
        
        # Plot the Strehl-ratio versus the PSF position
        plt.figure()
        plt.plot(fao.zenithSrc,fao.SR)
        plt.xlabel("Off-axis distance")
        plt.ylabel("Strehl-ratio (%)")
        plt.show()
    
    return fao

fao = demo()