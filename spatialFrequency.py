#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 15:00:44 2018

@author: omartin
"""
import numpy as np
import numpy.fft as fft
import matplotlib.pyplot as plt
import p3utils
import sys
from  telescope import telescope
from  atmosphere import atmosphere
from  source import source

def fprintf(stream, format_spec, *args):
    stream.write(format_spec % args)
    
class spatialFrequency:
    """ Fourier class gathering the PSD calculation for PSF reconstruction. 
    Inputs are:
        - tel
        - atm
        - src
        - nActuator
        - noiseVariance
        - loopGain
        - samplingTime
        - latency
        - resAO
                
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
    def __init__(self,tel, atm,src,nActuator, noiseVariance, loopGain,
                 samplingTime, latency, resAO,psInMas,fovInArcsec, nTimes=4, nGs=1):
    
        # PARSING INPUTS
        self.tel          = tel
        self.atm          = atm
        self.src          = src
        self.nActuator    = nActuator
        self.noiseVariance= noiseVariance
        self.loopGain     = loopGain
        self.samplingTime = samplingTime
        self.latency      = latency
        self.resAO        = resAO
        self.psInMas      = psInMas
        self.fovInArcsec  = fovInArcsec
        #self.nTimes       = nTimes
        self.nGs          = nGs
        self.atm.wvl = self.src.wvl
        
        # DEFINE THE FREQUENCY VECTORS WITHIN THE AO CORRECTION BAND
        kx = 2*self.kc*fft.fftshift(fft.fftfreq(self.resAO)) + 1e-10   
        ky = 2*self.kc*fft.fftshift(fft.fftfreq(self.resAO)) + 1e-10
        self.kx,self.ky = np.meshgrid(kx,ky)
                
        # DEFINE THE RECONSTRUCTOR
        if self.nGs ==1:
            self.reconstructionFilter(self.kx,self.ky)
        else:
            self.tomographicReconstructor(self.kx,self.ky)
        
        # DEFINE THE CONTROLLER
        self.controller(self.kx,self.ky)
        
    def __repr__(self):
        s = "Spatial Frequency \n kc=%.2fm^-1"%self.kc
        return s
        
        
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
        """
        """          
        # Get spectrums    
        #Wn   = self.noiseVariance/(2*self.kc)**2
        k    = np.hypot(kx,ky)     
        nK   = len(k[0,:])
        #Wphi = self.atm.spectrum(k);
            
        # Measure matrix
        i    = complex(0,1)
        d    = self.tel.D/(self.nActuator-1)     
        Sx   = 2*i*np.pi*kx*d
        Sy   = 2*i*np.pi*ky*d                        
        Av   = np.sinc(d*kx)*np.sinc(d*ky)*np.exp(i*np.pi*d*(kx+ky))   
        
        #pdb.set_trace()
        
        M    = np.zeros([2*self.nGs,self.nGs,nK,nK])
        for i in np.arange(0,self.nGs):
            M[2*i,i,:,:] = Sx*Av            
            M[2*i+1,i,:,:] = Sy*Av     
        self.M = M   
        # Projection matrix
        P    = np.zeros([self.atm.nL,self.nGs,nK,nK])
        
        #MP   = M*P
        
        #Tomographic reconstructor
        #self.SxAv = Sx*Av
        self.SyAv = Sy*Av
        #gPSD = abs(self.SxAv)**2 + abs(self.SyAv)**2 + Wn/Wphi
        
        #self.Rt = inv(inv(MP.T*Wn)*MP + inv(Wphi))*MP.T*inv(Wn)
                
        # Manage NAN value if any   
        #self.Rt[np.isnan(self.Rt)] = 0
        
            
        
        
#%% CONTROLLER DEFINITION
    def  controller(self,kx,ky,nTh=10):
        """
        """
        
        i  = complex(0,1)
        vx = self.atm.wSpeed*np.cos(self.atm.wDir*np.pi/180)
        vy = self.atm.wSpeed*np.sin(self.atm.wDir*np.pi/180)   
        nPts        = kx.shape[0]
        thetaWind   = np.linspace(0, 2*np.pi-2*np.pi/nTh,nTh)
        costh       = np.cos(thetaWind);
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
            index  = np.hypot(kxExt,kyExt) > self.kc;
            
        kExt       = np.hypot(kxExt[index],kyExt[index])
        psd[index] = self.atm.spectrum(kExt)
            
        
        return psd*p3utils.pistonFilter(self.tel.D,np.hypot(kxExt,kyExt))
    
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
                    PR = p3utils.pistonFilter(self.tel.D,k,fm=mi/d,fn=ni/d)
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
        psd = np.zeros(kx.shape)
        if self.noiseVariance > 0:
            if aoFilter == 'square':
                index  = (abs(kx) <=kc) | (abs(ky) <= kc)               
            elif aoFilter == 'circle':
                index  = np.hypot(kx,ky) <= self.kc
            
            psd[index] = self.noiseVariance/(2*self.kc)**2*(abs(self.Rx[index])**2 + abs(self.Ry[index]**2));
            
        return psd*p3utils.pistonFilter(self.tel.D,np.hypot(kx,ky))*self.noiseGain
    
    def servoLagPSD(self,kx,ky,aoFilter='circle'):
        """ SERVOLAGPSD Servo-lag power spectrum density
        """
            
        kc = self.kc
        psd = np.zeros(kx.shape)
        if aoFilter == 'square':
            index  = (abs(kx) <=kc) | (abs(ky) <= kc)               
        elif aoFilter == 'circle':
            index  = np.hypot(kx,ky) <= self.kc       
                
        F = (self.Rx[index]*self.SxAv[index] + self.Ry[index]*self.SyAv[index])
        Wphi = self.atm.spectrum(np.hypot(kx,ky))
        
        if (self.loopGain == 0):
            psd[index] = abs(1-F)**2*Wphi[index]
        else:
            psd[index] = (1 + abs(F)**2*self.h2[index] - 
               2*np.real(F*self.h1[index]))*Wphi[index]
                    
            
        return psd*p3utils.pistonFilter(self.tel.D,np.hypot(kx,ky))
    
    def anisoServoLagPSD(self,kx,ky,iSrc=0,aoFilter='circle'):
        """%% ANISOSERVOLAGPSD Anisoplanatism + Servo-lag power spectrum density
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
        if sum(sum(self.src.direction))!=0:
            th  = self.src.direction[:,iSrc]
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
                    
                
        return psd*p3utils.pistonFilter(self.tel.D,np.hypot(kx,ky))
    
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
        if sum(sum(self.src.direction))!=0:
            th  = self.src.direction[:,iSrc]
            for l in np.arange(0,self.atm.nL):
                red = 2*np.pi*heights[l]*(kx*th[0] + ky*th[1])
                A   = A + 2*weights[l]*( 1 - np.cos(red) )     
        else:
            A = np.ones(kx.shape)
        
        Wphi       = self.atm.spectrum(np.hypot(kx,ky))   
        psd[index] = A[index]*Wphi[index]
        
        return psd*p3utils.pistonFilter(self.tel.D,np.hypot(kx,ky))
    
    def powerSpectrumDensity(self,kx,ky,iSrc=0,aoFilter='circle'):
        """ POWER SPECTRUM DENSITY AO system power spectrum density
        """
        kc          = self.kc 
        resExt      = kx.shape[0]*self.nTimes 
        kxExt       = 2*self.nTimes*kc*fft.fftshift(fft.fftfreq(resExt))    
        kyExt       = 2*self.nTimes*kc*fft.fftshift(fft.fftfreq(resExt))            
        kxExt,kyExt = np.meshgrid(kxExt,kyExt)        
        psd         = np.zeros((resExt,resExt))
                      
        index  = (abs(kxExt) <= kc) & (abs(kyExt) <= kc) 
        # Sums PSDs
        tmp = self.noisePSD(kx,ky,aoFilter=aoFilter) \
        + self.aliasingPSD(kx,ky,aoFilter=aoFilter) \
        + self.anisoServoLagPSD(kx,ky,iSrc=iSrc,aoFilter=aoFilter)
        
        psd[np.where(index)] = tmp.ravel()        
        return psd + self.fittingPSD(kx,ky,aoFilter=aoFilter)
    
    def errorBreakDown(self,iSrc=0,aoFilter='circle'):
        """
        """
        #self.atm.wvl = self.src.wvl[iSrc]
        # Constants
        wvl    = self.src.wvl[iSrc]
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
        psdAS  = self.anisoServoLagPSD(kx,ky,iSrc=iSrc,aoFilter=aoFilter)
        psdS   = self.servoLagPSD(kx,ky,aoFilter=aoFilter)
        psdAni = self.anisoplanatismPSD(kx,ky,iSrc=iSrc,aoFilter=aoFilter)
        # Derives wavefront error
        wfeFit = np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdFit,kxExt),kxExt)))
        wfeAl  = np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdAl,kx),kx)))
        wfeN   = np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdN,kx),kx)))
        wfeAS  = np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdAS,kx),kx)))
        wfeS   = np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdS,kx),kx)))
        wfeAni = np.mean(rad2nm*np.sqrt(np.trapz(np.trapz(psdAni,kx),kx)))
        wfeTot = np.sqrt(wfeFit**2+wfeAl**2+wfeAS**2+wfeN**2)
        strehl = 100*np.exp(-(wfeTot/rad2nm)**2)
        # Print
        print('\n_____ ERROR BREAKDOWN _____')
        print('------------------------------------------')
        fprintf(sys.stdout,'.Strehl-ratio at %4.2fmicron:\t%4.2f%%\n',wvl*1e6,strehl)
        fprintf(sys.stdout,'.Residual wavefront error:\t%4.2fnm\n',wfeTot)
        fprintf(sys.stdout,'.Fitting error:\t\t\t%4.2fnm\n',wfeFit)
        fprintf(sys.stdout,'.Aliasing error:\t\t%4.2fnm\n',wfeAl)
        fprintf(sys.stdout,'.Noise error:\t\t\t%4.2fnm\n',wfeN)
        fprintf(sys.stdout,'.Aniso+servoLag error:\t\t%4.2fnm\n',wfeAS)
        print('-------------------------------------------')
        fprintf(sys.stdout,'.Sole anisoplanatism error:\t%4.2fnm\n',wfeAni)
        fprintf(sys.stdout,'.Sole servoLag error:\t\t%4.2fnm\n',wfeS)
        print('------------------------------------------')
    
    def getPSF(self,iSrc=0,aoFilter='circle',nyquistSampling=False):
        """
        """
        # Get constants
        psInMas = self.psInMas
        fovInArcsec = self.fovInArcsec
        dk    = 2*self.kc/self.resAO
        wvl   = self.src.wvl[iSrc]
        lonD  = (1e3*180*3600/np.pi*wvl/self.tel.D)
        if nyquistSampling == True:
            nqSmpl = 1
            psInMas= lonD/2
        else:
            nqSmpl= lonD/psInMas/2
            
        fovInPixel = int((np.ceil(2e3*fovInArcsec/psInMas))/2)
        fovInPixel   = max([fovInPixel,2*self.resAO])
        fprintf(sys.stdout,'.Field of view:\t\t%4.2f arcsec\n.Pixel scale:\t\t%4.2f mas\n.Nyquist sampling:\t%4.2f',fovInPixel*psInMas/1e3,psInMas,nqSmpl)
        #self.nTimes = int(np.round(fovInPixel/self.resAO))
        # Get the PSD        
        psd   = self.powerSpectrumDensity(self.kx,self.ky,iSrc=iSrc,aoFilter=aoFilter)        
        psd   = p3utils.enlargeSupport(psd,2)
        otfAO = fft.fftshift(p3utils.psd2otf(psd,dk))
        otfAO = p3utils.interpolateSupport(otfAO,2*self.tel.resolution)
        otfAO = otfAO/otfAO.max()
        # Get the telescope OTF
        otfTel= p3utils.pupil2otf(self.tel.pupil,0*self.tel.pupil,2)
        # Get the total OTF corresponding to a nyquist-sampled PSF
        otfTot= otfAO*otfTel
        
        if nqSmpl == 1:            
            # Interpolate the OTF to set the PSF FOV
            otfTot = p3utils.interpolateSupport(otfTot,fovInPixel)
            psf    = p3utils.otf2psf(otfTot)
        elif nqSmpl >1:
            # Zero-pad the OTF to set the PSF pixel scale
            otfTot = p3utils.enlargeSupport(otfTot,nqSmpl)
            # Interpolate the OTF to set the PSF FOV
            otfTot = p3utils.interpolateSupport(otfTot,fovInPixel)
            psf    = p3utils.otf2psf(otfTot)
        else:
            # Interpolate the OTF at high resolution to set the PSF FOV
            otfTot = p3utils.interpolateSupport(otfTot,int(np.round(fovInPixel/nqSmpl)))
            psf    = p3utils.otf2psf(otfTot)
            # Interpolate the PSF to set the PSF pixel scale
            psf    = p3utils.interpolateSupport(psf,fovInPixel)
            
        return psf,otfTel,otfAO,otfTot
    
    
    
def demo():
    
    tel = telescope(8,0,0.14,240,'vlt_pup_240.fits')
    atm = atmosphere(500e-9,0.16,[0.7,0.2,0.1],[500.,4e3,10e3],L0=25,wSpeed=[5.,10.,15.],wDir=[0.,30,60.])
    src = source(1.64e-6,12,[0],[0],verbose=True)
    fao = spatialFrequency(tel,atm,src,21,1e-4,0.5,1e-3,1e-3,41,10,3)
    psf,otfTel,otfAO,otfTot = fao.getPSF();
    
    plt.figure()
    plt.title('PSF')
    plt.imshow(np.log10(psf))
    
    plt.figure()
    plt.title('Tel')
    plt.imshow(tel.pupil)
    """
    plt.figure()
    plt.title('AO')
    plt.imshow(otfAO)
        
    plt.figure()
    plt.title('Tot')
    plt.imshow(otfTot)
    """
    fao.errorBreakDown()
    
    return fao
    
 #def demo2():
fao = demo()