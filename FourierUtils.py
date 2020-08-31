# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 01:17:43 2020

@author: StBernard
"""
# Libraries
import numpy as np
import scipy as sp
import scipy.special as spc
import matplotlib.pyplot as plt
import numpy.fft as fft

#%%  FOURIER TOOLS

def cov2sf(cov):
    return 2*cov.max() - cov - np.conjugate(cov)

def fftCorrel(x,y):
    nPts = x.shape
    
    if len(nPts) == 1:
        out =  fft.ifft(fft.fft(x)*np.conjugate(fft.fft(y)))/nPts
    elif len(nPts) == 2:
        out =  fft.ifft2(fft.fft2(x)*np.conjugate(fft.fft2(y)))/(nPts[0]*nPts[1])
    return out

def fftsym(x):
    if x.ndim ==2:
        nx,ny            = x.shape
        if np.any(np.iscomplex(x)):
            out              = np.zeros((nx,ny)) + complex(0,1)*np.zeros((nx,ny))
            out[0,0]         = x[0,0]
            out[0,1:ny-1]     = x[1,np.arange(ny-1,1,-1)]
            out[1:nx-1,0]     = x[np.arange(nx-1,1,-1),1]
            out[1:nx-1,1:ny-1] = x[np.arange(nx-1,1,-1),np.arange(ny-1,1,-1)]
        else:
            out              = np.zeros((nx,ny))
            out[0,0]         = x[0,0]
            out[0,1:ny-1]     = x[1,np.arange(ny-1,1,-1)]
            out[1:nx-1,0]     = x[np.arange(nx-1,1,-1),1]
            out[1:nx-1,1:ny-1] = x[np.arange(nx-1,1,-1),np.arange(ny-1,1,-1)]
            
        return out
    elif x.ndim ==1:
        return fft.fftshift(x)
            
def otf2psf(otf):        
    nX,nY   = otf.shape
    u1d     = fft.fftshift(fft.fftfreq(nX))
    v1d     = fft.fftshift(fft.fftfreq(nY))
    u2d,v2d = np.meshgrid(u1d,v1d)
    
    if nX%2 == 0:
        fftPhasor = np.exp(1*complex(0,1)*np.pi*(u2d+v2d))
    else:
        fftPhasor = 1
    
    if nX%2 == 0:
        out = np.real(fft.fftshift(fft.ifft2(fft.fftshift(otf*fftPhasor))))
    else:
        out = np.real(fft.fftshift(fft.ifft2(fft.ifftshift(otf*fftPhasor))))
                
    return out/out.sum()

def otfShannon2psf(otf,nqSmpl,fovInPixel):
    if nqSmpl == 1:            
        # Interpolate the OTF to set the PSF FOV
        otf    = interpolateSupport(otf,fovInPixel)
        psf    = otf2psf(otf)
    elif nqSmpl >1:
        # Zero-pad the OTF to set the PSF pixel scale
        otf    = enlargeSupport(otf,nqSmpl)
        # Interpolate the OTF to set the PSF FOV
        otf    = interpolateSupport(otf,fovInPixel)
        psf    = otf2psf(otf)
    else:
        # Interpolate the OTF at high resolution to set the PSF FOV
        otf    = interpolateSupport(otf,int(np.round(fovInPixel/nqSmpl)))
        psf    = otf2psf(otf)
        # Interpolate the PSF to set the PSF pixel scale
        psf    = interpolateSupport(psf,fovInPixel)
    return psf
                        
def pistonFilter(D,f,fm=0,fn=0):    
    f[np.where(f==0)] = 1e-10 
    if len(f.shape) ==1:
        Fx,Fy = np.meshgrid(f,f)            
        FX    = Fx -fm 
        FY    = Fy -fn    
        F     = np.pi*D*np.hypot(FX,FY)    
    else:
        F     = np.pi*D*f
    return 1-(2*spc.j1(F)/F)**2
             
def psd2cov(psd,pixelScale):
    nPts = np.array(psd.shape)
    psd  = fft.fftshift(psd)
    if len(nPts) ==1:
        out = fft.fft(psd)*pixelScale**2
    elif len(nPts) ==2:
        out = fft.fft2(psd)*pixelScale**2        
    return out

def psd2otf(psd,pixelScale):
    return sf2otf(cov2sf(psd2cov(psd,pixelScale)))

def psd2psf(psd,pixelScale):
    return otf2psf(fft.fftshift(psd2otf(psd,pixelScale)))
 
def psf2otf(psf):
    return fft.fft2(fft.fftshift(psf))/psf.sum()
       
def pupil2otf(pupil,phase,overSampling):   
    P    = enlargeSupport(pupil,overSampling)
    phi  = enlargeSupport(phase,overSampling)
    E    = P*np.exp(1*complex(0,1)*phi)    
    otf  = np.real(fft.fftshift(fftCorrel(E,E)))
    return otf/otf.max()

def pupil2psf(pupil,phase,overSampling):    
    otf = pupil2otf(pupil,phase,overSampling)
    return otf2psf(otf)
      
def sf2otf(sf):
    return np.exp(-0.5 * sf)
                 
def telescopeOtf(pupil,overSampling):    
    extendedPup  = enlargeSupport(pupil,2*overSampling)
    return fft.fftshift(fftCorrel(extendedPup,extendedPup))
           
def telescopePsf(pupil,overSampling,kind='spline'):
    nSize = np.array(pupil.shape)
    
    if overSampling >=1:
        otf = telescopeOtf(pupil,overSampling)
        return otf2psf(interpolateSupport(otf,nSize,kind=kind))
    else:
        otf = interpolateSupport(telescopeOtf(pupil,2),nSize/overSampling,kind=kind)
        return interpolateSupport(otf2psf(otf),nSize,kind=kind)

#%%  IMAGE PROCESSING
        
def cropSupport(im,n):    
    nx,ny = im.shape
    
    if np.isscalar(n) == 1:
        n = np.array([n,n])
    
    nx2     = int(nx/n[0])
    ny2     = int(ny/n[1])
    
    if np.any(np.iscomplex(im)):
        imNew = np.zeros((nx2,ny2)) + complex(0,1)*np.zeros((nx2,ny2))
    else:
        imNew = np.zeros((nx2,ny2))
        
    if nx2%2 ==0:
        xi = int(0.5*(nx-nx2))
        xf = int(0.5*(nx + nx2))
    else:
        xi = int(0.5*(nx-nx2))
        xf = int(0.5*(nx+nx2))
        
    if ny2%2 ==0:
        yi = int(0.5*(ny-ny2))
        yf = int(0.5*(ny + ny2))
    else:
        yi = int(0.5*(ny-ny2))
        yf = int(0.5*(ny+ny2))    
        
    imNew     = im[xi:xf,yi:yf]
    
    return imNew
            
            
def enlargeSupport(im,n):
    # Otf sizes
    nx,ny  = im.shape
    nx2 = int(n*nx)
    ny2 = int(n*ny)
    
    if np.any(np.iscomplex(im)):
        imNew = np.zeros((nx2,ny2)) + complex(0,1)*np.zeros((nx2,ny2))
    else:
        imNew = np.zeros((nx2,ny2))
        
    #Zero-padding    
    if nx2%2 ==0:
        xi = int(0.5*(nx2-nx))
        xf = int(0.5*(nx2 + nx))
    else:
        xi = int(0.5*(nx2-nx))
        xf = int(0.5*(nx2+nx))
        
    if ny2%2 ==0:
        yi = int(0.5*(ny2-ny))
        yf = int(0.5*(ny2 + ny))
    else:
        yi = int(0.5*(ny2-ny))
        yf = int(0.5*(ny2+ny))        
        
            
    imNew[xi:xf,yi:yf] = im
    
    return imNew

def interpolateSupport(otf,nRes,kind='spline'):
    # Define angular frequencies vectors
    nx,ny = otf.shape
    
    if np.isscalar(nRes):
        mx = my = nRes
    else:        
        mx = nRes[0]
        my = nRes[1]
               
    # Initial frequencies grid    
    if nx%2 == 0:
        uinit = np.linspace(-nx/2,nx/2-1,nx)*2/nx
    else:
        uinit = np.linspace(-np.floor(nx/2),np.floor(nx/2),nx)*2/nx
    if ny%2 == 0:
        vinit = np.linspace(-ny/2,ny/2-1,ny)*2/ny
    else:
        vinit = np.linspace(-np.floor(ny/2),np.floor(ny/2),ny)*2/ny    
         
    # Interpolated frequencies grid                  
    if mx%2 == 0:
        unew = np.linspace(-mx/2,mx/2-1,mx)*2/mx
    else:
        unew = np.linspace(-np.floor(mx/2),np.floor(mx/2),mx)*2/mx
    if my%2 == 0:
        vnew = np.linspace(-my/2,my/2-1,my)*2/my
    else:
        vnew = np.linspace(-np.floor(my/2),np.floor(my/2),my)*2/my
               
    # Interpolation
    import scipy.interpolate as interp        

    if kind == 'spline':
        # Surprinsingly v and u vectors must be shifted when using
        # RectBivariateSpline. See:https://github.com/scipy/scipy/issues/3164
        tmpReal = interp.fitpack2.RectBivariateSpline(vinit, uinit, np.real(otf))
        tmpImag = interp.fitpack2.RectBivariateSpline(vinit, uinit, np.imag(otf))
    else:
        tmpReal = interp.interp2d(uinit, vinit, np.real(otf),kind=kind)
        tmpImag = interp.interp2d(uinit, vinit, np.imag(otf),kind=kind)
    
    if np.any(np.iscomplex(otf)):
        return tmpReal(unew,vnew) + complex(0,1)*tmpImag(unew,vnew)
    else:
        return tmpReal(unew,vnew)
            