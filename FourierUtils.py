# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 01:17:43 2020

@author: StBernard
"""
# Libraries
import numpy as np
import scipy.special as spc
import numpy.fft as fft
import matplotlib.pyplot as plt

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
            


#%%  IMAGE PROCESSING TOOLS

def addNoise(im,ron,darkBg,skyBg,DIT,nDIT):
    Texp      = nDIT*DIT
    nPts      = im.shape
    im[im<0]  = 0
    noise_det = np.sqrt(nDIT*ron**2 + darkBg*Texp)*np.random.randn(nPts[0],nPts[1])
    noise_ph  = np.random.poisson(im + skyBg*Texp) - skyBg*Texp
    return im + noise_det + noise_ph

def centerPsf(psf,rebin,nargout=1):
    flux        = psf.sum()
    npsfx,npsfy = psf.shape
    npsfx2      = npsfx*rebin
    npsfy2      = npsfy*rebin
            
    # Get the high-resolution PSF
    if rebin > 1:
        psf_hr = interpolateSupport(psf,npsfx2)
    else:
        psf_hr = psf
            
    # Get the max value
    idx,idy = np.unravel_index(psf_hr.argmax(), psf_hr.shape)
    dx      = npsfx2/2-idx
    dy      = npsfy2/2-idy
    # Get the OTF
    otf_hr  = fft.fftshift(psf2otf(psf_hr))
    # Apply the Phasor
    u       = fft.fftshift(fft.fftfreq(otf_hr.shape[0]))
    u,v     = np.meshgrid(u,u)
    fftPhasor = np.exp(-1*complex(0,1)*np.pi*(u*dy+v*dx))
    otf_hr    = otf_hr*fftPhasor
    # Get the PSF low-resolution
    imCor  = otf2psf(otf_hr)
    imCor  = interpolateSupport(imCor,npsfx)
    imCor  = flux*imCor/imCor.sum()
    otf_lr = fft.fftshift(psf2otf(imCor))
    
    if nargout == 1:
        return imCor
    else:
        return imCor,otf_lr

def correctFromDeadPixels(im,badPixFrame):
    # Correcting the bad pixels on the matrix im from the bad pixel frame
            
    npixd = badPixFrame.shape[1]
    imCor = im
    for i in np.arange(0,npixd,1):
        w =  np.sum(badPixFrame[i,2:npixd-1,1])
        if w!=0:
            imCor[badPixFrame[i,0,0]] = np.sum(im[badPixFrame[i,2:npixd,0]]*badPixFrame[i,2:npixd-1,1]) / w;
                            
    return imCor
        
def createDeadPixFrame(badPixelMap):
    # dpframe = createDeadPixFrame(badPixelMap)
    # badPixelMap is the map of dead pixels
    #frame  is the image to be corrected
            
    #The dead pixel is replaced by a weighted average of the neighbours,
    #1 2 1
    #2 X 2
    #1 2 1
    #when they are "available". "Available" means that the sum of the
    #weights of neighbouring pixels must exceeds 4.
            
    #If no neighbouring pixel is available, the dead pixel is not
    #corrected, but a new "dead pixel map" is created, and the function is
    #called once again (recursive calls).
            
    #Get the number of dead pixels
    sx,sy       = badPixelMap.shape
    npixnoncorr = 0
    nnx,nny     = np.where(badPixelMap)
    nn1D        = np.where(badPixelMap[:])
    nDeadPix    = len(nn1D)
    #Instantiation
    tmp          = badPixelMap*0
    frame        = np.zeros(nDeadPix,10,2) #2nd row: #pixel (one pixel + 8 neighbors)
    frame[:,:,0] = 1                    #3rd row: adresses
            
    #loop on Pixel
    for i in np.arange(0,nDeadPix,1):
        nb = 2
        frame[i,0,0] = nn1D[i]  # 1st row = bad pixel
        frame[i,1,0] = 0        # number of used neighbour pixel for correction
        x            = nnx[i]
        y            = nny[i]
        wcum         = 0
                
        # Edges neighbours
        if x>0 and x<=sx and y+1>0 and y+1<=sy:
            if badPixelMap[x,y+1] == 0:
                nb            = nb+1
                frame[i,nb,0] = nn1D[i] + sx
                frame[i,nb,1] = 2
                wcum          = wcum + 2
                    
            
                
        if x>0 and x<=sx and y-1>0 and y-1<=sy:
            if badPixelMap[x,y-1] ==0:
                nb            = nb+1
                frame[i,nb,0] = nn1D[i]-sx
                frame[i,nb,1] = 2
                wcum          = wcum + 2
                    
                
        if x+1>0 and x+1<=sx and y>0 and y<=sy:
            if badPixelMap[x+1,y] ==0:
                nb            = nb+1
                frame[i,nb,0] = nn1D[i]+1
                frame[i,nb,1] = 2
                wcum          = wcum + 2
                    
        if x-1>0 and x-1<=sx and y>0 and y<=sy:
            if badPixelMap[x-1,y] ==0:
                nb            = nb+1
                frame[i,nb,0] = nn1D[i]-1
                frame[i,nb,1] = 2
                wcum          = wcum + 2
                                            
        #Diagonal neighbours
        if x+1>0 and x+1<=sx and y+1>0 and y+1<=sy:
            if badPixelMap(x+1,y+1) == 0:
                nb            = nb+1
                frame[i,nb,0] = nn1D[i]+1+sx
                frame[i,nb,1] = 1
                wcum          = wcum + 1
                    
                
        if x-1>0 and x-1<=sx and y+1>0 and y+1<=sy:
            if badPixelMap[x-1,y+1] == 0:
                nb            = nb+1
                frame[i,nb,0] = nn1D[i]-1+sx
                frame[i,nb,1] = 1
                wcum          = wcum + 1
                    
                
        if x+1>0 and x+1<=sx and y-1>0 and y-1<=sy:
            if badPixelMap[x+1,y-1]==0:
                nb            = nb+1
                frame[i,nb,0] = nn1D[i]+1-sx
                frame[i,nb,1] = 1
                wcum          = wcum + 1
                
                
        if x-1>0 and x-1<=sx and y-1>0 and y-1<=sy:
            if badPixelMap[x-1,y-1] ==0:
                nb            = nb+1
                frame[i,nb,0] = nn1D[i]-1-sx
                frame[i,nb,1] = 1
                wcum          = wcum + 1
                                                
        # Take decision regarding the number of avalaible neighbours
        if wcum<4:   #not enough neigbours
            npixnoncorr          = npixnoncorr + 1
            tmp[x,y]             = tmp(x,y) + 1
            frame[i,2:9,0]       = 1    # pixel adresses set to 1
            frame[i,:,1]         = 0    # weights set to 0
        else:
            frame[i,1,0]         = nb    #number of correcting pixels
                                        
    if npixnoncorr != 0:
        frame_suppl = createDeadPixFrame(tmp)
        nSup        = frame_suppl.shape[0]
        #Frame concatenation
        N                             = nDeadPix+nSup
        new_frame                     = np.zeros(N,10,2);
        new_frame[0:nDeadPix,:,:]     = frame
        new_frame[0+nDeadPix:N-1,:,:] = frame_suppl
        frame                         = new_frame
    
    return new_frame

def getEnsquaredEnergy(psf):            
    
    S     = psf.sum()
    nY,nX = psf.shape
    y0,x0 = np.unravel_index(psf.argmax(), psf.shape)
    nEE   = min([nY-y0,nX-x0])
    
    EE = np.zeros(nEE+1)
    for n in range(nEE+1):
        EE[n] = psf[y0 - n:y0+n+1,x0-n:x0+n+1].sum()/S
    return EE
            
def getFlux(psf,nargout=1):
    #Define the inner circle
    nx,ny    = psf.shape
    x        = np.linspace(-1,1,nx)
    y        = np.linspace(-1,1,ny)
    X,Y      = np.meshgrid(x,y)
    r        = np.hypot(X,Y)
    msk      = r>1
    #Computing the residual background
    psfNoise = psf*msk
    bg       = np.median(psfNoise)
    #Computing the read-out noise
    ron      = psfNoise.std()
    #Computing the normalized flux
    Flux     = np.sum(psf -bg)
    
    if nargout == 1:
        return Flux
    elif nargout == 2:
        return Flux,ron
    elif nargout == 3:
        return Flux,ron,bg

def getMSE(xtrue,xest,nbox=0,norm='L2'):
    if nbox != 0:
        n   = np.array(xtrue.shape)
        xest = cropSupport(xest,n/nbox)
        xtrue= cropSupport(xtrue,n/nbox)
        
    if norm == 'L2':
        return 1e2*np.sqrt(np.sum((xest-xtrue)**2))/xtrue.sum()
    elif norm == 'L1':
        return 1e2*np.sum(abs(xest-xtrue))/xtrue.sum()
    else:
        print('The input norm={:s} is not recognized, choose L1 or L2'.format(norm))
        return []

def getFWHM(psf,pixelScale,rebin=4,method='contour',nargout=2):
            
    # Gaussian and Moffat fitting are not really efficient on
    # anisoplanatic PSF. Prefer the coutour function in such a
    # case. The cutting method is not compliant to PSF not oriented
    #along x or y-axis.
            
           
    #Interpolation            
    Nx,Ny = psf.shape
    if rebin > 1:
        im_hr = interpolateSupport(psf,rebin*np.array([Nx,Ny]))
    else:
        im_hr = psf
        
    if method == 'cutting':
        # Brutal approach when the PSF is centered and aligned x-axis FWHM
        imy     = im_hr[:,int(Ny*rebin/2)]        
        w       = np.where(imy >= imy.max()/2)[0]
        FWHMy   = pixelScale*(w.max() - w.min())/rebin
        #y-axis FWHM
        imx     = im_hr[int(Nx*rebin/2),:]
        w       = np.where(imx >= imx.max()/2)[0]
        FWHMx   = (w.max() - w.min())/rebin*pixelScale
        theta   = 0
    elif method == 'contour':
        # Contour approach~: something wrong about the ellipse orientation
        fig     = plt.figure()
        C       = plt.contour(im_hr,levels=[im_hr.max()/2])
        plt.close(fig)
        C       = C.collections[0].get_paths()[0]
        C       = C.vertices
        xC      = C[:,0]
        yC      = C[:,1]
        # centering the ellispe
        mx      = np.array([xC.max(),yC.max()])
        mn      = np.array([xC.min(),yC.min()])
        cent    = (mx+mn)/2
        wx      = xC - cent[0]
        wy      = yC - cent[1] 
        # Get the module
        wr      = np.hypot(wx,wy)/rebin*pixelScale                
        # Getting the FWHM
        FWHMx   = 2*wr.max()
        FWHMy   = 2*wr.min()
        #Getting the ellipse orientation
        xm      = wx[wr.argmax()]
        ym      = wy[wr.argmax()]
        theta   = np.mean(180*np.arctan(ym/xm)/np.pi)
        
        #Angle are counted positively in the reverse clockwise direction.                                 
        
    # Get Ellipticity
    aRatio      = np.max([FWHMx/FWHMy,FWHMy/FWHMx])
    
    if nargout == 1:
        return np.hypot(FWHMx,FWHMy)
    elif nargout == 2:
        return FWHMx,FWHMy
    elif nargout == 3:
        return FWHMx,FWHMy,aRatio
    elif nargout == 4:
        return FWHMx,FWHMy,aRatio,theta
                          
def getStrehl(psf0,pupil,overSampling):    
    psf     = centerPsf(psf0,2)
    #% Get the OTF
    otf     = abs(fft.fftshift(psf2otf(psf)))
    otf     = otf/otf.max()
    notf    = np.array(otf.shape)
    # Get the Diffraction-limit OTF
    otfDL   = abs(telescopeOtf(pupil,overSampling))
    otfDL   = interpolateSupport(otfDL,notf)
    otfDL   = otfDL/otfDL.max()
    # Get the Strehl
    return np.trapz(np.trapz(otf))/np.trapz(np.trapz(otfDL))

#%% Data treatment
    
def eqLayers(Cn2, altitudes, nEqLayers, power=5/3):
    '''
             Cn2         ::  The input Cn2 profile (vector)
             altitudes   ::  The input altitudes (vector)
             nEqLayers   ::  The number of output equivalent layers (scalar)
             power       ::  the exponent of the turbulence (default 5/3)
             
             See: Saxenhuber17: Comparison of methods for the reduction of
             reconstructed layers in atmospheric tomography, App Op, Vol. 56, No. 10 / April 1 2017
    '''
    nCn2        = len(Cn2)
    nAltitudes  = len(altitudes)
    nSlab       = np.floor(np.round(nCn2)/np.fix(nEqLayers))
             
    posSlab =  np.round((np.linspace(0, nEqLayers-1, num=nEqLayers))*nSlab)
    for iii in range(nEqLayers-1):
        if posSlab[iii] >= posSlab[iii+1]:
            posSlab[iii+1] = posSlab[iii]+1
                              
    posSlab = np.concatenate((posSlab, [nAltitudes]))
    posSlab = posSlab.astype('b')
    Cn2eq = np.zeros(nEqLayers)
    altEq = np.zeros(nEqLayers)
    
    for ii in range(nEqLayers):
        Cn2eq[ii] = sum(Cn2[posSlab[ii]:posSlab[ii+1]])
        altEq[ii] = (sum(altitudes[posSlab[ii]:posSlab[ii+1]]**(power) * Cn2[posSlab[ii]:posSlab[ii+1]])/Cn2eq[ii])**(1/power)
       
    return Cn2eq,altEq


#%% Analytical models and fitting facilities
def gaussian(x,xdata):                     
    # ------- Grabbing parameters ---------%
    I0 = x[0]          #Amplitude
    ax = x[1]          #x spreading
    ay = x[2]          #y-spreading
    th = x[3]*np.pi/180  #rotation
    x0 = x[4]          #x-shift
    y0 = x[5]          #y-shift
            
    # ------- Including shifts ---------
    X     = xdata[0]
    Y     = xdata[1]
    #Shifts
    X     = X - x0
    Y     = Y - y0
    #Rotation
    Xr    = X*np.cos(th) + Y*np.sin(th)
    Yr    = Y*np.cos(th) - X*np.sin(th)
    # Gaussian expression
    return I0*np.exp(-0.5*((Xr/ax)**2 + (Yr/ay)**2) )