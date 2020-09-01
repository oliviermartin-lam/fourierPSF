#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:42:11 2020

@author: omartin
"""
import numpy as np



#%% TELESCOPE #%%
D           = 8                                                                 # Telescope diameter  in m
zenith_angle= 30                                                                # Telescope zenith angle in degrees
obsRatio    = 0.14                                                              # Central obstruction ratio
resolution  = 240                                                               # Pupil resolution in pixels
path_pupil  = "_calib_/vlt_pup_240.fits"                                        # .fits file for the telescope pupil

#%% TRUE ATMOSPHERE AT ZENITH#%%
wvlAtm      = 500                                                               # Atmosphere wavelength in nm
seeing      = 0.8                                                               # Seeing in arcsec
r0          = 0.976*wvlAtm*1e-9/seeing*206264.8                                 # Fried's parameter  in m
L0          = 25                                                                # Outer scale in m
weights     = [0.59, 0.02, 0.04, 0.06, 0.01, 0.05, 0.09, 0.04, 0.05, 0.05]      # Fractional weights of layers
heights     = [30, 140, 281, 562, 1125, 2250, 4500, 7750, 11000, 14000]         # Layers' altitude in m
wSpeed      = [6.6, 5.9, 5.1, 4.5, 5.1, 8.3, 16.3, 30.2, 34.3, 17.5]            # Wind speed in m/s
wDir        = [0., 0., 0., 0., 90., -90., -90., 90., 0., 0.]                    # WInd direction in degrees

#%% MODEL ATMOSPHERE AT ZENITH#%%
r0_mod      = 0.15  # m
L0_mod      = 25    #m
weights_mod = [0.41 , 0.16 , 0.1 , 0.09 , 0.08 , 0.05 , 0.045 , 0.035 , 0.02 , 0.01]
heights_mod = [0 , 200 , 600 , 1200 , 3000 , 4700 , 7300 , 8500 , 9700 , 11000] #m
wSpeed_mod  = [5., 5. , 4.5 , 15.6 , 25. , 10.2 , 8.3 , 20 , 10 , 20] #m/s
wDir_mod    = [0 , 180 , 120 , -60 , 5 , -85 , 85 , 180 , 0 , 180] # degrees

#%% PSF EVALUATION DIRECTIONS #%%
nSrc        = 9                                                                 # number of PSF evaluation directions    
wvlSrc      = 640*np.ones(nSrc)                                                 # Imaging wavelength [nm]
x           = [0 , 0 , 0 , 8.8 , 8.8 , 8.8 , 17.5 , 17.5 , 17.5]                # X-axis positions
y           = [0 , 8.8 , 17.5 , 0 , 8.8 , 17.5 , 0 , 8.8 , 17.5]                # Y-axis positions
zenithSrc   = np.hypot(x,y)
azimuthSrc  = np.arctan2(y, x)*180/np.pi #[arcseconds]
psInMas     = 7.4                                                               # PSF pixel scale in mas
fovInPix    = 400                                                               # PSF fov in pixels
fovInArcsec = psInMas * 1e-3 * fovInPix                                         # PSF fov in arcsec

#%% GUIDE STAR #%%
nGs         = 8                                                                 # Number of guide stars
wvlGs       = 589 * np.ones(nGs)                                                # Sensing wavelength in nm
Rast        = 17.5                                                              # Constellation radius in arcsec 
zenithGs    = Rast * np.ones(nGs)                                               
azimuthGs   = [0 , 45 , 90 , 135 , 180 , 225 , 270 , 315]                       
heightGs    = 90e3                                                              # Guide stars height in m [(0 if infinite)]

#%% AO #%%
pitchs_dm   = [0.22, 0.22, 0.35]                                                # DM actuators pitchs in m             
h_dm        = [0 , 4000 , 14000]                                                # DM altitude in m
nLenslet    = 40                                                                # number of WFS lenslets
pitchs_wfs  = D/nLenslet * np.ones(nGs)
loopGain    = 0.5                                                               # Loop gain
samplingTime= 1                                                                 # RTC sampling time in ms
latency     = 1                                                                 # AO loop latency in ms
resAO       = 41                                                                # AO correction area in pixels

# NOISE VARIANCE
nph         = 75*np.ones(nGs)                                                   # number of photon/frame/sub-aperture 
NS          = 6                                                                 # number of pixels for centroiding
sa_fov      = 5                                                                 # fov subaperture 6x6pix [arcsec]
pix_dim     = sa_fov/NS * 1e3                                                 # pixel dimension [mas]
sig_ron     = 0.2                                                               # read ou noise in e-
ND          = wvlGs[0]*1e-9/pitchs_wfs*2.06265e8/pix_dim                        # spot FWHM in pixels and without turbulence
r0_wfs      = r0*(wvlGs[0]/wvlAtm)**1.2                                         # [m]
NT          = wvlGs[0]*1e-9/r0_wfs*2.06265e8/pix_dim                            #spot FWHM in pixels and with turbulence
varNoise    = (np.pi**2/3) * ((sig_ron/nph)**2) * (NS**2/ND)**2 + np.pi**2*(NT/ND)**2/(2*nph)

# CORRECTION OPTIMIZATION
zenithOpt   = [0 , 15 , 15 , 15 , 15 , 15 , 15 , 15 , 15, 60 , 60 , 60 , 60 , 60 , 60 , 60 , 60] # Zenith position in arcsec
azimuthOpt  = [0 , 0 , 45 , 90 , 135 , 180 , 225 , 270 , 315 , 0 , 45 , 90 , 135 , 180 , 225 , 270 , 315] # Azimuth in degrees
weightOpt   = [4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1]   # Weights
condmax     = 9e3 ;                                                             # matrix conditionning for Popt calculation

