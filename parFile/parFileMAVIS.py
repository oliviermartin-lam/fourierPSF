#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:42:11 2020

@author: omartin
"""
import numpy as np
import FourierUtils


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
r0_mod      = r0  # m
L0_mod      = L0    #m
weights_mod = weights
heights_mod = heights
wSpeed_mod  = wSpeed
wDir_mod    = wDir

nL_mod = 10
if nL_mod != len(weights):
    weights_mod,heights_mod = FourierUtils.eqLayers(np.array(weights),np.array(heights),nL_mod)
    wSpeed_mod = np.linspace(min(wSpeed),max(wSpeed),num=nL_mod)
    wDir_mod   = np.linspace(min(wDir),max(wDir),num=nL_mod)


#%% PSF EVALUATION DIRECTIONS #%%
nSrc        = 9                                                                 # number of PSF evaluation directions    
wvlSrc      = 640*np.ones(nSrc)                                                 # Imaging wavelength [nm]
zenithSrc   = np.linspace(0,25,num=nSrc)
azimuthSrc  = 45*np.ones(nSrc) #[arcseconds]
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
resAO       = 2*nLenslet+1                                                      # AO correction area in pixels in the PSD domain

# NOISE VARIANCE
varNoise    = np.ones(nGs)

# CORRECTION OPTIMIZATION
zenithOpt   = [0 , 15 , 15 , 15 , 15 , 15 , 15 , 15 , 15, 60 , 60 , 60 , 60 , 60 , 60 , 60 , 60] # Zenith position in arcsec
azimuthOpt  = [0 , 0 , 45 , 90 , 135 , 180 , 225 , 270 , 315 , 0 , 45 , 90 , 135 , 180 , 225 , 270 , 315] # Azimuth in degrees
weightOpt   = [4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1]   # Weights
condmax     = 9e3 ;                                                             # matrix conditionning for Popt calculation

