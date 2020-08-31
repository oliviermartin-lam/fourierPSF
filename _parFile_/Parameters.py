#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:42:11 2020

@author: omartin
"""
import numpy as np



#%% TELESCOPE #%%
D           = 8 #m
zenith_angle= 30 #degrees
obsRatio    = 0.14 #percent
resolution  = 240 #number of pixels
path_pupil  = "_calib_/vlt_pup_240.fits" # [fichier.fits]

#%% TRUE ATMOSPHERE AT ZENITH#%%
wvlAtm      = 500                                                               # nm
seeing      = 0.6                                                               # arcsec
r0          = 0.976*wvlAtm*1e-9/seeing*206264.8                                 # m
L0          = 25                                                                # m
weights     = [0.59, 0.02, 0.04, 0.06, 0.01, 0.05, 0.09, 0.04, 0.05, 0.05]
heights     = [30, 140, 281, 562, 1125, 2250, 4500, 7750, 11000, 14000]         # m
wSpeed      = [6.6, 5.9, 5.1, 4.5, 5.1, 8.3, 16.3, 30.2, 34.3, 17.5]            # m/s
wDir        = [0., 0., 0., 0., 90., -90., -90., 90., 0., 0.]                    # degrees

#%% MODEL ATMOSPHERE AT ZENITH#%%
r0_mod      = 0.15  # m
L0_mod      = 25    #m
weights_mod = [0.41 , 0.16 , 0.1 , 0.09 , 0.08 , 0.05 , 0.045 , 0.035 , 0.02 , 0.01]
heights_mod = [0 , 200 , 600 , 1200 , 3000 , 4700 , 7300 , 8500 , 9700 , 11000] #m
wSpeed_mod  = [5., 5. , 4.5 , 15.6 , 25. , 10.2 , 8.3 , 20 , 10 , 20] #m/s
wDir_mod    = [0 , 180 , 120 , -60 , 5 , -85 , 85 , 180 , 0 , 180] # degrees

#%% SCIENTIFIC SOURCE #%%
nSrc        = 9
wvlSrc      = 640*np.ones(nSrc) #[nm]
zenithSrc   = [0 , 8.8 , 17.5 , 8.8 , 12.4 , 19.6 , 17.5 , 19.6 , 24.7 ]
azimuthSrc  = [0 , 0 , 0 , 90 , 45 , 26.6 , 90 , 63.4 , 45] #[arcseconds]

#%% GUIDE STAR #%%
nGs         = 8
wvlGs       = 589 * np.ones(nGs)                                                #nm
Rast        = 17.5                                                              #arcsec 
zenithGs    = Rast * np.ones(nGs)                                               #arcseconds
azimuthGs   = np.linspace(0,360 - 360/nGs,num=nGs)                              #degrees
heightGs    = 90e3                                                                 #[m (0 if infinite)]

#%% AO #%%
nActuator   = 41
nLenslet    = 40
loopGain    = 0.5  
samplingTime= 1 #ms
latency     = 1 #ms
resAO       = 41 #AO correction area in pixels
psInMas     = 7.4 # PSF pixel scale in mas
fovInPix    = 400 
fovInArcsec = psInMas * 1e-3 * fovInPix # PSF fov in arcsec
pitchs_dm   = [0.22, 0.22, 0.35] 
h_dm        = [0 , 4000 , 14000]                                                 # DM altitude in km

#%% Noise variance
pitchs_wfs  = D/nLenslet * np.ones(nGs)
nph         = 75*np.ones(nGs)                                                   #ph/ms/sa ;number of photon/frame/sub-aperture
rad2mas     = 2.06265e8
nPix        = 6
sa_fov      = 5                                                                 #fov subaperture 6x6pix [arcsec]
pix_dim     = sa_fov/nPix * 1e3                                                   #pixel dimension [mas]
sig_ron     = 0.2                                                               #read ou noise in e-
NS          = nPix                                                              #number of pixels used for the centroiding
ND          = wvlGs[0]*1e-9/pitchs_wfs*rad2mas/pix_dim                          #spot FWHM in pixels and without turbulence
r0_wfs      = r0*(wvlGs[0]/wvlAtm)**1.2                                      # [m]
NT          = wvlGs[0]*1e-9/r0_wfs*rad2mas/pix_dim                                   #spot FWHM in pixels and with turbulence
varNoise    = (np.pi**2/3) * ((sig_ron/nph)**2) * (NS**2/ND)**2 + np.pi**2*(NT/ND)**2/(2*nph)

#%% CORRECTION OPTIMIZATION
zenithOpt   = [0 , 15 , 15 , 15 , 15 , 15 , 15 , 15 , 60 , 60 , 60 , 60 , 60 , 60 , 60 , 60]
azimuthOpt  = [0 , 0 , 45 , 90 , 180 , 225 , 270 , 315 , 0 , 45 , 90 , 135 , 180 , 225 , 270 , 315]
weightOpt   = [4 , 4 , 4 , 4 , 4 , 4 , 4 , 4 , 1 , 1 , 1 , 1 , 1 , 1 , 1 , 1]
condmax     = 1e6 ;                                                             # matrix conditionning for Popt calculation

