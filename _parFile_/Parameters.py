#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 27 15:42:11 2020

@author: omartin
"""

#%% TRUE ATMOSPHERE #%%
r0      = 0.15  # m
L0      = 25    #m
wvlAtm = 500   #nm
weights =  [0.41 , 0.16 , 0.1 , 0.09 , 0.08 , 0.05 , 0.045 , 0.035 , 0.02 , 0.01]
heights = [0 , 200 , 600 , 1200 , 3000 , 4700 , 7300 , 8500 , 9700 , 11000] #m
wSpeed  = [5., 5. , 4.5 , 15.6 , 25. , 10.2 , 8.3 , 20 , 10 , 20] #m/s
wDir    = [0 , 180 , 120 , -60 , 5 , -85 , 85 , 180 , 0 , 180] # degrees

#%% MODEL ATMOSPHERE #%%
r0_mod      = 0.15  # m
L0_mod      = 25    #m
weights_mod =  [0.41 , 0.16 , 0.1 , 0.09 , 0.08 , 0.05 , 0.045 , 0.035 , 0.02 , 0.01]
heights_mod = [0 , 200 , 600 , 1200 , 3000 , 4700 , 7300 , 8500 , 9700 , 11000] #m
wSpeed_mod  = [5., 5. , 4.5 , 15.6 , 25. , 10.2 , 8.3 , 20 , 10 , 20] #m/s
wDir_mod    = [0 , 180 , 120 , -60 , 5 , -85 , 85 , 180 , 0 , 180] # degrees

#%% TELESCOPE #%%
D           = 8 #m
zenith_angle= 0 #degrees
obsRatio    = 0.14 #percent
resolution  = 240 #number of pixels
path_pupil  = "_calib_/vlt_pup_240.fits" # [fichier.fits]


#%% AO #%%
nActuator       = 21
noiseVariance   =  0.1 #rad^2
loopGain        = 0.5  
samplingTime    = 1 #ms
latency         = 2 #ms
resAO           = 41 #AO correction area in pixels
psInMas         = 10 # PSF pixel scale in mas 
fovInArcsec     = 3 # PSF fov in arcsec

#%% CORRECTION OPTIMIZATION
h_dm    = [0 , 4000 , 9000] #DM altitude in km
theta_x = [ 10. , -10.0 , 0 , 0 , 0, 0, 5, -5,0]  # optimization direction in x-axis
theta_y = [ 0.0 , 0.0 , 10 , -10 , 5, -5, 0, 0,0] # optimization direction in y-axis
theta_w = [ 0.5 , 0.5 , 0.5 , 0.5 , 0.1, 0.1, 0.1, 0.1, 1] # weight
condmax = 1000000 ; #matrix conditionning for Popt calculation

#%% SCIENTIFIC SOURCE #%%
wvlSrc     =  [1650 , 1650] #[nm]
zenithSrc  = [0 , 60] #[arcseconds]
azimuthSrc = [0 , 0] #[arcseconds]

#%% GUIDE STAR 1-8 #%%
wvlGs     = [500 , 500 , 500 , 500 , 500 , 500 , 500 , 500 ] #nm
zenithGs  = [60 , 60 , 60 , 60 , 60 , 60 , 60 , 60] #arcseconds
azimuthGs = [0 , 45 , 90 , 135 , 180 , 225 , 270 , 315] #degrees
heightGs  = 0 #[m (0 if infinite)]