#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 13:38:45 2020

@author: omartin
"""

#%% Import librairies
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from astropy.io import fits
from configparser import ConfigParser
from fourierModel import fourierModel

# Define array on desired parameters
dm_pitch = np.linspace(0.3,0.6,num=7)
nP  = dm_pitch.size

# Define parFiles path
path = '/home/omartin/Projects/fourierPSF/parFile/'

# Load the parFile
parfile = path + 'harmoniParams_dev.ini'
parser = ConfigParser()
parser.optionxform = str
parser.read(parfile)
        
#%% chnage the parameters
wfe = np.zeros(nP)
for k in range(nP):
    print('Doing case {:d} over {:d}\n'.format(k+1,nP))
    # update the parameter in the .ini file
    parser.set('DM','DmPitchs',str([dm_pitch[k]]))
    with open(parfile, 'w') as configfile:
        parser.write(configfile)
    # Instantiating the Fourier model object
    # Note : calcPSF =false -> only instantiation of matrices
    fao = fourierModel(parfile,calcPSF=False,verbose=False,display=False,getErrorBreakDown=False)
    # Get the PSD in nm^2 (accounts for the pixel scale already)
    PSD = fao.powerSpectrumDensity()
    # wavefront error
    wfe[k] = np.sqrt(PSD.sum())

#%% Save results
fits.writeto('wfe.fits',wfe,overwrite=True) 
   
#%% PLot
    plt.close('all')
# to have nice latex fonts
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Palatino"],
})
mpl.rcParams['font.size'] = 18

plt.figure(figsize=(10,10))
plt.plot(dm_pitch,wfe,'bs--',label='Total error')
plt.plot(dm_pitch,np.sqrt(0.23*(dm_pitch/fao.atm.r0)**(5/3))*fao.wvlSrc[0]*1e9/2/np.pi,'r-',label='Pure fitting error')
plt.ylabel('Wavefront error (nm)')
plt.xlabel('DM actuators pitch (m)')
plt.legend()
