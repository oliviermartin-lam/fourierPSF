#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 14:21:41 2018

@author: omartin
"""

import numpy as np
import math
import sys

def fprintf(stream, format_spec, *args):
    stream.write(format_spec % args)
    
class source:
    """
    """
    
    # DEPENDANT VARIABLES DEFINITION
    @property
    def direction(self):
        arcsec2rad = np.pi/180/3600
        deg2rad    = np.pi/180
        x          = np.tan(self.zenith*arcsec2rad)*np.cos(self.azimuth*deg2rad)
        y          = np.tan(self.zenith*arcsec2rad)*np.sin(self.azimuth*deg2rad)
        direction  = np.zeros((2,self.nSrc))
        direction[0,:] = x
        direction[1,:] = y
        return direction
            
    @property
    def waveNumber(self):
        return 2*np.pi/self.wvl
        
    # CONSTRUCTOR
    def __init__(self,wvl,magnitude,zenith,azimuth,height=math.inf,nSource=1,verbose=False):
       
        # Vectorizing inputs is required  
        if np.isscalar(wvl):
            wvl= np.array([wvl])
        if np.isscalar(magnitude):
            magnitude= np.array([magnitude])        
        if np.isscalar(zenith):
            zenith = np.array([zenith])
        if np.isscalar(azimuth):
            azimuth= np.array([azimuth])
        if np.isscalar(height):
            height= np.array([height])
        
        
         # PARSING INPUTS
        self.wvl       = wvl        # Wavelength value in meter        
        self.magnitude = magnitude  # Magnitude        
        self.zenith    = zenith     # Zenith angle in arcsec
        self.azimuth   = azimuth    # Azimuth angle in degree
        self.height    = height     # Source height in meter
        self.nSource   = nSource
        self.verbose   = verbose        
        
        
        test= lambda x: len(x) == 1
        if (test(zenith)) & (test(azimuth)):
            self.nSrc    = 1                        
        elif (test(zenith)) & (not test(azimuth)):    
            self.nSrc    = len(azimuth)
            self.zenith  = zenith[0]*np.ones(self.nSrc)            
        elif (not test(zenith)) & (test(azimuth)):   
            self.nSrc    = len(zenith)
            self.azimuth = azimuth[0]*np.ones(self.nSrc)           
        else:
            self.nSrc    = len(zenith)           
       
        # Vectorizes source properties
        test= lambda x: (len(x) != self.nSrc)
        if test(wvl):
            print('Select the first wavelength value out from the given inputs')
            self.wvl = self.wvl[0]*np.ones(self.nSrc)
        if test(magnitude):
            print('Select the first magnitude value out from the given inputs')
            self.magnitude = self.magnitude[0]*np.ones(self.nSrc)
        if test(height):
            print('Select the first height value out from the given inputs')
            self.height = self.height[0]*np.ones(self.nSrc)
    
        # Put into array format
        self.wvl       = np.array(self.wvl)
        self.magnitude = np.array(self.magnitude)
        self.zenith    = np.array(self.zenith)
        self.azimuth   = np.array(self.azimuth)
        self.height    = np.array(self.height)
        
        if self.verbose:        
            self.display()    
        
    def display(self):
        """Display object information: prints information about the source object
        """
       
        print('___ SOURCES', self.nSource,'___')
        print('--------------------------------------------------------------------------')        
        print(' Obj   zen[arcsec] azim[deg]  height[m]  wavelength[micron] magnitude[mag]\n')
        if self.nSrc > 1:
            for kObj in np.arange(0,self.nSrc):
                    fprintf(sys.stdout,' %2d     %5.2f      %6.2f       %g           %5.3f            %5.2f\n',
                            kObj,self.zenith[kObj],self.azimuth[kObj],
                            self.height[kObj],self.wvl[kObj]*1e6,
                            self.magnitude[kObj])
        else:
           fprintf(sys.stdout,' %2d     %5.2f      %6.2f       %g           %5.3f            %5.2f\n',
                   self.nSource,self.zenith[0],self.azimuth[0],self.height[0],self.wvl[0]*1e6,self.magnitude[0]) 
               
        print('--------------------------------------------------------------------------\n')        