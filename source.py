#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 18 14:21:41 2018

@author: omartin
"""

import numpy as np
import math
    
class Attribute(object):
    pass

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
    def __init__(self,wvl,zenith,azimuth,height=math.inf,nSource=1,types="SOURCE",verbose=False):
       
        # Vectorizing inputs is required  
        if np.isscalar(wvl):
            wvl= np.array([wvl])     
        if np.isscalar(zenith):
            zenith = np.array([zenith])
        if np.isscalar(azimuth):
            azimuth= np.array([azimuth])
        if np.isscalar(height):
            height= np.array([height])
        
        
         # PARSING INPUTS
        self.wvl       = wvl        # Wavelength value in meter             
        self.zenith    = zenith     # Zenith angle in arcsec
        self.azimuth   = azimuth    # Azimuth angle in degree
        self.height    = height     # Source height in meter
        self.nSource   = nSource
        self.type = types
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
        test= lambda x: (x.size != self.nSrc)
        if test(wvl):
            print('Vectorize the wavelength value to cope with the number of sources')
            self.wvl = self.wvl[0]*np.ones(self.nSrc)
        if self.height == math.inf:
            self.height = self.height*np.ones(self.nSrc)
    
        # Put into array format
        self.wvl       = np.array(self.wvl)
        self.zenith    = np.array(self.zenith)
        self.azimuth   = np.array(self.azimuth)
        self.height    = np.array(self.height)
        
        if self.verbose:        
            self
        
    def __repr__(self):
        """Display object information: prints information about the source object
        """
       
        s = "___ " + self.type + "___\n"
        s = s + "--------------------------------------------------------------------------\n"  
        s = s +" Obj zen[arcsec] azim[deg] height[m] wavelength[micron]\n"
        for kObj in np.arange(0,self.nSrc):
            s = s + " {:d}\t {:.2f}\t      {:.2f}\t\t {:g}\t\t   {:.3f}\n".format(kObj,self.zenith[kObj],self.azimuth[kObj],
                            self.height[kObj],self.wvl[kObj]*1e6)
        s = s + "--------------------------------------------------------------------------\n"
        
        return s