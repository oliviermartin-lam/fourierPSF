#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:34:02 2018

@author: omartin
"""
import numpy as np
from astropy.io import fits

import sys

def fprintf(stream, format_spec, *args):
    stream.write(format_spec % args)
    
class telescope:
    """ Telescope class that defines the telescope characteristics.
    Inputs are:
        - D: telescope diameter
        - elevation: telescope elevation in degree
        - obsRatio: central obstruction ratio from 0 to 1
        - resolution: pupil pixels resolution
    """
    
    # DEPENDANT VARIABLES DEFINITION   
    def get_pupil(self):
        return self.p_pupil
                 
    def set_pupil(self,val):
        self.p_pupil = val
        self.resolution = val.shape[0]
        
    pupil = property(get_pupil,set_pupil)        
        
    @property
    def R(self):
        """Telescope radius in meter"""
        return self.D/2
    @property
    def area(self):
        """Telescope area in meter square"""
        return np.pi*self.R**2*(1-self.obsRatio**2)
    @property
    def pupilLogical(self):
        """Telescope area in meter square"""
        return self.pupil.astype(bool)
    
    @property
    def airmass(self):
        return 1/np.cos(self.zenith_angle*np.pi/180)
    
    # CONSTRUCTOR
    def __init__(self,D,zenith_angle,obsRatio,resolution,file = [],verbose=True):
        
        # PARSING INPUTS
        self.D         = D          # in meter
        self.zenith_angle = zenith_angle  # in degree
        self.obsRatio  = obsRatio   # Ranges from 0 to 1
        self.resolution= resolution # In pixels
        self.verbose   = verbose
        self.file      = file
        # PUPIL DEFINITION
        if file != []:
            obj = fits.open(file)
            im = obj[0].data
            hdr = obj[0].header
            self.pupil = im
            obj.close()
            if hdr[3]!=resolution:
                resize = True
            else:
                resize = False
        
        elif (file == [] or obj == []):
            x   = np.linspace(-D/2,D/2,resolution)
            X,Y = np.meshgrid(x,x)
            R   = np.hypot(X,Y)
            P   = (R <= self.R) * (R > self.R*self.obsRatio)
            self.pupil = P
    
        if self.verbose:
            self.display()
        
    def display(self):
        """DISPLAY Display object information        
        display(obj) prints information about the atmosphere+telescope object
        """
                    
        print('___TELESCOPE___')
        print('----------------------------------------')
        if self.file != []:
            fprintf(sys.stdout,'. Ouverture du fichier:\t%s\n',self.file)
        if self.obsRatio==0:
            fprintf(sys.stdout,'. Aperture diameter:\t%4.2fm',self.D)
        else:
            fprintf(sys.stdout,'. Aperture diameter:\t%4.2fm\n. Central obstruction:\t%4.2f%%\n',
                self.D,self.obsRatio*100)
            
        fprintf(sys.stdout,'. Collecting area:\t%5.2fm^2\n',self.area)
        
        if np.isscalar(self.resolution):
            fprintf(sys.stdout,'. Pupil resolution:\t%dX%d pixels',
                    self.resolution,self.resolution)
                          
        print('\n----------------------------------------\n')
            