#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 16 16:50:31 2018

@author: omartin
"""

import numpy as np
import scipy.special as spc
import scipy.optimize as spo
import math
import sys

#%%
def fprintf(stream, format_spec, *args):
    stream.write(format_spec % args)
    
class Attribute(object):
    pass

class layer:
    def __init__(self,r0,weight,height,L0,wSpeed,wDir):
        self.r0     = r0
        self.weight = weight
        self.height = height
        self.L0     = L0
        self.wSpeed = wSpeed
        self.wDir   = wDir
        
class atmosphere:
    """ Atmosphere class that wraps up atmosphere identification.
    Inputs are:
        - wvl: wavelength for scalar parameters definition
        - r0: r0 value at wvl in meter
        - weights: bins fractional weight with w0.sum = 1
        - heights: bins altitude in meter
        - L0: outer scale profile. If scalar value, it is set identically to any bins
        - wSpeed: wind speed values in m/s        
        - wDir: wind direction in degree
    """
    
    # DEPENDANT VARIABLES DEFINITION     
    def get_wvl(self):
        return self.p_wvl
    
    def set_wvl(self,val):
        self.r0  = self.r0*(val/self.wvl)**1.2
        self.p_wvl = val
    
    wvl = property(get_wvl,set_wvl)
        
    @property
    def seeing(self):
        """Seeing value in arcsec at self.wvl"""
        return 3600*180/np.pi*0.98*self.wvl/self.r0
    
    @property
    def theta0(self):
        """ Isoplanatic angle in arcsec at self.wvl"""        
        if not any(self.heights):
            th0  = math.inf
        else:
            if np.isinf(self.L0):
                # We use the Roddier exponential coherence function decay
                cst = (24*spc.gamma(6/5)/5)**(-5/6)*self.r0**(5/3)
                th0 = ( cst/sum(self.weights*self.heights**(5/3) ) )**(3/5)
            else:                                
                func= lambda x: self.angularStructureFunction(x) -2
                th0 = abs(spo.fsolve(func,0))                
                              
        
        return th0[0]*3600*180/np.pi
    
    @property    
    def meanHeight(self):
        """ Mean-weighted height in meter"""
        return sum( self.weights*self.heights**(5/3) )**(3/5)
        
    @property            
    def meanWind(self):
        """ Mean-weighted wind speed in m/s"""
        return sum( self.weights*self.wSpeed**(5/3) )**(3/5)    
                    
    def __init__(self,wvl,r0,weights,heights,wSpeed=0.0,wDir=0.0,L0=math.inf,verbose=False):
        
        # PARSING INPUTS        
        self.r0  = r0        
        self.p_wvl = wvl # = source.wvl
        self.nL  = len(weights)
        self.weights = np.array(weights)
        self.heights = np.array(heights)
        self.wSpeed  = np.array(wSpeed)
        self.wDir    = np.array(wDir)
        self.verbose = verbose

        # MANAGE THE L0 VALUE
        if np.isscalar(L0) | (not np.isscalar(L0) and len(L0)==1):
            self.L0 = L0
            L0      = L0*np.ones(self.nL)
        if np.isscalar(wSpeed):            
            wSpeed  = wSpeed*np.ones(self.nL)
        if np.isscalar(wDir):            
            wDir    = wDir*np.ones(self.nL)
            
        # DEFINE LAYERS             
        if self.nL>1:
            self.layer = np.repeat(Attribute(),self.nL)
            for l in np.arange(0,self.nL):            
                tmp  = (weights[l]*r0**(-5/3))**(-3/5)
                self.layer[l] = layer(tmp,weights[l],heights[l],L0[l],wSpeed[l],wDir[l])           
        else:
            self.layer = layer(r0,weights,heights,L0,wSpeed,wDir)           
            
              
    
    def slab(self,layerIndex):
        """SLAB Create a single turbulence layer atmosphere object            
        singledAtm = slab(atm,k) creates an atmosphere object from
        the old atm object and the k-th turbulent layer"""
        r0l= (self.layer[layerIndex].weight*self.r0**(-5/3))**(-3/5)
        hl = [self.layer[layerIndex].height]
        L0l= self.layer[layerIndex].L0
        wSl= [self.layer[layerIndex].wSpeed]
        wDl= [self.layer[layerIndex].wDir]
        
        return atmosphere(self.wvl,r0l,[1.],hl,L0=L0l,wSpeed=wSl,wDir=wDl)
        
        
    def __repr__(self):
        """DISPLAY Display object information            
           atm.display prints information about the atmosphere object
        """  
        
        
        s = ('___ ATMOSPHERE ___\n')
        if np.isinf(self.L0):
            s += ' Kolmogorov-Tatarski atmospheric turbulence:\n'
            s+= '.wavelength=%5.2fmicron,\n.r0 \t\t= %5.2fcm,\n.seeing \t= %5.2farcsec,\n'%(self.wvl*1e6,self.r0*1e2,self.seeing)
        else:
            s += (' Von Karman atmospheric turbulence\n')
            s+= '.wavelength  \t=%5.2fmicron,\n.r0 \t\t= %5.2fcm,\n.L0 \t\t= %5.2fm,\n.seeing \t=%5.2farcsec,\n'%(self.wvl*1e6,self.r0*1e2,self.L0,self.seeing)
            
        if not np.isinf(self.theta0):
            s+=('.theta0 \t=%5.2farcsec'%self.theta0)
            
        
        s+=('\n----------------------------------------------------\n')
        s+=(' Layer\t Height [m]  Weight  L0[m]  wind([m/s] [deg])\n')
        if self.nL > 1:
            for l in np.arange(0,self.nL):
                fprintf(sys.stdout,' %2d\t %8.2f    %4.2f    %4.2f     (%5.2f %6.2f)\n',
                l,
                self.layer[l].height,
                self.layer[l].weight,
                self.layer[l].L0,
                self.layer[l].wSpeed,
                self.layer[l].wDir)
        else:
            fprintf(sys.stdout,' %2d\t %8.2f    %4.2f    %4.2f     (%5.2f %6.2f)\n',
                0,
                self.layer.height[0],
                self.layer.weight[0],
                self.layer.L0,
                self.layer.wSpeed[0],
                self.layer.wDir[0])
                
        s+=('----------------------------------------------------\n')              
        return s
#%%        
    # ATMOSPHERE STATISTICS
    def variance(atm):
        """ Phase variance: computes the phase variance in rd^2  
            from an atmosphere object            
        """    
        c1 = (24*spc.gamma(6/5)/5)**(5/6)
        c2 = spc.gamma(11/6)*spc.gamma(5/6)/(2*np.pi**(8/3))
        
        return c1*c2*(atm.L0/atm.r0)**(5/3)
        
    def covariance(atm,rho):
        """COVARIANCE Phase covariance        
        out = phaseStats.covariance(rho,atm) computes the phase covariance from
        the baseline rho and an atmosphere object
        """
        c1       = (24*spc.gamma(6/5)/5)**(5/6)
        c2       = spc.gamma(11/6)/(2**(5/6)*np.pi**(8/3))    
        c3       = spc.gamma(11/6)*spc.gamma(5/6)/(2*np.pi**(8/3))
        L0r0ratio= (atm.L0/atm.r0)**(5/3)        
            
        if not np.isscalar(rho):
            cov      = c1*c3*L0r0ratio*np.ones(rho.shape)        
            index    = rho!=0
            u        = 2*np.pi*rho[index]/atm.L0
            cov[index] = c1*c2*L0r0ratio*u**(5/6)*spc.kv(5/6,u)
            #cov[rho==0]= c1*c2*L0r0ratio*(2**(5/6)*spc.gamma(11/6))*0.6
        else:
            if rho==0:
                cov = c1*c3*L0r0ratio
            else:
                u   = 2*np.pi*rho/atm.L0
                cov = c1*c2*L0r0ratio*u**(5/6)*spc.kv(5/6,u)
        
        return cov
        
    def structureFunction(atm,rho):
        """STRUCTUREFUNCTION Phase structure function computes the phase structure function from
        the baseline rho and an atmosphere object
        """            
        var = atm.variance()    
        if np.isinf(atm.L0):         
            if not np.isscalar(rho):
                sf   = np.zeros(rho.shape)
                index = rho!=0
                sf[index] = 2*(24*spc.gamma(6/5)/5)**(5/6)*(rho[index]/atm.r0)**(5/3)
            else:
                sf = 2*(24*spc.gamma(6/5)/5)**(5/6)*(rho/atm.r0)**(5/3)
        else:
            sf = 2*(var- atm.covariance(rho))
            
        return sf    
        
            
    def spectrum(atm,k):        
        """SPECTRUM Phase power spectrum density computes the phase power
            spectrum density from the spatial frequency f and an
            atmosphere object
        """                        
        cte = (24*spc.gamma(6/5)/5)**(5/6)*(spc.gamma(11/6)**2./(2.*np.pi**(11/3)))
        return atm.r0**(-5/3)*cte*(k**2 + 1/atm.L0**2)**(-11/6)
    
    def angularCovariance(atm,theta):
        """ ANGULARCOVARIANCE Phase angular covariance computes the
        phase angular covariance from the zenith angle theta and an
        atmosphere object
        """        
        if not np.isscalar(theta):
            cov = np.zeros(theta.shape)            
        else:
            cov = 0            
        
        if atm.nL >1:
            for l in np.arange(0,atm.nL):
                atmSlab = atm.slab(l)
                tmp = atmSlab.covariance(atmSlab.layer.height[0]*np.tan(theta))
                cov = cov + tmp
                print(cov)
        else:
            cov = atm.covariance(atm.layer.height[0]*np.tan(theta))
            
        return cov    
                        
    def angularStructureFunction(atm,theta):
        """ANGULARSTRUCTUREFUNCTION Phase angular structure function computes
        the phase angular structure function from the zenith angle theta
        and an atmosphere object                    
        """
            
        if not np.isscalar(theta):
            sf = np.zeros(theta.shape)            
        else:
            sf = 0                       
        
        if atm.nL >1:
            for l in np.arange(0,atm.nL):
                atmSlab = atm.slab(l)                
                tmp = atmSlab.covariance(atm.layer[l].height*np.tan(theta))
                sf = sf + 2*( atmSlab.variance() - tmp)  
        else:
            sf = 2*(atm.variance() - atm.covariance(atm.layer.height[0]*np.tan(theta)))
            
        return sf