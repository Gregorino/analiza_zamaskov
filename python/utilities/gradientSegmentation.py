# -*- coding: utf-8 -*-
"""
Created on Fri Jun 07 15:29:24 2013

@author: gregor.podrekar
"""

from numpy import shape, transpose, reshape, where, vstack, hstack, deg2rad, mean, arange, size
from numpy import uint8, pi, sin, cos, linspace, argmax, argmin, zeros, multiply, tile, newaxis, gradient
from numpy import sqrt, median, r_, ones, round
from scipy.interpolate import RectBivariateSpline
from scipy.signal import argrelextrema
from matplotlib import pyplot as plt
from copy import copy
import numpy as np


def second_local_max( profile ):
    """
    Return second local max on the profile
    """
    inds = argrelextrema(profile, np.greater)[0]
    
    maxInd = inds[0]
        
    if 0:
        plt.plot(np.r_[0:profile.size], profile, 'r+')
        plt.plot( [maxInd, maxInd],  [0,0.02] , 'b' )
        plt.show()
        
    return maxInd


def segmentTablet_1( img, center, options ):
    """
    Standard procedure of finding points with maximum gradient and fitting circle to this points.
    
    Inputs: 
        - img
        - center [x,y]
        - options - 
            
    """
    # img = imm2; center = mean(xyF,0); pixSize = [ imGB2.px_size, imGB2.py_size ];
    
    # imshow( img )
    # plot( center[0], center[1], 'r+' )
    
    angleStep = options['angleStep'] # in degrees
    # starting and ending distance from center between which we are looking for max gradient
    dStart = options['dStart']
    dEnd = options['dEnd']
    profileStep = options['profileStep']  
    # suprese gradients whoes intensities are above this threshold
    intensity_threshold = options['intensity_threshold']    
    # weather to plot segmentation result 
    draw = options['draw']
    
    max_grad_function = options['max_grad_function'] # for example np.argmin
    
    # find coordinates at which to interpolate image -> r-Phi image
    angles = arange( 0, 2*pi, deg2rad(angleStep) )
    dists = np.linspace( dStart, dEnd, num = (dEnd-dStart)/profileStep , endpoint = True )
    anglesM = np.tile( angles[:,np.newaxis], [ 1, dists.size ] )
    distsM = np.tile( dists, [ angles.size, 1 ] )
    # go to cartesian
    XX =  center[0] +  distsM * np.cos( anglesM )   
    YY =  center[1] +  distsM * np.sin( anglesM )

    # prepare interpolation
    interP = RectBivariateSpline( r_[0:size(img,0)] , r_[0:size(img,1)] , img )
    
    borderPts = []
    borderPtsVal = []
    for x, y in zip(XX, YY):
        profile = interP.ev( y, x )
        costV = gradient( profile )
        
        msk = profile<intensity_threshold
        msk = np.hstack( [np.zeros(5), msk] )[:msk.size]
        
        eI = max_grad_function( costV * msk )
        borderPts.append( [ x[eI], y[eI] ] )
        borderPtsVal.append( costV[ eI ] ) 
    
    if 0:        
        for x, y in zip(XX, YY):
            plt.figure()
            profile = interP.ev( y, x )
            costV = gradient( profile )
           
            eI = max_grad_function( costV ) 
            
            plt.subplot(3,1,1)
            plt.imshow(img, cmap='gray')
            plt.plot( x[eI], y[eI], "r+" )
            plt.subplot(3,1,2)
            plt.plot(profile)
            plt.subplot(3,1,3)
            plt.plot(costV)
            plt.show()
    
    # CREATE ARRAYS
    borderPts = np.array( borderPts )
    borderPtsVal = np.array( borderPtsVal )    
    
    # remove NaN points    
    validPts = np.logical_not( np.isnan( borderPtsVal ) )
    borderPts = borderPts[ validPts, : ]
    borderPtsVal = borderPtsVal[ validPts ]
    angles = angles[ validPts ]    

    if draw:
      # borderPts = borderPts[ borderPtsVal < -1e+8 , :]
      plotPts( img , borderPts.T, color = '.b' )    
    
    return borderPts, borderPtsVal, angles, XX, YY
    

def segmentTablet_2( img, center, options ):
    """
    Standard procedure of finding points with maximum gradient and fitting circle to this points.
    Ridge gradient information can also be applied in this version
    """
    # img = imm2; center = mean(xyF,0); pixSize = [ imGB2.px_size, imGB2.py_size ];
    
    # imshow( img )
    # plot( center[0], center[1], 'r+' )
    
    angleStep = options['angleStep'] # in radians
    pixSize = options['pixSize']
    gradDirection = options['gradDirection'] # 'positive'  'negative'
    # additionaly ridge gradient image can be used when searching for max gradient
    # max value must also be a ridge 
    if options['useRidge'] == True:
        ridgeImage = options['ridgeImg']
    else:
        ridgeImage = ones( img.shape )
    
    # starting and ending distance from center between which we are looking for max gradient
    dStart = options['dStart']
    dEnd = options['dEnd']
    profileStep = options['profileStep']  
    # weather to plot segmentation result
    draw = options['draw']
    
    angles = arange( 0, 2*pi, deg2rad(angleStep) )    
    stPts = hstack( [ center[0]+ dStart*cos(angles)[:,newaxis]  ,  center[1]+ dStart*sin(angles)[:,newaxis]  ] )
    enPts = hstack( [ center[0]+ dEnd*cos(angles)[:,newaxis]  ,  center[1]+ dEnd*sin(angles)[:,newaxis]  ] )

    if gradDirection == 'positive':
        extremaLoc = argmax
    elif gradDirection == 'negative':
        extremaLoc = argmin
    else:
        raise ValueError('Invalid gradDirection value!')

    borderPts = zeros( shape(stPts) )
    borderPtsVal = zeros( stPts.shape[0] )
    # prepare interpolation
    interP = RectBivariateSpline( r_[0:size(img,0)] , r_[0:size(img,1)] , img )
    # interP_ridge = RectBivariateSpline( r_[0:size(img,0)] , r_[0:size(img,1)] , ridgeImage )
    nPts = (dEnd - dStart) / profileStep
    for profI in r_[ 0: size(stPts, 0) ]:
        profilePts = hstack( [ linspace( stPts[profI,0], enPts[profI,0], nPts )[:,newaxis] ,
                            linspace( stPts[profI,1], enPts[profI,1], nPts )[:,newaxis] ] )
        # check if points are out        
        validPts = np.logical_and( np.logical_and( profilePts[:,0] <= (img.shape[1]-1),  profilePts[:,1] <= (img.shape[0]-1) ) , 
                                  np.logical_and( profilePts[:,0] >= 0,  profilePts[:,1] >= 0 )  )           
        profilePts = profilePts[ validPts, : ]
        # check     
        if profilePts.size == 0:
            borderPts[profI,:] = np.array( [ np.NaN, np.NaN ] )
            borderPtsVal[ profI ] = np.NaN
            continue 
        # find point with max gradient                    
        profile = interP.ev( profilePts[:,1], profilePts[:,0]  )
        ridgeProfile = ridgeImage[ round(profilePts[:,1]).astype(int) , round( profilePts[:,0] ).astype(int)  ]
        costV = gradient( profile ) / ( abs(ridgeProfile) + 10e-8 )
        eI = extremaLoc( costV )
        # eI = extremaLoc( ( -abs(ridgeProfile) ) )
        borderPts[profI,:] = profilePts[eI, :]
        borderPtsVal[ profI ] = costV[ eI ] #* (-10e-8)
    
    # remove NaN points    
    validPts = np.logical_not( np.isnan( borderPtsVal ) )
    borderPts = borderPts[ validPts, : ]
    borderPtsVal = borderPtsVal[ validPts ]
    angles = angles[ validPts ]
    

    # fit circle
    # TODO
    if draw:
      # borderPts = borderPts[ borderPtsVal < -1e+8 , :]
      plotPts( img , borderPts.T, color = '.b' )
    
    # find median distance from center to border
    
    #first account for pixel size
    #center = multiply( center, pixSize )
    #borderPts = multiply( borderPts, tile(pixSize, [size(borderPts,0), 1] ) )
    #dist = borderPts - tile( center, [size(borderPts,0), 1] )
    #dist = sqrt( dist[:,0]**2 + dist[:,1]**2 )
    
    return borderPts, borderPtsVal, angles  # 2*mean(dist), 2*median(dist)


def segmentTablet_withContour( gradImg, contour, options ):
    """
    Here approximate contour is already given. This function finds max grads in some neighbourhood of given contour. 
    """
    # gradImg = edgeMap; contour = contour; options = { 'dOff_before': 2, 'dOff_after': 2, 'profileStep': 0.1, 'draw': False }
    
    # check inputs
    if len( contour.shape ) != 2:
        raise NameError('Contour must be of size [n,2]')
    if contour.shape[1] != 2:
        raise NameError('Contour must be of size [n,2] ')
    
    # starting and ending distance f
    dOff_before = options['dOff_before']
    dOff_after = options['dOff_after']
    profileStep = np.abs( options['profileStep'] )  
    # weather to plot segmentation result
    draw = options['draw']
    
    # calculate center of the contour
    center = np.mean( contour, 0 )    
    
    # find coordinates at which to interpolate image -> r-Phi image
    diff = contour - np.tile( center, [contour.shape[0],1])    
    angles = np.arctan2( diff[:,1], diff[:,0] )
    
    # get profiles locs    
    distsM = []
    for contourPoint, angle in zip( contour, angles ):
        # ind = 0; contourPoint = contour[ind]; angle = angles[ind]
        dist = np.sqrt( ( contourPoint[0] - center[0] )**2 + ( contourPoint[1] - center[1] )**2 )
        dists = np.linspace( dist-dOff_before , dist+dOff_after , num = (dOff_before+dOff_after)/profileStep , endpoint = True )
        distsM.append( dists )
    distsM = np.array( distsM )   
       
    anglesM = np.tile( angles, [ (dOff_before+dOff_after)/profileStep, 1] ).T

    # go to cartesian
    XX =  center[0] +  distsM * np.cos( anglesM )   
    YY =  center[1] +  distsM * np.sin( anglesM )

    # always search for maximum
    extremaLoc = argmax

    # prepare interpolation
    interP = RectBivariateSpline( r_[0:size(gradImg,0)] , r_[0:size(gradImg,1)] , gradImg )
    
    borderPts = []
    borderPtsVal = []
    for x, y in zip(XX, YY):
        profile = interP.ev( y, x )
        costV = gradient( profile )
        eI = extremaLoc( costV )  
        
        borderPts.append( [ x[eI], y[eI] ] )
        borderPtsVal.append( costV[ eI ] ) 
    
    # CREATE ARRAYS
    borderPts = np.array( borderPts )
    borderPtsVal = np.array( borderPtsVal )    
    
    # remove NaN points    
    validPts = np.logical_not( np.isnan( borderPtsVal ) )
    borderPts = borderPts[ validPts, : ]
    borderPtsVal = borderPtsVal[ validPts ]
    angles = angles[ validPts ]    

    if draw:
      # borderPts = borderPts[ borderPtsVal < -1e+8 , :]
      plotPts( gradImg , borderPts.T, color = '.b' )    
    
    return borderPts, borderPtsVal, angles


def getPtGradient( img, pt ):
    # function returns gradient of point pt [x,y] in given image, img    
    x = pt[0]
    y = pt[1]    
    GX = -np.double(img[ y-1, x-1]) -2*np.double(img[ y, x-1]) - np.double(img[ y+1, x-1]) +np.double(img[ y-1, x+1]) +2*np.double(img[ y, x+1]) + np.double(img[ y+1, x+1] )
    GY = -np.double(img[ y-1, x-1]) -2*np.double(img[ y-1, x]) - np.double(img[ y-1, x+1]) +np.double(img[ y+1, x-1]) +2*np.double(img[ y+1, x]) + np.double(img[ y+1, x+1] )
    return GX, GY, np.arctan2( GY, GX )
 
def getPtsGradient( img, pts ):
    # function returns gradient of points pt [x,y ; ...] in given image, img     
    x = np.round( pts[:,0] ).astype( np.int )
    y = np.round( pts[:,1] ).astype( np.int )    
    GX = -np.double(img[ y-1, x-1]) -2*np.double(img[ y, x-1]) - np.double(img[ y+1, x-1])    + np.double(img[ y-1, x+1]) +2* np.double(img[ y, x+1]) + np.double(img[ y+1, x+1])      
    GY = -np.double(img[ y-1, x-1]) -2*np.double(img[ y-1, x]) - np.double(img[ y-1, x+1])    +np.double(img[ y+1, x-1]) +2*np.double(img[ y+1, x]) + np.double(img[ y+1, x+1]) 
    return GX, GY, np.arctan2( GY, GX )
   
def plotPts( image, vertices, color = 'r' ):
    #fig = plt.figure()
    #ax = fig.add_subplot(111)
    #plt.ion()
    plt.figure()
    ax = plt.axes()
    # optionaly first draw image
    if image != None:
        if size(image) == size(image, 0) * size(image, 1):
            ax.imshow( image, cmap = 'gray' )  # grayscale
            #ax.imshow( image )  # grayscale
        else:
            ax.imshow( image )  # rgb
 
    ax.plot( vertices[0,:], vertices[1,:], color )
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_title('Simple XY point plot')
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    