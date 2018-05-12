# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 19:06:49 2013

@author: gregor.podrekar / sourced copied from scypy
"""

import numpy as np
from numpy import sqrt, newaxis, empty
from scipy import optimize
import matplotlib.pyplot as plt

def circleFit( pts ):
    """
    Fits circle using leatsq optimization!
    Input:
        - pts 2d [:,2]
    Output:
        - center
        - radius
    """
    x = pts[:,0]  
    y = pts[:,1] 
    
    def calc_R(xc, yc):
        """ calculate the distance of each data points from the center (xc, yc) """
        return sqrt((x-xc)**2 + (y-yc)**2)

    def f_2b(c):
        """ calculate the algebraic distance between the 2D points and the mean circle centered at c=(xc, yc) """
        Ri = calc_R(*c)
        return Ri - Ri.mean()
    
    def Df_2b(c):
        """ Jacobian of f_2b
        The axis corresponding to derivatives must be coherent with the col_deriv option of leastsq"""
        xc, yc     = c
        df2b_dc    = empty((len(c), x.size))    
        Ri = calc_R(xc, yc)
        df2b_dc[0] = (xc - x)/Ri                   # dR/dxc
        df2b_dc[1] = (yc - y)/Ri                   # dR/dyc
        df2b_dc    = df2b_dc - df2b_dc.mean(axis=1)[:, newaxis]
        return df2b_dc

    center_estimate = np.mean( pts, 0 )
    center_2b, ier = optimize.leastsq( f_2b , center_estimate, Dfun=Df_2b, col_deriv=True)
    
    xc_2b, yc_2b = center_2b
    Ri_2b        = calc_R(*center_2b)
    R_2b         = Ri_2b.mean()
    residu_2b    = np.sum((Ri_2b - R_2b)**2)
    return [xc_2b, yc_2b], R_2b
    
def drawCircle( image, center, radius):
    """
    Draws circle on image!
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # optionaly first draw image
    if image is not None:
        if np.size(image) == np.size(image, 0) * np.size(image, 1):
            ax.imshow( image, cmap = 'gray' )  # grayscale
            #ax.imshow( image )  # grayscale
        else:
            ax.imshow( image )  # rgb
            
    angles = np.linspace( 0, 2*np.pi, 100, endpoint=False )
    x = center[0] + radius*np.cos( angles )
    y = center[1] + radius*np.sin( angles )
 
    ax.plot( x, y, 'r.' )
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_title('Simple XY point plot')
    plt.show()

def drawCircles( image, centers, radiuses, labels =None):
    """
    Draws circles on image!
    """
    fig = plt.figure()
    ax = fig.add_subplot(111)
    # optionaly first draw image
    if image is not None:
        if np.size(image) == np.size(image, 0) * np.size(image, 1):
            ax.imshow( image, cmap = 'gray' )  # grayscale
            #ax.imshow( image )  # grayscale
        else:
            ax.imshow( image )  # rgb
            
    angles = np.linspace( 0, 2*np.pi, 100, endpoint=False )

    for circle in zip( centers, radiuses, labels ):
        x = circle[0][0] + circle[1]*np.cos( angles )
        y = circle[0][1] + circle[1]*np.sin( angles ) 
        ax.plot( x, y, 'r.' )
        if labels != None:
            ax.text( circle[0][0] , circle[0][1]+10, str(circle[2]), color='b' )
    # ax.set_xlabel('x')
    # ax.set_ylabel('y')
    # ax.set_title('Simple XY point plot')
    plt.show()
    

def trackCircle( center, rad, imShape ):
    """
    Using center and radius of a circle it finds unique pixel contour of this 
    """
    
    """
    center = ccnt
    rad = rd
    inShape = segImg.shape
    debug = False
    """
    
    # check if whole circle is inside image
    if (center[0] - rad) < 0 or (center[0] + rad) >= imShape[1] or  (center[1] - rad) < 0 or (center[1] + rad) >= imShape[0]:
        raise NameError( 'Circle partialy outside the image' )
    
    center = np.array( center )
    
    # start tracking at right side of circle, always pick neigbouring pixel which is closest to tabs radius and stop when came around
    startPoint1 = np.round( center + np.array( [ rad, 0] ) )
    
    currentPoint =  startPoint1.copy()
    contour = [ currentPoint ]
    iterNum = 0
    maxIterNum = 1000
    
    def getNextPoint():
        """
        gets next point 
        """
        surroundingPts_local = np.array( [ [1,0], [1,-1], [0,-1], [-1,-1], [-1,0], [-1,1], [0,1], [1,1] ])
        surroundingPts_global = np.tile( currentPoint, [8,1] ) + surroundingPts_local
        
        if len( contour ) > 1:
            # dont use last
            includeInd = np.sum( surroundingPts_global == contour[-2], 1 ) != 2
            # aditionaly exlude neighbout pts
            excludeInd = np.where( includeInd == False)[0][0]
            if excludeInd == 0:
                includeInd[ [1, 7] ] = False
            elif excludeInd == 7:
                includeInd[ [0, 6] ] = False
            else:
                includeInd[ [ excludeInd-1, excludeInd+1 ] ] = False
            
            surroundingPts_global = surroundingPts_global * np.tile( includeInd, [2,1] ).T
            
        # find closest to demamnded radius
        dists = np.abs( np.sqrt( np.sum( ( surroundingPts_global - np.tile( center, [8,1] ) )**2, 1 ) ) - rad )
        ind = np.argmin( dists )
        return surroundingPts_global[ ind, : ]
    
    while 1:
        # check if max num of iterations passed
        if iterNum == maxIterNum:
            print Warning( 'Reached max num of iterations. Tracking unsuccessful!' )
            #return np.array( contour ).astype(np.int), -1
            break
            
        # get next point
        nextPoint = getNextPoint()

        # in first iteraton also remember sesond tracked point.
        if iterNum is 0: 
            startPoint2 = nextPoint.copy()
        
        # check if came around
        if iterNum > 2 and ( np.sum(nextPoint == startPoint1) ==2 or  np.sum(nextPoint == startPoint2) == 2 ):
            # finished successfuly
            break 
        # print iterNum, nextPoint - startPoint1, nextPoint
            
        # add to storage
        contour.append( nextPoint )            
        # increment    
        iterNum += 1
        # reassign
        currentPoint = nextPoint.copy()

    # return result and successful flag
    return np.array( contour ).astype(np.int)
    
    
    
    
    
    
    
    
    
    



    