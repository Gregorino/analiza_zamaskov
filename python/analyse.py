import glob
import os

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import cv2

from skimage.filters import gaussian
from skimage.segmentation import active_contour

from utilities.circle import circleFit

def parse_path(path):
    """
    base name is coded as signature_good/bad_count
    
    for example:
        jr_g_1    jr, good, count 1
        br_b_10   bt, bad, count 10
    """
    
    base_name = os.path.splitext( os.path.basename(path) )[0]    
    splits = base_name.split("_")
    
    signature = splits[0]
    count = splits[2]    
    good = splits[1] == "g"
    
    return signature, count, good

def circle(center, rx, ry, num_pts):
    s = np.linspace(0, 2*np.pi, num_pts)
    x = center[0] + rx*np.cos(s)
    y = center[1] + ry*np.sin(s)
    return np.array([x, y]).T

def plot1(img, init, snake, fitted_circle):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap=plt.cm.gray)
            
    step = 1
    ax.plot(init[::step, 0], init[::step, 1], '--r', lw=3)
    ax.plot(snake[::step, 0], snake[::step, 1], 'b.', lw=3)
            
    ax.plot(fitted_circle[::step, 0], fitted_circle[::step, 1], 'g.', lw=3)
 
    plt.show()

if __name__ == "__main__":
    
    pth_good = r"C:\Users\gregor.podrekar\Google Drive\delo\privatni_projekti\analiza_zamaskov\image_database\good"
    pth_bad = r"C:\Users\gregor.podrekar\Google Drive\delo\privatni_projekti\analiza_zamaskov\image_database\bad"
    
    imgs_good = glob.glob(pth_good + '/*.bmp')
    imgs_bad = glob.glob(pth_bad + '/*.bmp')
    
    img_list = imgs_good + imgs_bad
    
    good_scores = []
    bad_scores = []
    
    for img_name in img_list:
        
        signature, count, good = parse_path(img_name)
        print signature, count, good
        
        img = cv2.imread( img_name, 0)   
        
        if 0:
            plt.imshow(img, cmap= 'gray')
            plt.show()
        
        # try active contours
        init = circle( np.array(img.shape)/2, img.shape[1]/2.01, img.shape[0]/2.01, 360 )
        
        # alpha: Snake length shape parameter. Higher values makes snake contract faster.
        # beta: Snake smoothness shape parameter. Higher values makes snake smoother.
        # gamma: Explicit time stepping parameter.
        # w_line: Controls attraction to brightness. Use negative values to attract to dark regions.
        # w_edge: Controls attraction to edges. Use negative values to repel snake from edges.    
        snake = active_contour(gaussian(img, 3), init, alpha=0.015, beta=0.01, gamma=0.001, w_line = 0, w_edge = 1 )
        
        # fit circle
        center, radius = circleFit(snake.astype(np.float32))
        fitted_circle = circle(center, radius, radius, 360)        
              
        if 0:
            plot1(img, init, snake, fitted_circle)
        
        snake_error = np.abs(np.sqrt((snake[:,0]-center[0])**2 + (snake[:,1]-center[1])**2) - radius)
        
        if 0:
            plt.plot(snake_error, "g.")
            plt.show()
            
        meanError = np.mean(snake_error)
        maxError = np.max(snake_error)
        
        if good:
            good_scores.append([meanError, maxError])
        else:
            bad_scores.append([meanError, maxError])
            
        maxError_thresh = 4
        if False and ( (good and (maxError > maxError_thresh) ) or ( (not good) and (maxError < maxError_thresh)) ):
            plt.imshow(img, cmap='gray')
            plt.show()
            
            plot1(img, init, snake, fitted_circle) 
     
    good_scores = np.array(good_scores) 
    bad_scores = np.array(bad_scores)  
        
    # plot
    plt.figure()
    nbins = 20
    
    plt.subplot(2,1,1)
    sns.distplot(good_scores[:,0], color = "g", kde=False, bins = nbins)
    sns.distplot(bad_scores[:,0], color = "r", kde=False, bins = nbins)
    
    plt.subplot(2,1,2)
    sns.distplot(good_scores[:,1], color = "g", kde=False, bins = nbins)
    sns.distplot(bad_scores[:,1], color = "r", kde=False, bins = nbins)
    
    plt.show()
    
    plt.plot(np.hstack([good_scores[:,0], bad_scores[:,0]]), np.hstack([good_scores[:,1], bad_scores[:,1]]), "go")
    plt.show()
    
    print "Finished"
        