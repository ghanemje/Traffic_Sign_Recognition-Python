import numpy as np
import os
import cv2

# TRY:
'''  resize the image to make it faster'''
def Mser(img):
        mser = cv2.MSER_create(_min_area=100,_max_area=800)
        regions, _ = mser.detectRegions(img)
        if (len(regions))>0:
            for p in regions:
                xmax, ymax = np.amax(p, axis=0)
                xmin, ymin = np.amin(p, axis=0)
        else:
                xmax,ymax = None,None
                xmin,ymin = None,None
        return xmax,ymax,xmin,ymin

def mser(img,str):
    if str == 'blue':
        mser = cv2.MSER_create(_min_area=500,_max_area=1000)
    if str=='red':
        mser = cv2.MSER_create(_min_area=400,_max_area=800)
    regions, _ = mser.detectRegions(img)

    return regions