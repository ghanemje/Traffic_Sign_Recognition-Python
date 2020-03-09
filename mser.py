import numpy as np
import os
import cv2

# TRY:
'''  resize the image to make it faster'''


def mser(img,str):
    if str == 'blue':
        mser = cv2.MSER_create(_min_area=200,_max_area=1000)
    if str=='red':
        mser = cv2.MSER_create(_min_area=200,_max_area=1000)
    regions, _ = mser.detectRegions(img)

    return regions