# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:48:14 2019

@author: Sneha
"""

import cv2
import numpy as np
import os
from trainClassifier import trainClassifier
from mser import mser
from preprocessing import preprocess
from getsign import getsign
from hsv_threshold import hsv
#from getsignred import getsignred

features, labels, classifier=trainClassifier()

frames = []
path = 'input\\'

for frame in os.listdir(path):
    frames.append(frame)
    
    
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('detect5.mp4',fourcc,20,(600,600))      
for i in range(95,110):
    print(i)
    img = cv2.imread('input\\'+str(frames[i]))
    img = cv2.resize(img,(600,600))
    red_norm, blue_norm = preprocess(img)
    red_mask,blue_mask = hsv(img)
    region_red = mser(red_norm,'red')
    red_mser = np.zeros((600,600))
    for p in region_red:
        for i in range(len(p)):
            red_mser[p[i][1],p[i][0]]=1
    
    region_blue = mser(blue_norm,'blue')
    blue_mser = np.zeros((600,600))
    for p in region_blue:
        for i in range(len(p)):
            blue_mser[p[i][1],p[i][0]]=1
    red = np.zeros((600,600))
    for i in range(600):
        for j in range(600):
            if red_mser[i,j]==1 and red_mask[i,j]==1:
                red[i,j]=255
            if red_mser[i,j]==1 and red_mask[i,j] ==0:
                red[i,j]=0
            if red_mser[i,j] == 0 and red_mask[i,j]==1:
                red[i,j]==0
            if red_mser[i,j]==0 and red_mask[i,j]==0:
                red[i,j]==0
    red = red.astype(np.uint8)
    x1,y1 = np.where(red==255)[0],np.where(red==255)[1]
    if len(x1)>0 and len(y1)>0:
        xmax1,xmin1 = np.max(y1),np.min(y1)
        ymax1,ymin1 = np.max(x1),np.min(x1)
        if (xmax1-xmin1)*(ymax1-ymin1)<200 and (xmax1-xmin1)*(ymax1-ymin1)>800 and (xmax1-xmin1)/(ymax1-ymin1)>1.2 and (xmax1-xmin1)/(ymax1-ymin1)<0.6:
            getsign(img,'red',[xmax1,ymax1,xmin1,ymin1],classifier,features,labels)
            cv2.rectangle(img, (xmin1,ymax1), (xmax1,ymin1), (0, 255, 0), 1)
            video.write(img)
    else:
        video.write(img)
    blue = np.zeros((600,600))
    for i in range(600):
        for j in range(600):
            if blue_mser[i,j]==1 and blue_mask[i,j]==1:
                blue[i,j]=255
            if blue_mser[i,j]==1 and blue_mask[i,j] ==0:
                blue[i,j]=0
            if blue_mser[i,j] == 0 and blue_mask[i,j]==1:
                blue[i,j]==0
            if blue_mser[i,j]==0 and blue_mask[i,j]==0:
                blue[i,j]==0
    blue = blue.astype(np.uint8)
    x2,y2 = np.where(blue==255)[0],np.where(blue==255)[1]
    if len(x2)>0 and len(y2)>0:
        xmax2,xmin2 = np.max(y2),np.min(y2)
        ymax2,ymin2 = np.max(x2),np.min(x2)
        if (xmax2-xmin2)*(ymax2-ymin2)<200 or (xmax2-xmin2)*(ymax2-ymin2)>800 or (xmax2-xmin2)/(ymax2-ymin2)>1.2 or (xmax2-xmin2)/(ymax2-ymin2)<0.6:
            getsign(img,'blue',[xmax2,ymax2,xmin2,ymin2],classifier,features,labels)
            cv2.rectangle(img, (xmin2,ymax2), (xmax2,ymin2), (0, 255, 0), 1)
            video.write(img)
    else:
        video.write(img)

video.release()
