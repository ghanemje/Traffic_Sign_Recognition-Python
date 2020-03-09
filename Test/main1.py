# -*- coding: utf-8 -*-
"""
Created on Wed May 15 16:48:14 2019

@author: Sneha
"""

import cv2
import numpy as np
import os
from trainClassifier import trainClassifier
from mser import Mser
from preprocessing import preprocess
from getsign import getsign

import matplotlib.pyplot as plt
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import PIL.Image, PIL.ImageTk
def nextImg(root):
    root.quit()
    root.destroy()
def click_and_rec(image,bbox,name):
	# grab references to the global variables
     image= image[bbox[3]:bbox[1],bbox[2]:bbox[0]]
     cv2.imwrite('neg/'+name+'.jpg',image)

def cropFalse(cv_img,box,name):
            root= tk.Tk() 
#            cv_img = cv2.cvtColor(cv2.imread(inputFolder+"frame0020.jpg"), cv2.COLOR_BGR2RGB)
            
            
#            image = cv2.imread(inputFolder+"frame0020.jpg", -1)
            # mask defaulting to black for 3-channel and transparent for 4-channel
            # (of course replace corners with yours)
#            mask = np.zeros(image.shape, dtype=np.uint8)
#            # fill the ROI so it doesn't get wiped out when the mask is applied
#            channel_count = image.shape[2]  # i.e. 3 or 4 depending on your image
#            ignore_mask_color = (255,)*channel_count
            
            # from Masterfool: use cv2.fillConvexPoly if you know it's convex
            
            
            
            
            # Get the image dimensions (OpenCV stores image data as NumPy ndarray)
            height, width, no_channels = cv_img.shape
             
            # Create a canvas that can fit the above image
            canvas = tk.Canvas(root, width = width, height = height)
            canvas.pack()
             
            # Use PIL (Pillow) to convert the NumPy ndarray to a PhotoImage
            photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(cv_img),master=root)
             
             # Add a PhotoImage to the Canvas
            canvas.create_image(0, 0, image=photo, anchor=tk.NW)
            btn_blur1=tk.Button(root, text="Next", width=50, command= lambda:nextImg(root))
            btn_blur1.pack(anchor=tk.CENTER, expand=True)
            btn_blur=tk.Button(root, text="Rectangle", width=50, command= lambda:click_and_rec(cv_img,box,name))
            btn_blur.pack(anchor=tk.CENTER, expand=True)

            
            root.mainloop()
#from getsignred import getsignred
#from getsignred import getsignred

#features, labels, classifier=trainClassifier()

frames = []
path = 'input\\'

for frame in os.listdir(path):
    frames.append(frame)
    
        
#im=cv2.imread('input/image.032729.jpg')
#height , width , layers =  im.shape
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter('detect1.mp4',fourcc,20,(600,600))  
for m in range(1400,len(frames)):
    print(m)
    img = cv2.imread('input\\'+str(frames[m]))
    img = cv2.resize(img,(600,600))
    red_norm,blue_norm = preprocess(img)

    xmax1,ymax1,xmin1,ymin1 = Mser(red_norm)
    xmax2,ymax2,xmin2,ymin2 = Mser(blue_norm)
    
   
#    getsignblue(img,[xmax2,ymax2,xmin2,ymin2],classifier)
    
    if xmax1!=None and (ymax1)!=None and xmin1!=None and (ymin1)!=None:
        flag=getsign(img,'red',[xmax1,ymax1,xmin1,ymin1],classifier,features,labels)
        cv2.rectangle(img, (xmin1,ymax1), (xmax1,ymin1), (0, 255, 0), 1)
        cropFalse(img,[xmax1,ymax1,xmin1,ymin1],str(m)+'-r')
#        cv2.imshow('dst_rt',img)
#        cv2.waitKey(0)
        video.write(img)
    else:
        video.write(img)
    if xmax2!=None and (ymax2)!=None and xmin2!=None and (ymin2)!=None:
        flag=getsign(img,'blue',[xmax2,ymax2,xmin2,ymin2],classifier,features,labels)
        cv2.rectangle(img, (xmin2,ymax2), (xmax2,ymin2), (0, 255, 0), 1)
        cropFalse(img,[xmax2,ymax2,xmin2,ymin2],str(m)+'-b')
#        cv2.imshow('dst_rt',img)
#        cv2.waitKey(0)
#        video.write(img)
        video.write(img)
    else:
        video.write(img)
#    cv2.imshow('dst_rt',final)
#    cv2.waitKey(0)
#    cv2.destroyAllWindows
video.release()

#im=cv.imread('testing_selected/00001/00025_00000.ppm')
#cv.imshow('result',im)

#frame_increase = 0
#frame_num = str(32640+frame_increase).zfill(4) + ".jpg"
#print(frame_num)
#cap = cv.VideoCapture("input/image.0" + frame_num)
#success, frame = cap.read()
#
#trainClassifier()
#
#while(success):
#
#   
#
#    #converting to HSV
#    hsv = cv.cvtColor(frame,cv.COLOR_BGR2HSV)
#
#    # get info from track bar and appy to result
#    h = cv.getTrackbarPos('h','result')
#    s = cv.getTrackbarPos('s','result')
#    v = cv.getTrackbarPos('v','result')
#
#    mh = cv.getTrackbarPos('mh','result')
#    ms = cv.getTrackbarPos('ms','result')
#    mv = cv.getTrackbarPos('mv','result')
#
#    # Normal masking algorithm
#    lower_blue = np.array([h,s,v])
#    upper_blue = np.array([mh,ms,mv])
#
#    mask = cv.inRange(hsv,lower_blue, upper_blue)
#
#    result = cv.bitwise_and(frame,frame,mask = mask)
#
#
#
#    cv.imshow('result',result)
#
#    k = cv.waitKey(5) & 0xFF
#    if k == 27:
#        break
#    
#    frame_increase += 1
#    frame_num = str(32640 + frame_increase).zfill(4) + ".jpg"
#    print(frame_num)
#    cap = cv.VideoCapture("input/image.0" + frame_num)
#    success, frame = cap.read()
#cap.release()
#
#cv.waitKey(0)
#cv.destroyAllWindows()