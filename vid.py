# -*- coding: utf-8 -*-
"""
Created on Wed May 15 17:11:42 2019

@author: Sneha
"""

# -*- coding: utf-8 -*-
"""
Created on Mon Apr 15 19:58:55 2019

@author: Sneha
"""

import cv2
import numpy as np


frame_increase = 0
frame_num = str(32640+frame_increase).zfill(4) + ".jpg"
print(frame_num)
cap = cv2.VideoCapture("input/image.0" + frame_num)
im=cv2.imread('input/image.032640.jpg')
height , width , layers =  im.shape
video = cv2.VideoWriter('video.mp4',-1,10,(width,height))
success, image = cap.read()

count = 0
p = np.zeros(6)
while success:
#    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    clone = image.copy()
    
#            cv2.imshow("image", image)
#            key = cv2.waitKey(1) & 0xFF

            # if the 'r' key is pressed, reset the cropping region
#            if key == ord("r"):
#                image = clone.copy()

    video.write(image)

    frame_increase += 1
    frame_num = str(32640 + frame_increase).zfill(4) + ".jpg"
    print(frame_num)
    cap = cv2.VideoCapture("input/image.0" + frame_num)
    success, image = cap.read()


video.release()
cv2.destroyAllWindows()


