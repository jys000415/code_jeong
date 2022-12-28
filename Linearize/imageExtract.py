# -*- coding: utf-8 -*-
"""
Created on Fri Oct 16 17:09:48 2020

@author: yeong
"""
rawpath = ['J:/Jeong Yeongseok/Project_spatial_coding/Recording/023274/023274-210326-Base-OF-EZM-2ndEZM-Base',
           'J:/Jeong Yeongseok/Project_spatial_coding/Recording/023275/023275-210326-Base-EZM-2ndEZM-OF-Base',
           'J:/Jeong Yeongseok/Project_spatial_coding/Recording/023277/023277-210330-Base-OF-EZM-2ndEZM-Base',
           'J:/Jeong Yeongseok/Project_spatial_coding/Recording/023697/023697-20210601-Base-1stEZM-2ndEZM-postEZM',
           'J:/Jeong Yeongseok/Project_spatial_coding/Recording/023698/023698-210531-Base-1stEZM-2ndEZM-Base',
           'J:/Jeong Yeongseok/Project_spatial_coding/Recording/023699/023699-210531-Base-1stEZM-2ndEZM-Base']
import cv2
import os
os.chdir('J:/Jeong Yeongseok/Project_spatial_coding/Recording/023699/023699-210531-Base-1stEZM-2ndEZM-Base')
cam = cv2.VideoCapture("J:/Jeong Yeongseok/Project_spatial_coding/Recording/023699/023699-210531-Base-1stEZM-2ndEZM-Base/ezm_1.avi") 
try: 
      
    # creating a folder named data 
    if not os.path.exists('data'): 
        os.makedirs('data') 
  
# if not created then raise error 
except OSError: 
    print ('Error: Creating directory of data') 
  
# frame 
currentframe = 0
  
while(True): 
      
    # reading from frame 
    ret,frame = cam.read() 
  
    if ret: 
        # if video is still left continue creating images 
        name = './data/frame' + str(currentframe) + '.jpg'
        print ('Creating...' + name) 
  
        # writing the extracted images 
        cv2.imwrite(name, frame) 
  
        # increasing counter so that it will 
        # show how many frames are created 
        currentframe += 1
    else: 
        break
  
# Release all space and windows once done 
cam.release() 
cv2.destroyAllWindows() 