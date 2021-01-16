#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 12 09:01:11 2019

@author: thomas.topilko
"""

import cv2;
import os;
import matplotlib.pyplot as plt;

_refPt = Tracker._refPt

RGBFrame = Tracker.RGBFrame
DEPTHFrame = Tracker.DEPTHFrame

_H,_W,_C = RGBFrame.shape; #Gets the Height, Width and Color of each frame
cloneFrame = RGBFrame.copy(); #[DISPLAY ONLY] Creates a clone frame for display purposes

upLeftX = int(_refPt[0][0]); #Defines the Up Left ROI corner X coordinates
upLeftY = int(_refPt[0][1]); #Defines the Up Left ROI corner Y coordinates
lowRightX = int(_refPt[1][0]); #Defines the Low Right ROI corner X coordinates
lowRightY = int(_refPt[1][1]); #Defines the Low Right ROI corner Y coordinates

#cv2.rectangle(cloneFrame,(upLeftX,upLeftY),(lowRightX,lowRightY),(0,255,0),3);
#plt.imshow(cloneFrame)

distanceRatio = (abs(upLeftX-lowRightX)/params.segmentationParameters["cageLength"]+\
                abs(upLeftY-lowRightY)/params.segmentationParameters["cageWidth"])/2; #Defines the resizing factor for the cage

croppedFrame = RGBFrame[upLeftY:lowRightY,upLeftX:lowRightX]; #Crops the initial frame to the ROI

#plt.imshow(croppedFrame)

maskDisplay = cv2.cvtColor(croppedFrame, cv2.COLOR_BGR2RGB); #[DISPLAY ONLY] Changes the croppedFrame to RGB for display purposes
maskDisplay = cv2.cvtColor(maskDisplay, cv2.COLOR_RGB2BGR);

colorFrame = cv2.cvtColor(cloneFrame, cv2.COLOR_BGR2RGB); #[DISPLAY ONLY] Changes the Frame to RGB for display purposes
colorFrame = cv2.cvtColor(colorFrame, cv2.COLOR_RGB2BGR);

#Filtering the ROI from noise
#----------------------------------------------------------------------------------------------------------------------------------

hsvFrame = cv2.cvtColor(croppedFrame, cv2.COLOR_BGR2HSV); #Changes the croppedFrame LUT to HSV for segmentation
#plt.imshow(hsvFrame)
blur = cv2.blur(hsvFrame,(5,5)); #Applies a Gaussian Blur to smoothen the image
#plt.imshow(blur)
mask = cv2.inRange(blur, params.segmentationParameters["threshMinMouse"], params.segmentationParameters["threshMaxMouse"]); #Thresholds the image to binary
#plt.imshow(mask)
opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,params.segmentationParameters["kernel"], iterations = 1); #Applies opening operation to the mask for dot removal
#plt.imshow(opening)
closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,params.segmentationParameters["kernel"], iterations = 1); #Applies closing operation to the mask for large object filling
#plt.imshow(closing)

cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]; #Finds the contours of the image to identify the meaningful object

cloneCroppedFrame = croppedFrame.copy();
areas = [];

for cnt in cnts :
    area = cv2.contourArea(cnt);
    areas.append(area);

#plt.hist(areas,bins=10);
#cv2.drawContours(cloneCroppedFrame,cnts,-1,(255,0,0),3)
#plt.imshow(cloneCroppedFrame);
    
cap = cv2.VideoCapture(os.path.join(_workingDir,"Raw_Video_Mice_226_217_7-2-2019_8-54-10.avi"),cv2.CAP_ANY);
#cap = cv2.VideoCapture(os.path.join(_workingDir,"Depth_Video_Mice_8b_226_217_7-2-2019_8-54-10.avi"),cv2.CAP_FFMPEG);

frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
cap.set(cv2.CAP_PROP_POS_FRAMES, frames-1)
res, frame = cap.read()
plt.imshow(frame)
    
params.segmentationParameters["capturesRGB"][0].set(cv2.CAP_PROP_POS_AVI_RATIO,1)
params.segmentationParameters["capturesRGB"][0].get(cv2.CAP_PROP_POS_MSEC)
params.segmentationParameters["capturesRGB"][0].get(cv2.CAP_PROP_POS_FRAMES)

import skvideo.io
cap = skvideo.io.vreader(os.path.join(_workingDir,"Raw_Video_Mice_226_217_7-2-2019_8-54-10.avi"))
    
for frame in cap :
    
    RGBFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB); #[DISPLAY ONLY] Changes the Frame to RGB for display purposes
    
    croppedFrame = RGBFrame[upLeftY:lowRightY,upLeftX:lowRightX]; #Crops the initial frame to the ROI
    hsvFrame = cv2.cvtColor(croppedFrame, cv2.COLOR_BGR2HSV); #Changes the croppedFrame LUT to HSV for segmentation
    blur = cv2.blur(hsvFrame,(5,5)); #Applies a Gaussian Blur to smoothen the image
    mask = cv2.inRange(blur, params.segmentationParameters["threshMinMouse"], params.segmentationParameters["threshMaxMouse"]); #Thresholds the image to binary
    opening = cv2.morphologyEx(mask,cv2.MORPH_OPEN,params.segmentationParameters["kernel"], iterations = 1); #Applies opening operation to the mask for dot removal
    closing = cv2.morphologyEx(opening,cv2.MORPH_CLOSE,params.segmentationParameters["kernel"], iterations = 1); #Applies closing operation to the mask for large object filling    
    cnts = cv2.findContours(closing.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]; #Finds the contours of the image to identify the meaningful object
    
    cloneCroppedFrame = croppedFrame.copy();
    cv2.drawContours(cloneCroppedFrame,cnts,-1,(255,0,0),3)
    
    cv2.imshow('clone',cloneCroppedFrame);
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break;
        
cv2.destroyAllWindows();

fig = plt.figure(figsize=(20,10))
ax0 = plt.subplot(221)
test = Tracker.croppedFrame.copy();
cv2.drawContours(test,Tracker.cnts,-1,(255,255,0),2)
cv2.drawContours(test,Tracker.cntsCotton,-1,(0,255,255),2)
cv2.circle(test, Tracker.center, 2, (255,0,0), 8);
ax0.imshow(test)   
ax0.set_title("RGB image with overlays of Mouse(Yellow)/Cotton(Cyan) contours")
ax0.set_yticklabels([])
ax0.set_xticklabels([])
ax0.set_yticks([])
ax0.set_xticks([])
ax1 = plt.subplot(223)
ax1.imshow(Tracker.closingMouse) 
ax1.set_title("Mouse binary mask")
ax1.set_yticklabels([])
ax1.set_xticklabels([])
ax1.set_yticks([])
ax1.set_xticks([])
ax2 = plt.subplot(224)
ax2.imshow(Tracker.closingCotton) 
ax2.set_yticklabels([])
ax2.set_xticklabels([])
ax2.set_yticks([])
ax2.set_xticks([])
ax2.set_title("Cotton binary mask")
ax3 = plt.subplot(222)
test2 = Tracker.croppedRegisteredDepth.copy()
cv2.drawContours(test2,Tracker.cntsCotton,-1,(0,255,255),2)
ax3.imshow(test2)
ax3.set_title("Registered depth image with overlays of Cotton contours")
ax3.set_yticklabels([])
ax3.set_xticklabels([])
ax3.set_yticks([])
ax3.set_xticks([])

#plt.savefig("/home/thomas.topilko/Desktop/Tracking_Overlay.png",dpi=200)
    
    
    

