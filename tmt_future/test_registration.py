#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 16:49:17 2019

@author: thomas.topilko
"""

Tracker.start_X = 0;
Tracker.end_X = Tracker.start_X+Tracker._H_DEPTH_RESIZED;
Tracker.start_Y = 295; #292
Tracker.end_Y = Tracker.start_Y+Tracker._W_DEPTH_RESIZED;

if Tracker.end_X >= Tracker._H_RGB :
    
    Tracker.end_X_foreground = Tracker._H_RGB-Tracker.start_X;
else :
    
    Tracker.end_X_foreground = Tracker._H_RGB;
    
if Tracker.end_Y >= Tracker._W_RGB :
    
    Tracker.end_Y_foreground = Tracker._W_RGB-Tracker.start_Y;
else : 
    
    Tracker.end_Y_foreground = Tracker._W_RGB;

Tracker.start_X_foreground = 110; #97
Tracker.start_Y_foreground = 0;

if Tracker.start_X_foreground != 0 :
    
    Tracker.end_X = Tracker.end_X-Tracker.start_X_foreground;
    Tracker.end_X_foreground = Tracker.end_X_foreground+Tracker.start_X_foreground;
    
if Tracker.start_Y_foreground != 0 :
    
    Tracker.end_Y = Tracker.end_Y-Tracker.start_Y_foreground;
    Tracker.end_Y_foreground = Tracker.end_Y_foreground+Tracker.start_Y_foreground;F
    


Tracker.depthFrame = cv2.resize(Tracker.DEPTHFrame,(0,0), fx=Tracker._resizingFactorRegistration, fy=Tracker._resizingFactorRegistration);
    
registeredDepth = np.zeros([Tracker._H_RGB,Tracker._W_RGB,3],dtype=np.uint8);

Tracker.blend = cv2.addWeighted(Tracker.depthFrame[Tracker.start_X_foreground:Tracker.end_X_foreground,Tracker.start_Y_foreground:Tracker.end_Y_foreground,:],
                1,
                registeredDepth[Tracker.start_X:Tracker.end_X,Tracker.start_Y:Tracker.end_Y,:],
                0,
                0,
                registeredDepth);

registeredDepth[Tracker.start_X:Tracker.end_X,Tracker.start_Y:Tracker.end_Y,:] = Tracker.blend;


threshMinCotton = np.array([0, 0, 150],np.uint8); #The lower parameter for the thresholding of the cotton (hsv)
threshMaxCotton = np.array([140, 55, 250],np.uint8); #The upper parameter for the thresholding of the cotton (hsv)

hsvFrame = cv2.cvtColor(Tracker.RGBFrame, cv2.COLOR_BGR2HSV); #Changes the croppedFrame LUT to HSV for segmentation
blur = cv2.blur(hsvFrame,(5,5)); #Applies a Gaussian Blur to smoothen the image
maskCotton = cv2.inRange(blur, threshMinCotton, threshMaxCotton);
cntsCotton = cv2.findContours(maskCotton.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]; #Finds the contours of the image to identify the meaningful object

cv2.drawContours(registeredDepth, cntsCotton, -1, (0,255,0), Tracker.contourThickness);

#plt.imshow(maskCotton)

cntsCotton = cv2.findContours(maskCotton.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2];
croppedRegisteredDepth = registeredDepth[Tracker.upLeftY:Tracker.lowRightY,Tracker.upLeftX:Tracker.lowRightX];
#cv2.drawContours(croppedRegisteredDepth, Tracker.cntsCottonFiltered, -1, (0,255,0), Tracker.contourThickness);

#plt.imshow(croppedRegisteredDepth)



fig =plt.figure()
ax0 = plt.subplot(121)
ax1 = plt.subplot(122,sharex=ax0,sharey=ax0)
multi = MultiCursor(fig.canvas, (ax0, ax1), color='r', lw=2, horizOn=True, vertOn=True);

ax0.imshow(Tracker.RGBFrame)
ax1.imshow(registeredDepth)
plt.show()
