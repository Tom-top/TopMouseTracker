#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 15 16:17:57 2019

@author: tomtop
"""

from matplotlib.widgets import MultiCursor

data.depthFrame = cv2.resize(data.DEPTHFrame,(0,0),fx = data._resizingFactorRegistration,fy = data._resizingFactorRegistration)
        
data._H_DEPTH_RESIZED,data._W_DEPTH_RESIZED = data.depthFrame.shape[0],data.depthFrame.shape[1]

start_X = 0
end_X = start_X+data._H_DEPTH_RESIZED
start_Y = 295
end_Y = start_Y+data._W_DEPTH_RESIZED

if end_X >= data._H_RGB :
    end_X_foreground = data._H_RGB-start_X
else :
    end_X_foreground = data._H_RGB
    
if end_Y >= data._W_RGB :
    end_Y_foreground = data._W_RGB-start_Y
else : 
    end_Y_foreground = data._W_RGB

start_X_foreground = 100
start_Y_foreground = 0

if start_X_foreground != 0 :
    end_X = end_X-start_X_foreground
    end_X_foreground = end_X_foreground+start_X_foreground
if start_Y_foreground != 0 :
    end_Y = end_Y-start_Y_foreground
    end_Y_foreground = end_Y_foreground+start_Y_foreground
    
registeredDepth = np.zeros([data._H_RGB,data._W_RGB,3],dtype=np.uint8)

data.blend = cv2.addWeighted(data.depthFrame[start_X_foreground:end_X_foreground,start_Y_foreground:end_Y_foreground,:],
                1,
                registeredDepth[start_X:end_X,start_Y:end_Y,:],
                0,
                0,
                registeredDepth)
                        
registeredDepth[start_X:end_X,start_Y:end_Y,:] = data.blend

croppedFrameDEPTH = registeredDepth[data.upLeftY:data.lowRightY,data.upLeftX:data.lowRightX] #Crops the initial frame to the ROI
cv2.drawContours(croppedFrameDEPTH,data.cntsCotton,-1,(255,0,0))

fig = plt.figure(figsize=(20,10))
ax1 = plt.subplot(121)
ax1.imshow(data.croppedFrame)
ax2 = plt.subplot(122,sharex=ax1,sharey=ax1)
ax2.imshow(croppedFrameDEPTH)
multi = MultiCursor(fig.canvas, (ax1, ax2), horizOn=True, vertOn=True, color='r', lw=1)
