#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on Fri Dec  7 13:58:46 2018

@author: tomtop
"""

import os;
import numpy as np;
import cv2;
import matplotlib.pyplot as plt;

import TopMouseTracker.Utilities as utils;
import TopMouseTracker.IO as IO;
import TopMouseTracker.Tracker as tracker;

_mainDir = os.path.expanduser("~");
_desktopDir = os.path.join(_mainDir,"Desktop");
_resultDir = os.path.join(_mainDir,"TopMouseTracker");
_workingDir = os.path.join(_resultDir,"190207-01");

utils.CheckDirectoryExists(_resultDir);

mainParameters = {"resultDir" : _resultDir,
                  "workingDir" : _workingDir,
                  "mouse" : None,
                  "capturesRGB" : None,
                  "capturesDEPTH" : None,
                  "testFrameRGB" : None,
                  "testFrameDEPTH" : None,
                  "playSound" : True,
                  };
                  
segmentationParameters = {
                "threshMinMouse" : np.array([0, 0, 0],np.uint8),
                "threshMaxMouse" : np.array([179, 255, 50],np.uint8),
                "threshMinCotton" : np.array([0, 20, 150],np.uint8),
                "threshMaxCotton" : np.array([110, 105, 250],np.uint8),
                "kernel" : np.ones((5,5),np.uint8),
                "minAreaMask" : 1000.0,
                "maxAreaMask" : 8000.0,
                "minDist" : 0.3,
                "minCottonSize" : 1000.,
                "nestCottonSize" : 20000.,
                "cageLength" : 50.,
                "cageWidth" : 25.,
                };
        
displayParameters = {
        "showStream" : False,
        };
        
savingParameters = {
        "framerate" : None,
        "fourcc" : cv2.VideoWriter_fourcc(*'MJPG'),
        "saveStream" : True,
        "saveCottonMask" : True,
        "resizeTracking" : 4.,
        };
        
trackerParameters = {
        "main" : mainParameters,
        "segmentation" : segmentationParameters,
        "display" : displayParameters,
        "saving" : savingParameters,
        };
        
#%%###########################################################################
# Loading images to memory#
##############################################################################

mainParameters["capturesRGB"], mainParameters["capturesDEPTH"],\
 mainParameters["testFrameRGB"], mainParameters["testFrameDEPTH"] = IO.VideoLoader(_workingDir,**mainParameters);

#%%###########################################################################
#Initializes the tracker object#
##############################################################################

mainParameters["mouse"] = "217";

data = tracker.TopMouseTracker(**trackerParameters);

#%%############################################################################
#Creating ROI for analysis#
###############################################################################

data.SetROI();    
   
#%%############################################################################
#Launch segmentation on video(s)#
###############################################################################  

tracker.TopTracker(data,**trackerParameters);

#%%############################################################################
#Save segmentation results
###############################################################################  

tracker.SaveTracking(data,_workingDir); 

#%%############################################################################
#Plotting and Analysis
###############################################################################  

workDir = os.path.join(baseDir,"181217-201");                   
videoDir = os.path.join(workDir, "Raw_Data/");
resultDir = os.path.join(videoDir,"Results/");

PlotParameters = {
                "baseDir" : segmentationParameters["baseDir"],
                "directory" : videoDir+'Results',
                "mouse" : "201",
                "cageLength" : 21.8,
                "cageWidth" : 36.4,
                "minDist" : 0.5,
                "maxDist" : 10,
                "framerate" : segmentationParameters["framerate"],
                "gridsize" : 100,
                };
        
Plot = analysis.Plot(**PlotParameters);

res = 1;

#Plot.CheckTracking();
Plot.CompleteTrackingPlot(res,limit=6,save=True);
#Plot.TrackingPlot(res,limit=6);
#Plot.HeatMapPlot(PlotParameters["gridsize"]);

#%%


from matplotlib.widgets import MultiCursor

data.depthFrame = cv2.resize(data.DEPTHFrame,(0,0),fx = data._resizingFactorRegistration,fy = data._resizingFactorRegistration)
        
data._H_DEPTH_RESIZED,data._W_DEPTH_RESIZED = data.depthFrame.shape[0],data.depthFrame.shape[1]

start_X = 0;
end_X = start_X+data._H_DEPTH_RESIZED;
start_Y = 295;
end_Y = start_Y+data._W_DEPTH_RESIZED;

if end_X >= data._H_RGB :
    end_X_foreground = data._H_RGB-start_X;
else :
    end_X_foreground = data._H_RGB;
    
if end_Y >= data._W_RGB :
    end_Y_foreground = data._W_RGB-start_Y;
else : 
    end_Y_foreground = data._W_RGB;

start_X_foreground = 100;
start_Y_foreground = 0;

if start_X_foreground != 0 :
    end_X = end_X-start_X_foreground
    end_X_foreground = end_X_foreground+start_X_foreground
if start_Y_foreground != 0 :
    end_Y = end_Y-start_Y_foreground
    end_Y_foreground = end_Y_foreground+start_Y_foreground
    
registeredDepth = np.zeros([data._H_RGB,data._W_RGB,3],dtype=np.uint8);

data.blend = cv2.addWeighted(data.depthFrame[start_X_foreground:end_X_foreground,start_Y_foreground:end_Y_foreground,:],
                1,
                registeredDepth[start_X:end_X,start_Y:end_Y,:],
                0,
                0,
                registeredDepth);
                        
registeredDepth[start_X:end_X,start_Y:end_Y,:] = data.blend; 

croppedFrameDEPTH = registeredDepth[data.upLeftY:data.lowRightY,data.upLeftX:data.lowRightX]; #Crops the initial frame to the ROI
cv2.drawContours(croppedFrameDEPTH,data.cntsCotton,-1,(255,0,0))

fig = plt.figure(figsize=(20,10))
ax1 = plt.subplot(121)
ax1.imshow(data.croppedFrame)
ax2 = plt.subplot(122,sharex=ax1,sharey=ax1)
ax2.imshow(croppedFrameDEPTH)
multi = MultiCursor(fig.canvas, (ax1, ax2), horizOn=True, vertOn=True, color='r', lw=1)
