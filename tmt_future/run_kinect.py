#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 13:58:46 2018

@author: tomtop
"""

import os

import numpy as np

import cv2

from pykinect2 import PyKinectV2, PyKinectRuntime

import TopMouseTracker.tracker_kinect as tracker
import TopMouseTracker.analysis as analysis
import TopMouseTracker.utilities as utils

_mainDir = os.path.expanduser("~");
_desktopDir = os.path.join(_mainDir,"Desktop");
_resultDir = os.path.join(_mainDir,"TopMouseTracker");

utils.CheckDirectoryExists(_resultDir);
                 
mainParameters = {"resultDir" : _resultDir,
                  "mice" : ["1","2"], #Name of the mice that are tracked
                  "mode" : "light",
                  "resizeROI": 2., #Resizing factor for screen display
                  "resizeDisplay" : 2.,
                  "kinectRGB" : None,
                  "kinectDEPTH" : None,
                  "rawFrameRGB" : None,
                  "rawFrameDEPTH" : None,
                  "testFrameRGB" : None,
                  "testFrameDEPTH" : None,
                  };
        
segmentationParameters = {
                "mouse" : None,
                "threshMinRGB" : np.array([0, 0, 0],np.uint8),
                "threshMaxRGB" : np.array([179, 255, 93],np.uint8),
                "kernel" : np.ones((5,5),np.uint8),
                "minAreaMask" : 800.0/mainParameters["resize_ROI"],
                "maxAreaMask" : 8000.0/mainParameters["resize_ROI"],
                "minDist" : 3.0,
                "cageLength" : 25.,
                "cageWidth" : 50.,
                };
        
displayParameters = {
        "mode" : "v",
        "showStream" : False,
        };
        
savingParameters = {
        "framerate" : 15,
        "fourcc" : cv2.VideoWriter_fourcc(*'MJPG'),
        "mode" : "v",
        "rawVideoFileName" : "raw_video_mouse",
        "segVideoFileName" : "seg_video_mouse",
        "rawWriter" : None,
        "segWriter" : None,
        "saveVideo" : False,
        };
        
trackerParameters = {
        "main" : mainParameters,
        "display" : displayParameters,
        "saving" : savingParameters,
        "segmentation" : segmentationParameters,
        };

#%%###########################################################################
#Setting up cameras#
##############################################################################

mainParameters["kinectRGB"] = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body); #Initializes the RGB camera
mainParameters["kinectDEPTH"] = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth); #Initializes the DEPTH camera

mainParameters["rawFrameRGB"] = tracker.getColorFrame(mainParameters["kinectRGB"],mainParameters["resizeDisplay"]); #Imports a test frame from RGB camera in full resolution
#mainParameters["rawFrameDEPTH"] = tracker.getDepthFrame(_kinectDEPTH,1); #Imports a test frame from DEPTH camera in full resolution

mainParameters["testFrameDEPTH"] = tracker.getDepthFrame(mainParameters["kinectDEPTH"],mainParameters["resizeROI"]); #Imports a test RESIZED test frame from RGB camera
mainParameters["testFrameRGB"] = tracker.getColorFrame(mainParameters["kinectRGB"],mainParameters["resizeROI"]); #Imports a test RESIZED test frame from DEPTH camera
        
#%%############################################################################
#Creating ROI for analysis#
###############################################################################

data = {};

for mouse in mainParameters["mice"] :
    
    segmentationParameters["mouse"] = mouse;
    
    data["{0}".format(mouse)] = tracker.Tracker(mode=mainParameters["light"],**trackerParameters);
    data["{0}".format(mouse)].SetROI();
    
    cv2.destroyAllWindows();

#%%############################################################################
#Launching segmentation#
############################################################################### 
 
tracker.KinectTopMouseTracker(data,**segmentationParameters);

#%%############################################################################
#Plotting and Analysis
###############################################################################  

plotParameters = {
                "baseDir" : segmentationParameters["baseDir"],
                "resultDir" : mainParameters["resultDir"],
                "mouse" : "201",
                "cageLength" : segmentationParameters["cageLength"],
                "cageWidth" : segmentationParameters["cageWidth"],
                "minDist" : 0.5,
                "maxDist" : 10,
                "framerate" : savingParameters["framerate"],
                "gridsize" : 100,
                };
        
Plot = analysis.Plot(**plotParameters);

res = 1;

#Plot.CheckTracking();
Plot.CompleteTrackingPlot(res,limit=6,save=True);
#Plot.TrackingPlot(res,limit=6);
#Plot.HeatMapPlot(PlotParameters["gridsize"]);
