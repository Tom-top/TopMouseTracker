#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
        
segmentationParameters = {
                "resultDir" : _resultDir,
                "workingDir" : _workingDir,
                "mouse" : None,
                "capturesRGB" : None,
                "capturesDEPTH" : None,
                "testFrame" : None,
                "framerate" : None,
                "threshMinMouse" : np.array([0, 0, 0],np.uint8),
                "threshMaxMouse" : np.array([179, 255, 93],np.uint8),
                "kernel" : np.ones((5,5),np.uint8),
                "minAreaMask" : 800.0,
                "maxAreaMask" : 8000.0,
                "minDist" : 3.0,
                "showStream" : False,
                "saveStream" : False,
                "cageLength" : 25.,
                "cageWidth" : 50.,
                "playSound" : True,
                };

#%%###########################################################################
# Loading images to memory#
##############################################################################

segmentationParameters["capturesRGB"], segmentationParameters["capturesDEPTH"], segmentationParameters["testFrame"] = IO.VideoLoader(_workingDir,**segmentationParameters);

#%%###########################################################################
#Initializes the tracker object#
##############################################################################

segmentationParameters["mouse"] = "217";

data = tracker.TopMouseTracker(**segmentationParameters);

#%%############################################################################
#Creating ROI for analysis#
###############################################################################

data.SetROI();    
   
#%%############################################################################
#Launch segmentation on video(s)#
###############################################################################  

tracker.TopTracker(data,**segmentationParameters);

#%%############################################################################
#Save segmentation results
###############################################################################  

tracker.SaveTracking(data,videoDir); 
#tracker.SaveStream(data,videoDir);

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
