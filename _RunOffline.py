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
import TopMouseTracker._Tracker as tracker;
import TopMouseTracker.Analysis as analysis;

_mainDir = os.path.expanduser("~");
_desktopDir = os.path.join(_mainDir,"Desktop");
_resultDir = os.path.join(_mainDir,"TopMouseTracker");
_workingDir = os.path.join(_resultDir,"14-2-2019_9-27-36");

utils.CheckDirectoryExists(_resultDir);

mainParameters = {"resultDir" : _resultDir,
                  "workingDir" : _workingDir,
                  "mouse" : None,
                  "capturesRGB" : None,
                  "capturesDEPTH" : None,
                  "testFrameRGB" : None,
                  "testFrameDEPTH" : None,
                  "playSound" : False,
                  };
                  
segmentationParameters = {
                "threshMinMouse" : np.array([100, 70, 0],np.uint8),
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
        "segmentCotton" : False,
        "saveStream" : True,
        "saveCottonMask" : False,
        "resizeTracking" : 4.,
        };
        
plotParameters = {
                "minDist" : 0.5,
                "maxDist" : 10,
                "res" : 1,
                "limit" : 6.,
                "gridsize" : 100,
                "save" : True,
                };
        
trackerParameters = {
        "main" : mainParameters,
        "segmentation" : segmentationParameters,
        "display" : displayParameters,
        "saving" : savingParameters,
        "plot" : plotParameters,
        };
        
#%%###########################################################################
# Loading images to memory#
##############################################################################

mainParameters["capturesRGB"], mainParameters["capturesDEPTH"],\
 mainParameters["testFrameRGB"], mainParameters["testFrameDEPTH"] = IO.VideoLoader(_workingDir,**mainParameters);

#%%###########################################################################
#Initializes the tracker object#
##############################################################################

mainParameters["mouse"] = "1";

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
#Plotting and Analysis
###############################################################################  
     
Plot = analysis.Plot(**trackerParameters);

Plot.CompleteTrackingPlot(cBefore='b',cAfter='r',alpha=0.1, line=True);

#Plot.HeatMapPlot(plotParameters["gridsize"]);
