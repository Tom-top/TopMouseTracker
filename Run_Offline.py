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
                  "playSound" : True,
                  };
                  
segmentationParameters = {
                "threshMinRGB" : np.array([0, 0, 0],np.uint8),
                "threshMaxRGB" : np.array([179, 255, 80],np.uint8),
                "threshMinCotton" : np.array([0, 50, 130],np.uint8),
                "threshMaxCotton" : np.array([110, 130, 200],np.uint8),
                "kernel" : np.ones((5,5),np.uint8),
                "minAreaMask" : 1000.0,
                "maxAreaMask" : 8000.0,
                "minDist" : 0.2,
                "minCottonSize" : 300.,
                "nestCottonSize" : 20000.,
                "cageLength" : 50.,
                "cageWidth" : 25.,
                };
        
displayParameters = {
        "showStream" : True,
        };
        
savingParameters = {
        "framerate" : None,
        "fourcc" : cv2.VideoWriter_fourcc(*'MJPG'),
        "saveStream" : True,
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

mainParameters["capturesRGB"], mainParameters["capturesDEPTH"], mainParameters["testFrameRGB"] = IO.VideoLoader(_workingDir,**mainParameters);

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
