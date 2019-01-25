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

import TopMouseTracker.Settings as settings;
import TopMouseTracker.Parameters as params;
import TopMouseTracker.IO as IO;
import TopMouseTracker.Tracker as tracker;
import TopMouseTracker.Analysis as analysis;
import TopMouseTracker.Utilities as utils;

baseDir = "/Users/tomtop/Desktop/Data/Experiments/Inhibitory_DREADDS/Behavior_Data/";     
workDir = os.path.join(baseDir,"181217-201");                   
videoDir = os.path.join(workDir, "Raw_Data/");
resultDir = os.path.join(videoDir,"Results/");

utils.CheckDirectoryExists(videoDir);
utils.CheckDirectoryExists(resultDir);
        
videoParameters = {
                "encoder" : params.encoders["mpeg4"],
                "framerate" : 30,
                "quality" : 22,
                "runstr" : '{0} -i {1} -o {2} -e {3} -q {4} -r {5} --vfr',
                "handBrakeCLI" : "/Applications/HandBrakeCLI",
                "playSound" : True,
                };
        
segmentationParameters = {
                "baseDir" : baseDir,
                "mouse" : None,
                "captures" : None,
                "testFrame" : None,
                "framerate" : 30,
                "TRESH_MIN" : np.array([0, 0, 0],np.uint8),
                "TRESH_MAX" : np.array([179, 255, 93],np.uint8),
                "kernel" : np.ones((5,5),np.uint8),
                "minAreaMask" : 200.0,
                "minDist" : 3.0,
                "showStream" : False,
                "saveStream" : False,
                "cageLength" : 21.8,
                "cageWidth" : 36.4,
                };
        
#%%############################################################################
###############################################################################
                        #OFFLINE Modules#
###############################################################################
###############################################################################

##############################################################################
#Convert videos to the right format#
##############################################################################
        
IO.VideoConverter(videoDir,**videoParameters);

#%%###########################################################################
# Loading images to memory#
##############################################################################

segmentationParameters["captures"], segmentationParameters["testFrame"] = IO.VideoLoader(videoDir,**videoParameters);

#%%############################################################################
#Creating ROI for analysis#
###############################################################################

segmentationParameters["mouse"] = "201";

data = tracker.TopMouseTracker(**segmentationParameters);

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
        
