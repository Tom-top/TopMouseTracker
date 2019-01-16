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
import TopMouseTracker.Params as params;
import TopMouseTracker.IO as IO;
import TopMouseTracker.Tracker as tracker;
import TopMouseTracker.Analysis as analysis;

baseDir = "Enter the path to your Directory here";     
workDir = os.path.join(baseDir,"Name of the experiment");                   
videoDir = os.path.join(workDir, "Raw_Data/");
resultDir = os.path.join(videoDir,"Results/");

if not os.path.exists(videoDir) :
    os.mkdir(videoDir);

if not os.path.exists(resultDir) :
    os.mkdir(resultDir);
        
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
                "cageLength" : 36.4,
                "cageWidth" : 21.8,
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

segmentationParameters["mouse"] = "204";

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

LinePlotParameters = {
                "baseDir" : segmentationParameters["baseDir"],
                "directory" : videoDir+'Results',
                "mouse" : "195bis",
                "cageLength" : 36.4,
                "cageWidth" : 21.8,
                "minDist" : 0.2,
                "maxDist" : 1000,
                "framerate" : segmentationParameters["framerate"],
                };
        
Plot = analysis.Plot(**LinePlotParameters);

#analysis.CheckTracking();
analysis.TrackingPlot(100);
        
