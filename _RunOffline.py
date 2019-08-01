#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on Fri Dec  7 13:58:46 2018

@author: tomtop
"""

# Importing libraries#

import os;
import numpy as np;
import cv2;
import matplotlib.pyplot as plt;

# Checking if the TMT is the current working directory#

_topMouseTrackerDir = "/home/thomas.topilko/Documents/GitHub/TopMouseTracker-master/TopMouseTracker/Test"; #Sets path to the TMT directory

if os.getcwd() !=  _topMouseTrackerDir : #If the current working directory is not the TMT directory changes it
    os.chdir(_topMouseTrackerDir)

# Loads TMT modules#

import TopMouseTracker.Parameters as params;
import TopMouseTracker.Utilities as utils;
import TopMouseTracker.IO as IO;
import TopMouseTracker._Tracker as tracker;
import TopMouseTracker.Analysis as analysis;

# Global parameters#

params.mainParameters["mouse"] = "test"; #The number of the mouse to be analyzed
params.mainParameters["rgbVideoName"] = "Raw"; #The prefix of the RGB video to be analyzed
params.mainParameters["depthVideoName"] = "Depth"; #The prefix of the depth video to be analyzed
params.mainParameters["extensionLoad"] = "avi"; #The extension of the video to be analyzed
params.mainParameters["email"] = None; #The email of the user in case email notification is wanted

params.segmentationParameters["cageLength"] = 50; #The length of the segmentation field in cm
params.segmentationParameters["cageWidth"] = 25.; #The width of the segmentation field in cm
params.segmentationParameters["threshMinCotton"] = np.array([0, 0, 150],np.uint8); #The lower parameter for the thresholding of the cotton (hsv)
params.segmentationParameters["threshMaxCotton"] = np.array([140, 62, 250],np.uint8); #The upper parameter for the thresholding of the cotton (hsv)
params.segmentationParameters["showStream"] = False;  #Parameters to display/or not the segmentation in LIVE

params.savingParameters["segmentCotton"] = True; #Parameters to segment/or not the cotton
params.savingParameters["saveStream"] = True; #Parameters to save/or not the full segmentation as a video

params.plotParameters["minDist"] = 0.5; #Parameters to filter out the jitter of the tracked centroid (px)

# Path parameters#

params.mainParameters["workingStation"] = "Black Sabbath"; #Name of the machine on which the code is being run
params.mainParameters["tmtDir"] = _topMouseTrackerDir; #Path to the TMT folder "/mnt/raid/TopMouseTracker"
params.mainParameters["videoInfoFile"] = os.path.join(_topMouseTrackerDir,"Video_Info.xlsx"); #Path to the video info file
params.mainParameters["dataDir"] = os.path.join(params.mainParameters["tmtDir"],"190801"); #Path to the Data folder
params.mainParameters["workingDir"] = [ os.path.join(params.mainParameters["dataDir"],"01-08-2019_9-0-0") ]; #Path(s) to the video(s) folder(s)
params.mainParameters["resultDir"] = os.path.join(params.mainParameters["dataDir"],"{0}".format(params.mainParameters["mouse"])); #Path to the Result folder
params.mainParameters["segmentationDir"] = os.path.join(params.mainParameters["resultDir"],"segmentation"); #Path to the segmentation folder in case sequential frame segmentation was activated

utils.CheckDirectories(); #Check is all the necessary directories were created and asks for the user to clean them if not empty
        
#%%######################################################################################################################################################
# Loading captures/test frames to memory#
#########################################################################################################################################################

params.mainParameters["capturesRGB"], params.mainParameters["capturesDEPTH"],\
 params.mainParameters["testFrameRGB"], params.mainParameters["testFrameDEPTH"] = IO.VideoLoader(params.mainParameters["dataDir"],**params.trackerParameters);
 
#########################################################################################################################################################
# Initializes the TopMouseTracker class#
#########################################################################################################################################################

Tracker = tracker.TopMouseTracker(**params.trackerParameters);

#%%#######################################################################################################################################################
# Creating ROI for analysis#
##########################################################################################################################################################

'''

Click with the left mouse button in the upper-left part of the ROI to select 
and drag the mouse until reaching the lower-right part of the ROI to select then release

Press R to RESET the ROI
Press C to CROP and save the created ROI

'''

Tracker.SetROI();
   
#%%#######################################################################################################################################################
#/!\ [OPTIONAL] Adjusting the segmentation parameters for Animal/Object
##########################################################################################################################################################

Tracker.AdjustThresholding(params.mainParameters["capturesRGB"][0], 10, which='object'); #Parameters : capture, video frame to display

#%%#######################################################################################################################################################
#Launch segmentation on video(s)#
##########################################################################################################################################################  

tracker.TopTracker(Tracker,**params.trackerParameters);

#%%#######################################################################################################################################################
#Complete Tracking plot
##########################################################################################################################################################  

Plot = analysis.Plot(**params.trackerParameters);

params.plotParameters["limit"] = 6.;

params.completeTrackingPlotParameters["cottonSubplots"] = True;
params.completeTrackingPlotParameters["alpha"] = 0.2;

Plot.CompleteTrackingPlot(**params.completeTrackingPlotParameters);

#%%#######################################################################################################################################################
#Nesting Raster plot
##########################################################################################################################################################

Plot.NestingRaster(**params.nestingRasterPlotParameters);

#%%#######################################################################################################################################################
#/!\ [OPTIONAL] Make a live tracking plot
##########################################################################################################################################################  

Plot.LiveTrackingPlot(res=Plot._framerate[0],tStartLivePlot=0,tEndLivePlot=1*60,acceleration=10);