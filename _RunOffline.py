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

_topMouseTrackerDir = "/home/thomas.topilko/Documents/GitHub/TopMouseTracker-master"; #Sets path to the TMT directory

if os.getcwd() !=  _topMouseTrackerDir :
    os.chdir(_topMouseTrackerDir)

import TopMouseTracker.Parameters as params;
import TopMouseTracker.Utilities as utils;
import TopMouseTracker.IO as IO;
import TopMouseTracker._Tracker as tracker;
import TopMouseTracker.Analysis as analysis;

# Global parameters#

params.mainParameters["mouse"] = "306";
params.mainParameters["videoName"] = "Raw";
params.mainParameters["extensionLoad"] = "avi";
params.mainParameters["email"] = "sigrid.belleville@gmail.com";

params.segmentationParameters["cageLength"] = 50;
params.segmentationParameters["cageWidth"] = 25.;
params.segmentationParameters["threshMinCotton"] = np.array([0, 0, 150],np.uint8);
params.segmentationParameters["threshMaxCotton"] = np.array([140, 50, 250],np.uint8);

params.displayParameters["showStream"] = False; 

params.savingParameters["segmentCotton"] = True;

params.plotParameters["minDist"] = 0.5;

# Path parameters#

params.mainParameters["workingStation"] = "Black Sabbath";
params.mainParameters["tmtDir"] = "/mnt/raid/TopMouseTracker";
params.mainParameters["dataDir"] = os.path.join(params.mainParameters["tmtDir"],"190629");
params.mainParameters["workingDir"] = [ os.path.join(params.mainParameters["dataDir"],'29-6-2019_9-24-39') ];
params.mainParameters["resultDir"] = os.path.join(params.mainParameters["dataDir"],"{0}".format(params.mainParameters["mouse"]));
params.mainParameters["segmentationDir"] = os.path.join(params.mainParameters["resultDir"],"segmentation");

utils.CheckDirectories();
        
#%%######################################################################################################################################################
# Loading images to memory#
#########################################################################################################################################################

params.mainParameters["capturesRGB"], params.mainParameters["capturesDEPTH"],\
 params.mainParameters["testFrameRGB"], params.mainParameters["testFrameDEPTH"] = IO.VideoLoader(params.mainParameters["dataDir"],**params.mainParameters);
 
#########################################################################################################################################################
#Initializes the tracker object#
#########################################################################################################################################################

Tracker = tracker.TopMouseTracker(**params.trackerParameters);

#%%#######################################################################################################################################################
#Creating ROI for analysis#
##########################################################################################################################################################

Tracker.SetROI();
   
#%%#######################################################################################################################################################
#/!\ [OPTIONAL] Adjusting the segmentation parameters for Mouse
##########################################################################################################################################################

Tracker.AdjustThresholdingMouse(params.mainParameters["capturesRGB"][0], 200);

#%%#######################################################################################################################################################
#/!\ [OPTIONAL] Adjusting the segmentation parameters for Cotton
##########################################################################################################################################################

Tracker.AdjustThresholdingCotton(params.mainParameters["capturesRGB"][0], 2000);

#%%#######################################################################################################################################################
#Launch segmentation on video(s)#
##########################################################################################################################################################  

tracker.TopTracker(Tracker,**params.trackerParameters);

#%%#######################################################################################################################################################
#Complete Tracking plot
##########################################################################################################################################################  

Plot = analysis.Plot(**params.trackerParameters);

params.completeTrackingPlotParameters["cottonSubplots"] = False;

Plot.CompleteTrackingPlot(**params.completeTrackingPlotParameters);

#%%#######################################################################################################################################################
#Nesting Raster plot
##########################################################################################################################################################

Plot.NestingRaster(**params.nestingRasterPlotParameters);

#%%#######################################################################################################################################################
#/!\ [OPTIONAL] Make a live tracking plot
##########################################################################################################################################################  

Plot.LiveTrackingPlot(res=1,tStartLivePlot=0,tEndLivePlot=10*60,acceleration=10);