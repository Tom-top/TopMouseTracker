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

import TopMouseTracker.Parameters as params;
import TopMouseTracker.Utilities as utils;
import TopMouseTracker.IO as IO;
import TopMouseTracker._Tracker as tracker;
import TopMouseTracker.Analysis as analysis;

_mainDir = os.path.expanduser("~");
_desktopDir = os.path.join(_mainDir,"Desktop");
_tmtDir = "/mnt/raid/TopMouseTracker";
_workingDir = os.path.join(_tmtDir,"190222-01/22-2-2019_9-30-28");
_resultDir = os.path.join(_workingDir,"247");#Results-252

utils.CheckDirectoryExists(_tmtDir);
utils.CheckDirectoryExists(_resultDir);
utils.ClearDirectory(_resultDir);

mainParameters = {"tmtDir" : _tmtDir,
                  "workingDir" : _workingDir,
                  "resultDir" : _resultDir,
                  "mouse" : None,
                  "capturesRGB" : None,
                  "capturesDEPTH" : None,
                  "testFrameRGB" : None,
                  "testFrameDEPTH" : None,
                  "playSound" : False,
                  "sound2Play" : None,
                  };
                  
segmentationParameters = {
                "threshMinMouse" : np.array([100, 70, 0],np.uint8),
                "threshMaxMouse" : np.array([179, 255, 50],np.uint8),
                "threshMinCotton" : np.array([0, 20, 150],np.uint8),
                #"threshMaxCotton" : np.array([120, 120, 250],np.uint8), #Upper Side
                "threshMaxCotton" : np.array([120, 90, 250],np.uint8), #Lower Side
                "kernel" : np.ones((5,5),np.uint8),
                "minAreaMask" : 1000.0,
                "maxAreaMask" : 8000.0,
                "minDist" : 0.3,
                "minCottonSize" : 4000.,
                "nestCottonSize" : 15000.,
                "cageLength" : 50.,
                "cageWidth" : 25.,
                };
        
displayParameters = {
        "showStream" : False,
        };

'''   
FOURCC :
XVID : cv2.VideoWriter_fourcc(*'XVID') --> Preferable
MJPG : cv2.VideoWriter_fourcc(*'MJPG') --> Very large videos
X264 : cv2.VideoWriter_fourcc(*'X264') --> Gives small videos
'''
       
savingParameters = {
        "framerate" : None,
        "fourcc" : None, #cv2.VideoWriter_fourcc(*'XVID')
        "extension" : "avi",
        "segmentCotton" : True,
        "saveStream" : True,
        "saveCottonMask" : False,
        "resizeTracking" : 4.,
        };
        
plotParameters = {
                "minDist" : 0.5,
                "maxDist" : 10,
                "res" : 1,
                "limit" : 6.,
                "gridsize" : 200,
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

mainParameters["mouse"] = "247";

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
     
mainParameters["mouse"] = "215"

Plot = analysis.Plot(**trackerParameters);

Plot.CompleteTrackingPlot(cBefore='blue',cAfter='red',alpha=0.1, line=True, res=1, rasterSpread=100);

#Plot.HeatMapPlot(bins=1000,sigma=6);
