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
import matplotlib as mpl;
import moviepy.editor as mpy;
import smtplib;

_topMouseTrackerDir = "/home/thomas.topilko/Documents/GitHub/TopMouseTracker-master"; #Sets path to the TMT directory

if os.getcwd() !=  _topMouseTrackerDir :
    os.chdir(_topMouseTrackerDir)

import TopMouseTracker.Parameters as params;
import TopMouseTracker.Utilities as utils;
import TopMouseTracker.IO as IO;
import TopMouseTracker._Tracker as tracker;
import TopMouseTracker.Analysis as analysis;

Animal = "6"; #Name of the animal to be analyzed

_mainDir = os.path.expanduser("~"); #Sets path to the main directory
_desktopDir = os.path.join(_mainDir,"Desktop"); #Sets path to the Desktop directory
_tmtDir = "/mnt/raid/TopMouseTracker"; #Sets path to the TMT result directory

_dataDir = os.path.join(_tmtDir,"190715"); #Sets path to the data directory

_workingDir = [ os.path.join(_dataDir,"15-7-2019_7-26-38"),\
               os.path.join(_dataDir,"15-7-2019_10-34-59"),\
               os.path.join(_dataDir,"15-7-2019_14-15-41") ]; #Sets path to the working directories

_resultDir = os.path.join(_dataDir,"{0}".format(Animal)); #Sets path to the result directory
_segmentationDir = os.path.join(_resultDir,"segmentation"); #Sets path to the segmentation directory (if the file saving mode was enabled)

utils.CheckDirectoryExists(_tmtDir); #Checks if directory exists
utils.CheckDirectoryExists(_resultDir); #Checks if directory exists
utils.ClearDirectory(_resultDir); #Asks for clearing the directory if not empty

mainParameters = {
                    "workingStation" : "Black Sabbath", #The name of the machine using the script
                    "tmtDir" : _tmtDir, #The TMT directory
                    "dataDir" : _dataDir, #The data directory
                    "workingDir" : _workingDir, #The working directory
                    "resultDir" : _resultDir, #The result directory
                    "segmentationDir" : _segmentationDir, #The segmentation directory
                    "extension" : "avi", #The extension of the movie to be analyzed
                    "testFramePos" : 100, #The position of the frame used for ROI selection
                    "email" : "thomas.topilko@gmail.com", #The email adress to send notifications to
                    "password" : None, #The email password
                    "smtp" : "smtp.gmail.com", #The server smtp
                    "port" : 587, #The server directory
                    "mouse" : Animal, #The name of the animal
                    "capturesRGB" : None, #The RGB captures (list)
                    "capturesDEPTH" : None, #The DEPTH captures (list)
                    "testFrameRGB" : None, #The RGB test frame (for ROI selection)
                    "testFrameDEPTH" : None, #The DEPTH test frame (why not ?!)
                    "playSound" : False, #Parameters to enable ping sound after code finished running
                    "sound2Play" : None, #The sound to be played
                    };
                  
segmentationParameters = {
                    "threshMinMouse" : np.array([100, 70, 0],np.uint8), #Lower parameter for thresholding the mouse (hsv) #"threshMinMouse" : np.array([0, 10, 40],np.uint8),
                    "threshMaxMouse" : np.array([179, 255, 50],np.uint8), #Upper parameter for thresholding the mouse (hsv) #"threshMaxMouse" : np.array([255, 60, 90],np.uint8),
                    "threshMinCotton" : np.array([0, 0, 150],np.uint8), #Lower parameter for thresholding the cotton (hsv)
                    "threshMaxCotton" : np.array([140, 30, 250],np.uint8), #Upper parameter for thresholding the cotton (hsv)
                    "kernel" : np.ones((5,5),np.uint8), #Parameter for the kernel size used in filters
                    "minAreaMask" : 1000.0, #Parameter for minimum size of mouse detection (pixels) #"minAreaMask" : 100.0,
                    "maxAreaMask" : 8000.0, #Parameter for maximum size of mouse detection (pixels) #"maxAreaMask" : 8000.0,
#                    "minDist" : 0.3, #Parameter for minimum distance that the mouse has to travel to be counted (noise filter)
                    "minCottonSize" : 4000., #Parameter for minimum size of cotton detection (pixels)
                    "nestCottonSize" : 15000., #Parameter for maximum size of cotton detection (pixels)
                    "cageLength" : 50., #Length of the cage in cm
                    "cageWidth" : 25., #Width of cage in cm
                    };
        
displayParameters = {
                    "showStream" : False, #Display the tracking in LIVE MODE
                    };

'''   
FOURCC :
XVID : cv2.VideoWriter_fourcc(*'XVID') --> Preferable
MJPG : cv2.VideoWriter_fourcc(*'MJPG') --> Very large videos
X264 : cv2.VideoWriter_fourcc(*'X264') --> Gives small videos
None : skvideo.io.FFmpegWriter --> Default writer (small videos)
"Frame" : Saving video as sequential frames
'''
       
savingParameters = {
        "saveStream" : True, #Whether or not to save the segmentation
        "framerate" : None, #The framerate of the video
        "fourcc" : None, #fourcc to be used for video compression
        "extension" : "avi", #The extension of the video to be saved
        "segmentCotton" : True, #Whether or not to segment cotton in the cage
        "saveCottonMask" : False, #Whether or not to save the cotton mask
        "resizeTracking" : 1., #Resizing factor for video size
        };
        
if savingParameters["fourcc"] == "Frames" :
    
    utils.CheckDirectoryExists(_segmentationDir);  #Check if directory exists
        
plotParameters = {
                "minDist" : 0.5,
                "maxDist" : 10,
                "res" : 1,
                "limit" : 10.,
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
        
#Checks if the emailing mode has been enabled. cf : mainParameters["email"]

if mainParameters["email"] != None :
    
    mainParameters["password"] = input("Type the password for your email {0} : ".format(mainParameters["email"]));
    
    try :
        s = smtplib.SMTP(mainParameters["smtp"], mainParameters["port"]);
        s.ehlo();
        s.starttls();
        s.ehlo();
        s.login(mainParameters["email"], mainParameters["password"]);
        utils.PrintColoredMessage("Emailing mode has been enabled","darkgreen");
        
    except :
        
        utils.PrintColoredMessage("[WARNING] Wrong Username or Password !","darkred");
    
else :
    
    utils.PrintColoredMessage("Emailing mode has been disabled","darkgreen");
        
#%%######################################################################################################################################################
# Loading images to memory#
#########################################################################################################################################################

mainParameters["capturesRGB"], mainParameters["capturesDEPTH"],\
 mainParameters["testFrameRGB"], mainParameters["testFrameDEPTH"] = IO.VideoLoader(_dataDir,**mainParameters);

#########################################################################################################################################################
#Initializes the tracker object#
#########################################################################################################################################################

data = tracker.TopMouseTracker(**trackerParameters);

#%%#######################################################################################################################################################
#Creating ROI for analysis#
##########################################################################################################################################################

data.SetROI();
   
#%%#######################################################################################################################################################
#/!\ [OPTIONAL] Adjusting the segmentation parameters for Mouse/Cotton
##########################################################################################################################################################

data.AdjustThresholding();

#%%#######################################################################################################################################################
#Launch segmentation on video(s)#
##########################################################################################################################################################  

tracker.TopTracker(data,**trackerParameters);

#%%#######################################################################################################################################################
#Complete Tracking plot
##########################################################################################################################################################  

Plot = analysis.Plot(**trackerParameters);

Plot.CompleteTrackingPlot(cBefore='blue',cAfter='red',alpha=0.1, line=True, res=1, rasterSpread=None);

#%%#######################################################################################################################################################
#Nesting Raster plot
##########################################################################################################################################################

peakTresh = 0.7;

if type(peakTresh) == float :
    
    PeakTresh = str(peakTresh)[0]+'-'+str(peakTresh)[2:]

else :
    
    PeakTresh = peakTresh

minDist = 7;

Plot.NestingRaster(cBefore='blue',cAfter='red',res=1, rasterSpread=None, peakThresh = peakTresh, peakDist =1, minDist=minDist);
#plt.savefig(os.path.join(_resultDir,"{0}_{1}_{2}.png".format(Animal,PeakTresh,minDist)))