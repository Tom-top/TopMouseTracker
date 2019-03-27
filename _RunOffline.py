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
import smtplib;

import TopMouseTracker.Parameters as params;
import TopMouseTracker.Utilities as utils;
import TopMouseTracker.IO as IO;
import TopMouseTracker._Tracker as tracker;
import TopMouseTracker.Analysis as analysis;

_mainDir = os.path.expanduser("~");
_desktopDir = os.path.join(_mainDir,"Desktop");
_tmtDir = "/mnt/raid/TopMouseTracker";
_workingDir = os.path.join(_tmtDir,"190220-01/20-2-2019_10-5-34");
_resultDir = os.path.join(_workingDir,"254");#Results-252

utils.CheckDirectoryExists(_tmtDir);
utils.CheckDirectoryExists(_resultDir);
utils.ClearDirectory(_resultDir);

mainParameters = {"workingStation" : "Black Sabbath",
                  "tmtDir" : _tmtDir,
                  "workingDir" : _workingDir,
                  "resultDir" : _resultDir,
                  "email" : "thomas.topilko@gmail.com",
                  "password" : None,
                  "smtp" : "smtp.gmail.com",
                  "port" : 587, #587 TLS, 465 SSL
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
                "threshMaxCotton" : np.array([120, 120, 250],np.uint8), #Upper Side
                #"threshMaxCotton" : np.array([120, 90, 250],np.uint8), #Lower Side
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
        "fourcc" : cv2.VideoWriter_fourcc(*'MJPG'), #cv2.VideoWriter_fourcc(*'XVID')
        "extension" : "avi",
        "segmentCotton" : True,
        "saveStream" : True,
        "saveCottonMask" : False,
        "resizeTracking" : 1.,
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
        "server" : None,
        };

if mainParameters["email"] != None :
    
    mainParameters["password"] = input("Type the password for your email : {0}".format(mainParameters["email"]));
    utils.PrintColoredMessage("Emailing mode has been enabled","darkgreen");
    
    try :
        
        server = smtplib.SMTP_SSL(mainParameters["smtp"], mainParameters["port"]);
        server.starttls();
        server.login(mainParameters["email"], mainParameters["password"]);
        trackerParameters["server"] = server;
        utils.PrintColoredMessage("Successfully connected to {0}".format(mainParameters["email"]),"darkgreen");
        
    except :
        
        utils.PrintColoredMessage("Failed to connect to {0}".format(mainParameters["email"]),"darkred");
    
else :
    
    utils.PrintColoredMessage("Emailing mode has been disabled","darkgreen");
    

        
#%%###########################################################################
# Loading images to memory#
##############################################################################

mainParameters["capturesRGB"], mainParameters["capturesDEPTH"],\
 mainParameters["testFrameRGB"], mainParameters["testFrameDEPTH"] = IO.VideoLoader(_workingDir,**mainParameters);

#%%###########################################################################
#Initializes the tracker object#
##############################################################################

mainParameters["mouse"] = "254";

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
     
mainParameters["mouse"] = "254"

Plot = analysis.Plot(**trackerParameters);

Plot.CompleteTrackingPlot(cBefore='blue',cAfter='red',alpha=0.1, line=True, res=1, rasterSpread=100);

#Plot.HeatMapPlot(bins=1000,sigma=6);
