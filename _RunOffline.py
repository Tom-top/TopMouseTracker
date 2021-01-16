#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on Fri Dec  7 13:58:46 2018

@author: tomtop
"""

# Importing libraries#

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib import cm
from twilio.rest import Client

# Checking if the TMT is the current working directory#

_topMouseTrackerDir = "/home/thomas.topilko/Documents/" #Sets path to the TMT directory

if os.getcwd() !=  _topMouseTrackerDir : #If the current working directory is not the TMT directory changes it
    os.chdir(_topMouseTrackerDir)

# Loads TMT modules#

import TopMouseTracker.parameters as params
import TopMouseTracker.utilities as utils
import TopMouseTracker.tmt_io as IO
import TopMouseTracker._Tracker as tracker
import TopMouseTracker.Analysis as analysis
# Global parameters#

Mice = ["68", "69", "70", "72", "74", "77", "78" , "75", "82", "76"]
#Mice = ["76"]

Folders = ["68_69_70", "68_69_70", "68_69_70",\
           "72", "74_77_78", "74_77_78", "74_77_78",\
           "75_82", "75_82", "76"]
#Folders = ["76"]

refPt = [[(358, 111), (570, 223)], [(352, 261), (574, 375)], [(101, 122), (307, 231)],\
        [(346, 94), (566, 200)], [(346, 113), (571, 220)], [(346, 255), (568, 365)],\
        [(97, 117), (298, 219)], [(385, 130), (597, 244)], [(390, 278), (598, 381)],\
        [(418, 111), (620, 221)]]
#refPt = [[(601, 137), (1441, 538)]]

maxMouse = [np.array([255, 255, 105],np.uint8), np.array([255, 255, 105],np.uint8), np.array([255, 255, 105],np.uint8),\
            np.array([255, 255, 80],np.uint8), np.array([255, 255, 105],np.uint8), np.array([255, 255, 105],np.uint8),\
            np.array([255, 255, 105],np.uint8), np.array([255, 255, 85],np.uint8), np.array([255, 255, 85],np.uint8),\
            np.array([255, 255, 85],np.uint8)]
#maxMouse = [np.array([255, 255, 60],np.uint8)]

all_params = [len(i) for i in [Mice, Folders, refPt, maxMouse]]

if not all_params.count(all_params[0]) == len(all_params) :
    raise RuntimeError("One of the preset parameters is missing some data! {0}".format(all_params))

for m, f, r, v in zip(Mice, Folders, refPt, maxMouse) :

    params.mainParameters["mouse"] = m #The number of the mouse to be analyzed
    params.mainParameters["rgbVideoName"] = "Raw" #The prefix of the RGB video to be analyzed
    params.mainParameters["depthVideoName"] = "Depth" #The prefix of the depth video to be analyzed
    params.mainParameters["extensionLoad"] = "avi" #The extension of the video to be analyzed
    params.mainParameters["email"] = None #The email of the user in case email notification is wanted
    
    params.segmentationParameters["cageLength"] = 37. #The length of the segmentation field in cm 50
    params.segmentationParameters["cageWidth"] = 20. #The width of the segmentation field in cm 25
    params.segmentationParameters["threshMinCotton"] = np.array([0, 0, 0],np.uint8) #The lower parameter for the thresholding of the cotton (hsv)
    params.segmentationParameters["threshMaxCotton"] = np.array([255, 98, 255],np.uint8) #The upper parameter for the thresholding of the cotton (hsv)
    params.segmentationParameters["threshMaxMouse"] = v
    params.segmentationParameters["registrationX"] = 306  #Parameters for the X shift registration
    params.segmentationParameters["registrationY"] = 85  #Parameters for the Y shift registration
    params.segmentationParameters["showStream"] = False  #Parameters to display/or not the segmentation in LIVE
    params.segmentationParameters["resize_stream"] = 1/4 #Parameters to resize the segmentation in LIVE
    
    params.savingParameters["segmentCotton"] = False #Parameters to segment/or not the cotton
    params.savingParameters["saveStream"] = True #Parameters to save/or not the full segmentation as a video
    
    params.plotParameters["minDist"] = 0.5 #Parameters to filter out the jitter of the tracked centroid (px)
    
    # Path parameters#
    
    params.mainParameters["worckingStation"] = "Black Sabbath"; #Name of the machine on which the    code is being run
    params.mainParameters["tmtDir"] = "/home/thomas.topilko/Desktop/Tracking" #Path to the TMT folder "/mnt/raid/TopMouseTracker"
    params.mainParameters["videoInfoFile"] = os.path.join(params.mainParameters["tmtDir"],"Video_Info.xlsx") #Path to the video info file
    params.mainParameters["dataDir"] = os.path.join(params.mainParameters["tmtDir"],f) #Path to the Data folder
    params.mainParameters["workingDir"] = [ os.path.join(params.mainParameters["tmtDir"],f) ] #Path(s) to the video(s) folder(s)
    params.mainParameters["resultDir"] = os.path.join(params.mainParameters["dataDir"],"{0}".format(params.mainParameters["mouse"])) #Path to the Result folder
    params.mainParameters["segmentationDir"] = os.path.join(params.mainParameters["resultDir"],"segmentation") #Path to the segmentation folder in case sequential frame segmentation was activated
    
    utils.CheckDirectories(); #Check is all the necessary directories were created and asks for the user to clean them if not empty
        
    #######################################################################################################################################################
    # Loading captures/test frames to memory#
    #########################################################################################################################################################
    
#    params.mainParameters["capturesRGB"], params.mainParameters["capturesDEPTH"],\
#     params.mainParameters["testFrameRGB"], params.mainParameters["testFrameDEPTH"] = IO.VideoLoader(params.mainParameters["dataDir"], in_folder=True, **params.trackerParameters)
    
    params.mainParameters["testFrameRGB"], params.mainParameters["capturesRGB"], clips = IO.concatenate_video_clips(os.path.join(params.mainParameters["tmtDir"], f))
     
    #########################################################################################################################################################
    # Initializes the TopMouseTracker class#
    #########################################################################################################################################################
    
    Tracker = tracker.TopMouseTracker(**params.trackerParameters)

    ########################################################################################################################################################
    # Creating ROI for analysis#
    ##########################################################################################################################################################
    
    '''
    
    Click with the left mouse button in the upper-left part of the ROI to select 
    and drag the mouse until reaching the lower-right part of the ROI to select then release
    
    Press R to RESET the ROI
    Press C to CROP and save the created ROI
    
    '''
    
    Tracker.SetROI(refPt=r) #refPt=r
#    print(Tracker._refPt)
#    break
   
    ##%%#######################################################################################################################################################
    ##/!\ [OPTIONAL] Adjusting the segmentation parameters for Animal/Object
    #######################################################################################q####################################################################
    #
#    params.segmentationParameters["threshMaxMouse"] = np.array([255, 255, 85],np.uint8); #The upper parameter for the thresholding of the cotton (hsv)
#    
#    Tracker.AdjustThresholding(params.mainParameters["capturesRGB"][0], 2000, which='animal'); #Parameters : capture, video frame to display
  
    ##%%#######################################################################################################################################################
    ##/!\ [OPTIONAL] Adjusting the registration parameters
    ########################################q###################################################################################################################
    #
    #Tracker._args["segmentation"]["registrationX"] = 306;  #Parameters for the X shift registration 314 306
    #Tracker._args["segmentation"]["registrationY"] = 85;  #Parameters for the Y shift registration 97 85
    #
    #Tracker.AdjustRegistration(params.mainParameters["capturesRGB"][0], params.mainParameters["capturesDEPTH"][0], 1000)

    ########################################################################################################################################################
    #Launch segmentation on video(s)#
    ##########################################################################################################################################################  

    tracker.TopTracker(Tracker,**params.trackerParameters)

#%%#######################################################################################################################################################
#Complete Tracking plot
##########################################################################################################################################################  

params.plotParameters["limit"] = 8.

params.completeTrackingPlotParameters["cottonSubplots"] = True
params.completeTrackingPlotParameters["alpha"] = 0.5

cmap = cm.coolwarm
params.completeTrackingPlotParameters["cBefore"] = cmap(0.) #cmap(0.3)
params.completeTrackingPlotParameters["cAfter"] = cmap(1.) #cmap(0.7)

Plot = analysis.Plot(**params.trackerParameters)
Plot.GeneratePeakFile(**params.nestingRasterPlotParameters)

Plot.CompleteTrackingPlot(**params.completeTrackingPlotParameters)

#%%#######################################################################################################################################################
#Nesting Raster plot
##########################################################################################################################################################

Plot.NestingRaster(**params.nestingRasterPlotParameters)

#%%#######################################################################################################################################################
#/!\ [OPTIONAL] Make a live tracking plot
##########################################################################################################################################################  

Plot.LiveTrackingPlot(res=Plot._framerate[0],tStartLivePlot=19*60,tEndLivePlot=20*60,acceleration=10)