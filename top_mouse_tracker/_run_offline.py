#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on Fri Dec  7 13:58:46 2018

@author: tomtop
"""

import os

from matplotlib import cm

import top_mouse_tracker.parameters as params
import top_mouse_tracker.utilities as utils
from top_mouse_tracker import tmt_io
from top_mouse_tracker import analysis
import top_mouse_tracker._tracker as tracker

f = 'test'
params.main["mouse"] = 'test'  # The number of the mouse to be analyzed
params.main["rgbVideoName"] = "RGB"  # The prefix of the RGB video to be analyzed
params.main["depthVideoName"] = "DEPTH"  # The prefix of the depth video to be analyzed
params.main['email'] = None  # Skip sending emails

params.segmentation["showStream"] = False   # Parameters to display/or not the segmentation in LIVE
params.segmentation["resize_stream"] = 1 / 4  # Parameters to resize the segmentation in LIVE

params.saving["segmentCotton"] = False  # Parameters to segment/or not the cotton
params.saving["saveStream"] = True  # Parameters to save/or not the full segmentation as a video

params.plot["minDist"] = 0.5  # Parameters to filter out the jitter of the tracked centroid (px)

# Path parameters #

tmt_dir = os.path.expanduser("~/own_cloud/video_data_test")  # Path to the TMT folder "/mnt/raid/TopMouseTracker"
params.main["tmtDir"] = tmt_dir
params.main["videoInfoFile"] = os.path.join(tmt_dir, "Video_Info.xlsx")  # Path to the video info file
params.main["dataDir"] = os.path.join(tmt_dir, f)  # Path to the Data folder
params.main["workingDir"] = [os.path.join(tmt_dir, f)]  # Path(s) to the video(s) folder(s)
params.main["resultDir"] = os.path.join(params.main["dataDir"],
                                                  "{0}".format(params.main["mouse"]))  # Path to the Result folder
params.main["segmentationDir"] = os.path.join(params.main["resultDir"], "segmentation")  # Path to the segmentation folder in case sequential frame segmentation was activated

utils.check_directories()  # Check is all the necessary directories were created and asks for the user to clean them if not empty

# ##################################################################################################################
# Loading captures/test frames to memory#
# ##################################################################################################################

params.main["capturesRGB"], params.main["capturesDEPTH"], \
params.main["testFrameRGB"], params.main["testFrameDEPTH"] = tmt_io.VideoLoader(params.main["dataDir"], in_folder=True, **params.tracker)


# ##################################################################################################################
# Initializes the TopMouseTracker class#
# ##################################################################################################################

Tracker = tracker.TopMouseTracker(**params.tracker)

# ##################################################################################################################
# Creating ROI for analysis#
# ##################################################################################################################

'''

Click with the left mouse button in the upper-left part of the ROI to select 
and drag the mouse until reaching the lower-right part of the ROI to select then release

Press R to RESET the ROI
Press C to CROP and save the created ROI

'''

Tracker.SetROI(refPt=None)

##%%#######################################################################################################################################################
##/!\ [OPTIONAL] Adjusting the segmentation parameters for Animal/Object
#######################################################################################q####################################################################
#
# params.segmentationParameters["threshMaxMouse"] = np.array([255, 255, 85],np.uint8); #The upper parameter for the thresholding of the cotton (hsv)
#
# Tracker.AdjustThresholding(params.mainParameters["capturesRGB"][0], 2000, which='animal'); #Parameters : capture, video frame to display

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

tracker.TopTracker(Tracker, **params.tracker)

#%%#######################################################################################################################################################
#Complete Tracking plot
##########################################################################################################################################################  

params.plot["limit"] = 8.

params.complete_tracking_plot["cottonSubplots"] = True
params.complete_tracking_plot["alpha"] = 0.5

cmap = cm.coolwarm
params.complete_tracking_plot["cBefore"] = cmap(0.)  # cmap(0.3)
params.complete_tracking_plot["cAfter"] = cmap(1.)  # cmap(0.7)

Plot = analysis.Plot(**params.tracker)
Plot.GeneratePeakFile(**params.nesting_raster_plot)

Plot.CompleteTrackingPlot(**params.complete_tracking_plot)

#%%#######################################################################################################################################################
#Nesting Raster plot
##########################################################################################################################################################

Plot.NestingRaster(**params.nesting_raster_plot)

#%%#######################################################################################################################################################
#/!\ [OPTIONAL] Make a live tracking plot
##########################################################################################################################################################  

Plot.LiveTrackingPlot(res=Plot._framerate[0], tStartLivePlot=19*60, tEndLivePlot=20*60, acceleration=10)
