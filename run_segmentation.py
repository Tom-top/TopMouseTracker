#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on Fri Dec  7 13:58:46 2018

@author: Thomas Topilko
"""

import os
from configobj import ConfigObj

import tmt_utilities as utils
import tmt_io
import tmt_tracker as tracker
import tmt_analysis as analysis

configspec = ConfigObj("config.configspec", interpolation=False, list_values=False, _inspec=True)
config = ConfigObj("config.cfg", configspec=configspec, unrepr=True)

config["general"]["experiment_directory"] = os.path.join("Test", config["general"]["general"]["experiment_tag"])
config["general"]["recording_directory"] = os.path.join(config["general"]["experiment_directory"],
                                                        config["general"]["general"]["recording_tag"])
config["general"]["segmentation_directory"] = os.path.join(config["general"]["recording_directory"],
                                                           "segmentation_results_{}".format(
                                                               config["general"]["general"]["animal_tag"]))

utils.setup_and_clear_segmentation_dir(config["general"]["segmentation_directory"])

#######################################################################################################################################################
# Loading video recordings#
#########################################################################################################################################################

color_capture = tmt_io.concatenate_video_clips(config["general"]["general"]["color_video_prefix"], **config)
depth_capture = tmt_io.concatenate_video_clips(config["general"]["general"]["depth_video_prefix"], **config)

#########################################################################################################################################################
# Initializes the TopMouseTracker class#
#########################################################################################################################################################

Tracker = tracker.TopMouseTracker(color_capture, depth_capture, **config)

########################################################################################################################################################
# Creating ROI for segmentation#
##########################################################################################################################################################

'''
Click with the left mouse button in the upper-left part of the ROI to select 
and drag the mouse until reaching the lower-right part of the ROI to select then release

Press R to RESET the ROI
Press C to CROP and save the created ROI
'''

Tracker.set_segmentation_roi(pos_frame=0)

##%%#######################################################################################################################################################
##/!\ [OPTIONAL] Adjusting the segmentation parameters for Animal/Object
#######################################################################################q####################################################################

Tracker.adjust_thresholding(pos_frame=100, which='material')

########################################################################################################################################################
# Launch segmentation on video(s)#
##########################################################################################################################################################

Tracker.segment()

# %%#######################################################################################################################################################
# Complete Tracking plot
##########################################################################################################################################################

Plot = analysis.Plot(color_capture, depth_capture, time_limit=None, **config)
Plot.detect_nesting_events()
Plot.generate_raster_file()

Plot.plot_complete_tracking(track_length=5)  # track_length in s

# %%#######################################################################################################################################################
# /!\ [OPTIONAL] Make a live tracking plot
##########################################################################################################################################################

Plot.live_tracking_plot(res=Plot._framerate_color,
                        start_live_plot=0 * 60,
                        end_live_plot=5 * 60,
                        acceleration=5)
