#!/usr/bin/env python3
# -*- coding: utf-8
"""
Created on Fri Dec  7 13:58:46 2018

@author: Thomas Topilko
"""

from tmt_kinect import Kinect

saving_path = "C://path/to/the/saving_folder"

experiment, animal = '210107', '250' #Experiment tag; animal tag
kinect = Kinect(experiment, animal)

# SELECT REGION OF INTEREST (ROI) FOR VIDEO RECORDING
# LEFT CLICK AND DRAG TO SELECT
# PRESS C TO CONFIRM; PRESS R TO RESET
kinect.set_roi()

recording_framerate = 15  # Recording framerate (fps)
recording_duration = 20  # Duration in seconds
kinect.capture_camera_feed(recording_framerate, saving_path, recording_duration) #Launch recording
