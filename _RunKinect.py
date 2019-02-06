#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 13:29:39 2019

@author: tomtop
"""

import os;
import cv2;
from pykinect2 import PyKinectV2,PyKinectRuntime;

import TopMouseTracker.Utilities as utils;
import TopMouseTracker._Kinect as kinect;

_mainDir = os.path.expanduser("~");
_desktopDir = os.path.join(_mainDir,"Desktop");
_savingDir = os.path.join(_mainDir,"TopMouseTracker");

utils.CheckDirectoryExists(_savingDir);

kinectParameters = {"savingDir" : _savingDir,
                    "mice" : "193_195",
                    "kinectRGB" : None,
                    "kinectDEPTH" : None,
                    "depthMinThresh" : 130,
                    "depthMaxThresh" : 140,
                    "gridRes" : 20,
                    "rawVideoFileName" : None,
                    "depthVideoFileName" : None,
                    "framerate" : 15,
                    "fourcc" : cv2.VideoWriter_fourcc(*'MJPG'),
                    };
                    
kinectParameters["rawVideoFileName"] = "Raw_Video_Mice_{0}".format(kinectParameters["mice"]);
kinectParameters["depthVideoFileName"] = "Depth_Video_Mice_{0}".format(kinectParameters["mice"]);
        
#%%###########################################################################
#Setting up cameras#
##############################################################################
        
kinectParameters["kinectRGB"] = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body); #Initializes the RGB camera
kinectParameters["kinectDEPTH"] = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth); #Initializes the DEPTH camera

#%%###########################################################################
#Initializes the kinect object#
##############################################################################

Kinect = kinect.Kinect();

#%%###########################################################################
#[OPTIONAL] Test the kinect for positioning#
##############################################################################

Kinect.TestKinect();

#%%###########################################################################
#Launch saving#
##############################################################################

Kinect.PlayAndSave(display=True);