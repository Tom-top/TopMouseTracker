#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Created on Tue Feb  5 13:29:39 2019

@author: tomtop

"""

import os;
import cv2;
import matplotlib.pyplot as plt;
from pykinect2 import PyKinectV2,PyKinectRuntime;

import TopMouseTracker.Utilities as utils;
import TopMouseTracker._Kinect as kinect;

_mainDir = os.path.expanduser("~");
_desktopDir = os.path.join(_mainDir,"Desktop");
_savingDir = os.path.join(_mainDir,"TopMouseTracker");

utils.CheckDirectoryExists(_savingDir);

kinectParameters = {"savingDir" : _savingDir,
                    "mice" : "226_217",
                    "kinectRGB" : None,
                    "kinectDEPTH" : None,
                    "gridRes" : 20,
                    "rawVideoFileName" : None,
                    "depthVideoFileName8Bit" : None,
                    "depthVideoFileName16Bit" : None,
                    "fourcc" : cv2.VideoWriter_fourcc(*'MJPG'),
                    };              

kinectParameters["rawVideoFileName"] = "Raw_Video_Mice_{0}".format(kinectParameters["mice"]);
kinectParameters["depthVideoFileName8Bit"] = "Depth_Video_Mice_8b_{0}".format(kinectParameters["mice"]);
kinectParameters["depthVideoFileName16Bit"] = "Depth_Video_Mice_16b_{0}".format(kinectParameters["mice"]);

#%%###########################################################################
#Setting up cameras#
##############################################################################

kinectParameters["kinectRGB"] = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Color | PyKinectV2.FrameSourceTypes_Body); #Initializes the RGB camera
kinectParameters["kinectDEPTH"] = PyKinectRuntime.PyKinectRuntime(PyKinectV2.FrameSourceTypes_Depth); #Initializes the DEPTH camera

#%%###########################################################################
#Initializes the kinect object#
##############################################################################

Kinect = kinect.Kinect(**kinectParameters);

#%%###########################################################################
#[OPTIONAL] Test the kinect for positioning#
##############################################################################

Kinect.TestKinect(grid=False); #If grid == True : Displays the calibration grid on depth image

#%%###########################################################################
#Launch saving#
##############################################################################

Kinect.PlayAndSave(display=True); #If display == True : Displays the RGB feed from the camera