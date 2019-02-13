#!/usr/bin/env python3

# -*- coding: utf-8 -*-

"""
Created on Tue Feb  5 13:29:39 2019

@author: tomtop

"""

import os;
import cv2;
import threading;
import Queue;
import matplotlib.pyplot as plt;
from pykinect2 import PyKinectV2,PyKinectRuntime;

import TopMouseTracker.Utilities as utils;
import TopMouseTracker._KinectVideoWriter as kinect;
import TopMouseTracker.Threading as kinectThread;

_mainDir = os.path.expanduser("~");
_desktopDir = os.path.join(_mainDir,"Desktop");
_savingDir = os.path.join(_mainDir,"TopMouseTracker");

utils.CheckDirectoryExists(_savingDir);

kinectParameters = {"savingDir" : _savingDir,
                    "mice" : "226_217",
                    "kinectRGB" : None,
                    "kinectDEPTH" : None,
                    "gridRes" : 20,
                    "depthMinThresh" : 130,
                    "depthMaxThresh" : 140,
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

Kinect.TestKinect(grid=True); #If grid == True : Displays the calibration grid on depth image

#%%###########################################################################
#Launch saving#
##############################################################################

""" If display == True the framerate will be aproximatively 20fps for both RGB and DEPTH streams
    [INFO] To stop the code press Cntrl+C

    If display == False the framerate will be slightly higher, approximatively 21fps for both RGB and DEPTH streams
    [INFO] To stop the code press Q
    
    The parallel variable is used to parallelize the video saving (Work in progress)
    The samplingTime is the time in seconds that the program uses to estimate the maximal framerate of the camera
"""

samplingTime = 5*60; #5 minutes

Kinect.PlayAndSave(display=True,parallel=False,samplingTime=samplingTime);


#%%###########################################################################
#THREADING TEST
##############################################################################

import TopMouseTracker.Threading as thread;

RGB_Queue = Queue.Queue();
DEPTH_Queue = Queue.Queue();

MetaData = thread.MetaData();

RGB_Grabber = thread.ImageGrabber(kinectParameters["kinectRGB"], "rgb", RGB_Queue);
DEPTH_Grabber = thread.ImageGrabber(kinectParameters["kinectDEPTH"], "depth", DEPTH_Queue);

RGB_Writer = thread.VideoWriter(MetaData.RGBWriter,RGB_Queue);
DEPTH_Writer = thread.VideoWriter(MetaData.DEPTHWriter,DEPTH_Queue);

RGB_Grabber.start();
DEPTH_Grabber.start();

RGB_Writer.start();
DEPTH_Writer.start();

RGB_Grabber.join();
DEPTH_Grabber.join();

RGB_Writer.join();
DEPTH_Writer.join();
