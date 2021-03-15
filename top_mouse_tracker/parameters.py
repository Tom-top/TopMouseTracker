# -*- coding: utf-8 -*-
"""
TopMouseTracker default parameters module.

This module defines default parameters used by TopMouseTracker.

"""

import os

import numpy as np


main = {
    "testFramePos": 0,  # The position of the frame used for ROI selection
    "smtp": "smtp.gmail.com",  # The server smtp
    "port": 587,  # The server port
    "playSound": False,  # Parameters to enable ping sound after code finished running
    "sound2Play": None,  # The sound to be played
    "extensionLoad": 'avi'
}

segmentation = {
    "threshMinMouse": np.array([0, 0, 0], np.uint8),  # Lower parameter for thresholding the mouse (hsv) #"threshMinMouse" : np.array([0, 10, 40],np.uint8)
    "threshMaxMouse": np.array([255, 255, 70], np.uint8),  # Upper parameter for thresholding the mouse (hsv) #"threshMaxMouse" : np.array([255, 60, 90],np.uint8)
    "threshMinCotton": np.array([0, 0, 150], np.uint8),  # Lower parameter for thresholding the cotton (hsv)
    "threshMaxCotton": np.array([140, 57, 250], np.uint8),  # Upper parameter for thresholding the cotton (hsv) "threshMaxCotton" : np.array([140, 42, 250],np.uint8)
    "kernel": np.ones((5, 5), np.uint8),  # Parameter for the kernel size used in filters
    "minAreaMask": 200.0,  # Parameter for minimum size of mouse detection (pixels) #"minAreaMask" : 100.0,
    "maxAreaMask": 8000.0,  # Parameter for maximum size of mouse detection (pixels)
    "minCottonSize": 200.,  # Parameter for minimum size of cotton detection (pixels)
    "nestCottonSize": 15000.,  # Parameter for maximum size of cotton detection (pixels)
    "showStream": False,  # Display the tracking in LIVE MODE
    "resize_stream": 4,
    "cageWidth": 37,  # FIXME:
    "cageLength": 20
}

'''   
FOURCC :
XVID : cv2.VideoWriter_fourcc(*'XVID') --> Preferable
MJPG : cv2.VideoWriter_fourcc(*'MJPG') --> Very large videos
X264 : cv2.VideoWriter_fourcc(*'X264') --> Gives small videos
None : skvideo.io.FFmpegWriter --> Default writer (small videos)
"Frame" : Saving video as sequential frames
'''
       
saving = {
    "saveStream": False,  # Whether or not to save the segmentation
    "fourcc": None,  # fourcc to be used for video compression
    "savingExtension": "avi",  # The extension of the tracking for saving
    "saveCottonMask": False,  # Whether or not to save the cotton mask
    "resizeTracking": 1.,  # Resizing factor for tracking video
}

plot = {
    "minDist": 0.5,
    "maxDist": 10,
    "res": 1,
    "limit": 10.,
    "gridsize": 200,
}
        
nesting_raster_plot = {
    "cBefore": "blue",
    "cAfter": "red",
    "res": 1,
    "rasterSpread": None,
    "peakThresh": 0.7,
    "peakDist": 1,
    "minDist": 7,
    "displayManual": False,
    "save": True,
}
        
if isinstance(nesting_raster_plot['peakThresh'], float):
    nesting_raster_plot['PeakTresh'] = str(nesting_raster_plot['peakThresh'])[0] + '-' + str(nesting_raster_plot['peakThresh'])[2:]
else:
    nesting_raster_plot['PeakTresh'] = nesting_raster_plot['peakThresh']
                
complete_tracking_plot = {
    "cBefore": 'blue',
    "cAfter": 'red',
    "alpha": 0.1,
    "res": 1,
    "line": True,
    "rasterSpread": None,
    "cottonSubplots": True,
    "save": True,
}
        
tracker = {
    "main": main,
    "segmentation": segmentation,
    "saving": saving,
    "plot": plot,
}
    
##############################################################################
# Parameters for sounds outputs
##############################################################################
if os.path.exists("/System/Library/Sounds/"):
    sounds = {sound.split(".")[0]: sound.split(".")[0] for sound in os.listdir("/System/Library/Sounds/")}
else:
    sounds = None
    print("/!\ [WARNING] The directory : {0} for sound files doesn't exist".format("/System/Library/Sounds/"))
                         
##############################################################################
# Parameters for message outputs
##############################################################################
colors = {
    'white':    "\033[1;37m",
    'yellow':   "\033[1;33m",
    'green':    "\033[1;32m",
    'blue':     "\033[1;34m",
    'cyan':     "\033[1;36m",
    'red':      "\033[1;31m",
    'magenta':  "\033[1;35m",
    'black':      "\033[1;30m",
    'darkwhite':  "\033[0;37m",
    'darkyellow': "\033[0;33m",
    'darkgreen':  "\033[0;32m",
    'darkblue':   "\033[0;34m",
    'darkcyan':   "\033[0;36m",
    'darkred':    "\033[0;31m",
    'darkmagenta': "\033[0;35m",
    'darkblack':  "\033[0;30m",
    'bold':      "\033[1m",
    'off':        "\033[0;0m"
}
