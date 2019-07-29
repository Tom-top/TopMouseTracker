# -*- coding: utf-8 -*-
"""
TopMouseTracker default parameters module.

This module defines default parameters used by TopMouseTracker.

"""

import os;
import cv2;
import numpy as np;

mainParameters = {  
                    "extensionLoad" : "avi",
                    "extension" : "avi", #The extension of the movie to be analyzed
                    "testFramePos" : 300, #The position of the frame used for ROI selection
                    "email" : None, #The email adress to send notifications to
                    "password" : None, #The email password
                    "smtp" : "smtp.gmail.com", #The server smtp
                    "port" : 587, #The server directory
                    "mouse" : None, #The name of the animal
                    "capturesRGB" : None, #The RGB captures (list)
                    "capturesDEPTH" : None, #The DEPTH captures (list)
                    "testFrameRGB" : None, #The RGB test frame (for ROI selection)
                    "testFrameDEPTH" : None, #The DEPTH test frame (why not ?!)
                    "playSound" : False, #Parameters to enable ping sound after code finished running
                    "sound2Play" : None, #The sound to be played
                    };

segmentationParameters = {
                    "threshMinMouse" : np.array([90, 70, 0],np.uint8), #Lower parameter for thresholding the mouse (hsv) #"threshMinMouse" : np.array([0, 10, 40],np.uint8), np.array([100, 70, 0],np.uint8)
                    "threshMaxMouse" : np.array([179, 255, 50],np.uint8), #Upper parameter for thresholding the mouse (hsv) #"threshMaxMouse" : np.array([255, 60, 90],np.uint8), np.array([179, 255, 50],np.uint8)
                    "threshMinCotton" : np.array([0, 0, 150],np.uint8), #Lower parameter for thresholding the cotton (hsv) 
                    "threshMaxCotton" : np.array([140, 57, 250],np.uint8), #Upper parameter for thresholding the cotton (hsv) "threshMaxCotton" : np.array([140, 42, 250],np.uint8)
                    "kernel" : np.ones((5,5),np.uint8), #Parameter for the kernel size used in filters
                    "minAreaMask" : 1000.0, #Parameter for minimum size of mouse detection (pixels) #"minAreaMask" : 100.0, 1000.0
                    "maxAreaMask" : 8000.0, #Parameter for maximum size of mouse detection (pixels) #"maxAreaMask" : 8000.0,
#                    "minDist" : 0.3, #Parameter for minimum distance that the mouse has to travel to be counted (noise filter)
                    "minCottonSize" : 4000., #Parameter for minimum size of cotton detection (pixels)
                    "nestCottonSize" : 15000., #Parameter for maximum size of cotton detection (pixels)
                    "cageLength" : 50, #Length of the cage in cm #50 28.5
                    "cageWidth" : 25., #Width of cage in cm #25 17.
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
        "segmentCotton" : False, #Whether or not to segment cotton in the cage
        "saveCottonMask" : False, #Whether or not to save the cotton mask
        "resizeTracking" : 1., #Resizing factor for video size
        };

plotParameters = {
                "minDist" : 0.5,
                "maxDist" : 10,
                "res" : 1,
                "limit" : 10.,
                "gridsize" : 200,
                };
        
nestingRasterPlotParameters = {
                                "cBefore" : "blue",
                                "cAfter" : "red",
                                "res" : 1,
                                "rasterSpread" : None,
                                "peakThresh" : 0.7,
                                "peakDist" : 1,
                                "minDist" : 7,
                                "displayManual" : False,
                                "save" : True,
                                };
        
if type(nestingRasterPlotParameters['peakThresh']) == float :
    
    nestingRasterPlotParameters['PeakTresh'] = str(nestingRasterPlotParameters['peakThresh'])[0]+'-'+str(nestingRasterPlotParameters['peakThresh'])[2:]

else :
    
    nestingRasterPlotParameters['PeakTresh'] = nestingRasterPlotParameters['peakThresh']
                
completeTrackingPlotParameters = {
                                    "cBefore" : 'blue',
                                    "cAfter" : 'red',
                                    "alpha" : 0.1,
                                    "res" : 1,
                                    "line" : True,
                                    "rasterSpread" : None,
                                    "cottonSubplots" : True,
                                    "save" : True,
                                    };
        
trackerParameters = {
        "main" : mainParameters,
        "segmentation" : segmentationParameters,
        "display" : displayParameters,
        "saving" : savingParameters,
        "plot" : plotParameters,
        };
    
##############################################################################
# Parameters for sounds outputs
##############################################################################             

if os.path.exists("/System/Library/Sounds/") :
    
    sounds = {sound.split(".")[0]: sound.split(".")[0] for sound in os.listdir("/System/Library/Sounds/")};
    
else :
    
    sounds = None;
    print("/!\ [WARNING] The directory : {0} for sound files doesn't exist".format("/System/Library/Sounds/"));
                         
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
    'darkmagenta':"\033[0;35m",
    'darkblack':  "\033[0;30m",
    'bold' :      "\033[1m",
    'off':        "\033[0;0m"
};


##############################################################################
# Parameters for ImageProcessing
##############################################################################

#Encoders for video file conversion
encoders = {
    "x264" : "x264",
    "x265" : "x265",
    "mpeg4" : "mpeg4",
    "mpeg2" : "mpeg2",
};

#Parameter for file import
imageImportParameter = {
    "unchanged" : cv2.IMREAD_UNCHANGED, #returns image as is (with alpha)
    "grayscale" : 0, #converts image to grayscale image
    "color" : 1, #converts image to BGR color image
    "anydepth" : cv2.IMREAD_ANYDEPTH #if image is 16bit, or 32bit loads the corresponding depth
};

#Parameter for LUT conversion
imageTransformParameter = {
    "BGR2RGB" : cv2.COLOR_BGR2RGB,
    "RGB2BGR" : cv2.COLOR_RGB2BGR,
    "BGR2GRAY" : cv2.COLOR_BGR2GRAY,
};
        
#Parameter for gaussian blur filter
gaussianBlurParameter = {
    "kernel" : 5,
    "kernel_small" : 3,
};

#Parameter for thresholding method
thresholdingType = {
    "binary" : cv2.THRESH_BINARY,
    "binary_inv" : cv2.THRESH_BINARY_INV,
    "trunc" : cv2.THRESH_TRUNC,
    "tozero" : cv2.THRESH_TOZERO,
    "tozero_inv" : cv2.THRESH_TOZERO_INV,
};

#Parameters for thresholding
thresholdingParameter = {
    "low" : 7,
    "high" : 255,
    "method" : thresholdingType['binary'],
}

#Parameters for morphological operations
morphologyTransforationParameter = {
    "opening" : cv2.MORPH_OPEN,
    "closing" : cv2.MORPH_CLOSE,
    "gradient" : cv2.MORPH_GRADIENT,
    "top_hat" : cv2.MORPH_TOPHAT,
    "black_hat" : cv2.MORPH_BLACKHAT,
};

contourHierarchyParameter = {
    "list" : cv2.RETR_LIST,
    "external" : cv2.RETR_EXTERNAL,
    "ccomp" : cv2.RETR_CCOMP,
    "tree" : cv2.RETR_TREE,
}

contourApproximationMethod = {
     "none" : cv2.CHAIN_APPROX_NONE,
     "simple" : cv2.CHAIN_APPROX_SIMPLE
}

dilationParameter = {
    "iterations" : 2,
}

distanceTransformParameter = {
    "kernel" : 5,
}

localMaximaParameter = {
    "min_dist" : 20,
}

textFontParameter = {
    "hershey_simplex" : cv2.FONT_HERSHEY_SIMPLEX,
    "hershey_plain" : cv2.FONT_HERSHEY_PLAIN,
    "italic" : cv2.FONT_ITALIC
}

textSizeParameter = {
    "small" : 0.5,
    "medium" : 1.0,
    "large" : 3.0
}

textThicknessParameter = {
    "small" : 2,
    "medium" : 4,
    "large" : 10
}
    
contourSizeParameter = {
    "low" : 50,
    "high" : 700 
}

openingKernelMaskParameter = {
    "x" : 4,
    "y" : 4
}

closingKernelMaskParameter = {
    "x" : 15,
    "y" : 15
}

dilatingKernelMaskParameter = {
    "x" : 25,
    "y" : 25
}

maskParameter = {
    "openingKernelMaskParameter" : openingKernelMaskParameter,
    "closingKernelMaskParameter" : closingKernelMaskParameter,
    "dilatingKernelMaskParameter" : dilatingKernelMaskParameter
}

#Parameter for file saving
compressionSavingParameter = {
    "PNG" : cv2.IMWRITE_PNG_COMPRESSION, #compression range : 0-9 higher = more compressed
    "JPEG" : cv2.IMWRITE_JPEG_QUALITY #quality range : 0-100 higher = better
};


