# -*- coding: utf-8 -*-
"""
TopMouseTracker default parameters module.

This module defines default parameters used by TopMouseTracker.

"""

import os;
import cv2;
            
##############################################################################
# Parameters for sounds outputs
##############################################################################             

if os.path.exists("/System/Library/Sounds/") :
    
    sounds = {sound.split(".")[0]: sound.split(".")[0] for sound in os.listdir("/System/Library/Sounds/")};
    
else :
    
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
# Parameters to set working folders, and sets of images to process
##############################################################################       

foldersToCreate = {
    'Results' : True
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


