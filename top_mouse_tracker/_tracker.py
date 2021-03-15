#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:03:44 2018

@author: tomtop
"""

import os
import time

import smtplib
from email.mime.multipart import MIMEMultipart

import numpy as np
import pandas as pd

import cv2
import skvideo.io

import xlwt

import TopMouseTracker.parameters as params
import TopMouseTracker.utilities as utils
import TopMouseTracker.tmt_io as IO


class TopMouseTracker():
    '''Main tracking class :
        
       Methods :
           
           • __init__ : Sets all the parameters usefull for the tracking,
           displaying and saving processes.
           
           • SetRegistrationParameters : Computes and sets variables for depth 
           registration based on the size of RGB and DEPTH images.
           
           • SetMetaDataParameters : Loads all the metadata from excel files : 
           The 'Mice_Video_Info.xlsx' file that contains all timing data for 
           each animal. The 'MetaData*.xlsx' file that contains all video meta-
           data.
           
           • SetROI : Displays the first frame from the video for the user to
           select the ROI where the segmentation has to run. Press 'r' to reset
           the ROI; Press 'c' to set the ROI.
           
           • Main : Retrieves the next RGB and DEPTH frames and runs the
           segmentation on them.
           
           • RunSegmentations : Wrapper to run the Mouse segmentation, the DEPTH
           registration, and the Cotton segmentation in one method call
           
           • RunSegmentationMouse : Segments the Mouse position. Steps :
               
               1) RGB --> HSV (Hue, Saturation, Value) LUT conversion
               2) Gaussian blur
               3) Binarization based on HSV parameters
               4) Opening to filter salt and pepper noise
               5) Closing to fill the objects with holes
               6) Contour detection on the binary mask
               7) Detection of the biggest contour
               8) Computation of the centroid of the largest detected object
    
           • RunSegmentationCotton : Segments the Cotton average height. Steps :
               
               * Using the blurred HSV image from RunSegmentationMouse Method
               1) Binarization based on HSV parameters
               2) Opening to filter salt and pepper noise
               3) Closing to fill the objects with holes
               4) Contour detection on the binary mask
               5) Bitwise AND operation between binary mask and registered depth
               6) Computing the average height values of the pixels inside the Bitwise AND mask
               7) Computing the number of Large/Small objects based on contour area sizes
    
           • RegisterDepth : Registers DEPTH image onto the RGB image
           
           • CreateDisplay : Creates the canvas for the segmentation display
           and saving
           
           • WriteDisplay : Writes metadata information on the created canvas 
           
           • SaveTracking : Saves tracking as a video
           
           • ComputeDistanceTraveled : Computes the distance travelled between 
           the previous and the current position in (cm)
           
           • UpdatePosition : Updates the position of the mouse 
           
           • StorePosition : Stores the position of the mouse
           
           • Error : Stores the number of times that the program failed to detect
           the Mouse
           
           • ReturnTracking : Returns the tracking canvas for display purposes
           
    Parameters :
        
        **kwargs (dict) : dictionnary with all the parameters useful for the tracker
    '''
    
    def __init__(self, **kwargs):
        # General variables
        # ----------------------------------------------------------------------
        self._args = kwargs  # Loads the main arguments
        self._break = False  # Checks if the tracker was manually aborted to resume segmentation
        self._Start = None  # Time at which the segmentation starts
        self._End = None  # Time at which the segmentation ends
        self._Stop = False  # Trigger to stop segmentation when video is empty
        self._mouseDetectedExcel = False
        self._mouse = self._args["main"]["mouse"]  # Loads the name of the mouse
        self._cageWidth = self._args["segmentation"]["cageWidth"]
        self._cageLength = self._args["segmentation"]["cageLength"]
        self._testFrameRGB = self._args["main"]["testFrameRGB"].copy()  # Loads a test RGB frame
        self._H_RGB, self._W_RGB = self._testFrameRGB.shape[0], self._testFrameRGB.shape[1]  # Height, Width of the RGB frames
        
        if self._args["saving"]["segmentCotton"]:
            self._testFrameDEPTH = self._args["main"]["testFrameDEPTH"].copy()  # Loads a test DEPTH frame
            self._H_DEPTH, self._W_DEPTH = self._testFrameDEPTH.shape[0], self._testFrameDEPTH.shape[1]  # Height, Width of the DEPTH frames
            
        self._framerate = [x.fps for x in self._args["main"]["capturesRGB"]]
        
        # Registration variables
        # ----------------------------------------------------------------------
        self._resizingFactorRegistration = 2.8965  # Parameters for resizing of the depth image for registration
        
        if self._args["saving"]["segmentCotton"]:
            self._testFrameDEPTHResized = cv2.resize(self._testFrameDEPTH,(0,0),fx = self._resizingFactorRegistration,fy = self._resizingFactorRegistration)
            self._H_DEPTH_RESIZED, self._W_DEPTH_RESIZED = self._testFrameDEPTHResized.shape[0], self._testFrameDEPTHResized.shape[1]
            
            self.SetRegistrationParameters(self._args["segmentation"]["registrationX"],
                                           self._args["segmentation"]["registrationY"])  # Computes all the placement parameters for registration

        # Global tracking variables
        # ----------------------------------------------------------------------
        self._positions = []  # Position of the mouse every s/frameRate
        self._maskAreas = []  # Size of the mouse mask every s/frameRate
        self._distances = []  # Distances traveled by the mouse every s/frameRate
        self._cottonAveragePixelIntensities = []  # List containing all the depth information of segmented objects coming from the cotton mask
        self._largeObjects = []  # List containing the number of large objects detected every s/framerate
        self._smallObjects = []  # List containing the number of small objects detected every s/framerate
        self._detectedNest = []  # List containing the detection status of the nest
        self._cottonSpread = []  # List containing the spread of cotton
        self._cottonCenter = []  # List containing the center of the cotton mask
        self._distance = 0.  # Cumulative distance traveled by the mouse
        self._errors = 0  # Counter for the number of times that the trackers fails to segment the animal
        
        # Real-Time tracking variables
        # ----------------------------------------------------------------------
        self.videoNumber = 0
        self.frameNumber = 0
        self.fn = 0
        self.realTimePosition = []  # Position of the mouse in real time
        self.realTimeSpeed = 0.  # Speed of the mouse in real time
        self.center = None  # Centroid (x,y) of the mouse binary mask in real time
        self.correctedCenter = None  # Corrected centroid (x,y) of the mouse binary mask in real time
        self.correctedCenterCotton = None  # Corrected centroid (x,y) of the cotton binary mask in real time
        
        # MetaData variables
        # ----------------------------------------------------------------------
        self.SetMetaDataParameters()
        self._videoLength = sum(self._tEnd) - self._tStart
        
        # Tracking canvas variables
        # ----------------------------------------------------------------------
        self._metaDataCanvasSize = 750  # Size of the canvas
        self.textFontSize = 1.5  # font size of the text
        self.textFontThickness = 3  # thickness of the text
        self.contourThickness = 2  # thickness of the contours
        self.centerSize = 3  # size of the object centroid
        self.centerThickness = 5  # thickness of the object centroid
        self.metaDataPos1 = 50  # position of the metadata
        
        # Saving variables
        # ----------------------------------------------------------------------
        self._displayOutputShape = True
        self._startSaving = False
        self._startSavingMask = False
        
    def SetTrackingVideoSaving(self):
        if self._args["saving"]["saveStream"]:
            self.videoString = "Tracking_{0}.{1}".format(self._mouse, self._args["saving"]["savingExtension"])
            self.testCanvas = np.zeros((self._W_RGB_CROPPED, self._H_RGB_CROPPED))
            self.testCanvas = cv2.resize(self.testCanvas, (0, 0),
                                         fx=1./self._args["saving"]["resizeTracking"],
                                         fy=1./self._args["saving"]["resizeTracking"])
                                   
            if self._args["saving"]["fourcc"] is None:  # TODO: check
                self.videoWriter = skvideo.io.FFmpegWriter(os.path.join(self._args["main"]["resultDir"], self.videoString),
                                                           inputdict={'-r': str(np.mean(self._framerate))},
                                                           outputdict={'-r': str(np.mean(self._framerate))})
                utils.PrintColoredMessage("\n[INFO] Default skvideo video saving mode selected", "darkgreen")
            elif self._args["saving"]["fourcc"] == "Frame":
                utils.PrintColoredMessage("\n[INFO] Image saving mode selected \n", "darkgreen")
            else:
                self.videoWriter = cv2.VideoWriter(os.path.join(self._args["main"]["resultDir"],
                                                                self.videoString), self._args["saving"]["fourcc"],
                                                   np.mean(self._framerate),
                                                   (self.testCanvas.shape[0], self.testCanvas.shape[1]))
                
                utils.PrintColoredMessage("\n[INFO] Video saving mode selected", "darkgreen")
                utils.PrintColoredMessage("[INFO] VideoWriter : {0} ; {1} \n"
                                          .format((self.testCanvas.shape[0], self.testCanvas.shape[1]),
                                                  self._args["saving"]["fourcc"]), "darkgreen")
    
    def SetRegistrationParameters(self, x, y):
        self.start_X = 0
        self.end_X = self.start_X + self._H_DEPTH_RESIZED
        self.start_Y = x
        self.end_Y = self.start_Y + self._W_DEPTH_RESIZED
        
        if self.end_X >= self._H_RGB:
            self.end_X_foreground = self._H_RGB - self.start_X
        else:
            self.end_X_foreground = self._H_RGB
            
        if self.end_Y >= self._W_RGB:
            self.end_Y_foreground = self._W_RGB - self.start_Y
        else:
            self.end_Y_foreground = self._W_RGB
        
        self.start_X_foreground = y
        self.start_Y_foreground = 0
        
        if self.start_X_foreground != 0:
            self.end_X = self.end_X-self.start_X_foreground
            self.end_X_foreground = self.end_X_foreground + self.start_X_foreground
            
        if self.start_Y_foreground != 0:
            self.end_Y = self.end_Y-self.start_Y_foreground
            self.end_Y_foreground = self.end_Y_foreground + self.start_Y_foreground
            
    def TestRegistrationParameters(self, x, y, h, w):
        start_X = 0
        end_X = start_X+h
        start_Y = x
        end_Y = start_Y+w
        
        if end_X >= self._H_RGB:
            end_X_foreground = self._H_RGB - start_X
        else:
            end_X_foreground = self._H_RGB
            
        if end_Y >= self._W_RGB:
            end_Y_foreground = self._W_RGB - start_Y
        else:
            end_Y_foreground = self._W_RGB
        
        start_X_foreground = y
        start_Y_foreground = 0
        
        if start_X_foreground != 0:
            end_X = end_X - start_X_foreground
            end_X_foreground = end_X_foreground + start_X_foreground
            
        if start_Y_foreground != 0:
            end_Y = end_Y - start_Y_foreground
            end_Y_foreground = end_Y_foreground + start_Y_foreground
            
        return start_X, end_X, start_Y, end_Y, start_X_foreground, end_X_foreground, start_Y_foreground, end_Y_foreground
            
    def SetMetaDataParameters(self):
        '''Method that loads metaData for segmentation
        
        -----------------------------------------------------------------------
        |  n  | tStart | tStartBehav |  tEnd  | tEndbis | ...
        -----------------------------------------------------------------------
  Ex :  | 01  | 10*60  |    20*60    | 1*3600 |  20*60  | ...
        -----------------------------------------------------------------------
        
        - (Animal name) In this example the animal that will be analyzed is : 01
        - (tStart) The segmentation will start at : 10*60(s) = 10(mins) from the begining of the video
        - (tStartBehav) Time at which the animal started a specific task (if there is no specific task : N.A) : 20*60(s) = 20(mins) from the begining of the video
        - (tEnd) Time at which the segmentation ends : 1*3600(s) = 1(h) from the begining of the video
        - (tEndBis) Time at which the next video ends (if multiple videos in tandem) ...
        
        '''
        videoInfoWorkbook = pd.read_excel(self._args["main"]["videoInfoFile"])  # Load video info excel sheet
        nColumns = len(videoInfoWorkbook.columns)  # Gets the number of columns that the spreadsheet contains
        
        try:
            nameColumn = videoInfoWorkbook["n"]  # Gets name data from column "n"
        except:
            nameColumn = videoInfoWorkbook[videoInfoWorkbook.columns[0]]  # If no column "n" gets name data from the first column

        for pos, n in enumerate(nameColumn):
            try:
                name = str(int(n))
            except:
                name = str(n)
                
            if name == self._args["main"]["mouse"]:
                self._tStart = int(videoInfoWorkbook["tStart"][pos])  # Moment at which the cotton is added (s)
                self.frameNumber = int(self._tStart*self._framerate[self.videoNumber])  # The first frame from which the segmentation should start
                  
                if videoInfoWorkbook["tStartBehav"][pos] == 'N.A':
                    self._tStartBehav = 999999  # if tStartBehav is N.A : set it to infinite, i.e : never starting
                else:
                    self._tStartBehav = int(videoInfoWorkbook["tStartBehav"][pos])  # Moment at which the animal starts the specific task
                      
                self._tEnd = [int(videoInfoWorkbook[videoInfoWorkbook.columns[x]][pos]) for x in np.arange(3, nColumns, 1)] #Length of each video of the experiment
                
                self._mouseDetectedExcel = True  # The entry for self._mouse was successfully detected
                
                utils.PrintColoredMessage("\n[INFO] Entry in Video_Info file detected for mouse {0}"
                                          .format(self._mouse), "darkgreen")
        
        if not self._mouseDetectedExcel:  # The entry for self._mouse was not successfully detected
            raise RuntimeError("Data for mouse {0} was not detected in Video_Info file!".format(self._mouse))
        
        self._Length = sum(self._tEnd)  # Computes the Length of the segmentation (s)
        self._nVideos = len([x for x in self._tEnd if x > 0])  # Computes the number of videos to segment

    def SetROI(self, refPt=None):
        '''Method that displays a test frame from the video for the user to select the ROI in which
        the segmentation will be run
        
        '''
        print("\n")
        utils.PrintColoredMessage("[INFO] Press R to reset ROI, and C to crop the selected ROI", "bold")
        
        if refPt is None:  # TODO: check
            self._refPt = IO.CroppingROI(self._args["main"]["testFrameRGB"].copy()).roi()  # Defining the ROI for segmentation
        else:
            self._refPt = refPt
        
        self.upLeftX = int(self._refPt[0][0])  # Defines the Up Left ROI corner X coordinates
        self.upLeftY = int(self._refPt[0][1])  # Defines the Up Left ROI corner Y coordinates
        self.lowRightX = int(self._refPt[1][0])  # Defines the Low Right ROI corner X coordinates
        self.lowRightY = int(self._refPt[1][1])  # Defines the Low Right ROI corner Y coordinates
        
        self.ROIWidth = abs(self.lowRightX-self.upLeftX)  # Computes the ROI width in (px)
        self.ROILength = abs(self.lowRightY-self.upLeftY)  # Computes the ROI length in (px)

        # Resizing factor for distance calculus
        self.distanceRatio = (abs(self.upLeftX-self.lowRightX)/self._args["segmentation"]["cageLength"] +
                              abs(self.upLeftY-self.lowRightY)/self._args["segmentation"]["cageWidth"])/2
                              
        self._testFrameRGBCropped = self._testFrameRGB[self.upLeftY:self.lowRightY, self.upLeftX:self.lowRightX]  # Creates a croped test frame according to ROI
        self._H_RGB_CROPPED, self._W_RGB_CROPPED = self._testFrameRGBCropped.shape[0], self._testFrameRGBCropped.shape[1]  # Computes the height/width of the cropped frame
        
        if self._args["saving"]["saveCottonMask"]:  # If the Cotton mask has to be saved
            self.depthMaskString = "Mask_Cotton_{0}.avi".format(self._mouse)
            if self._args["saving"]["fourcc"] is None:  # TODO: check
                self.depthMaskWriter = skvideo.io.FFmpegWriter(self._args["main"]["resultDir"]+"/" + self.depthMaskString)
            elif self._args["saving"]["fourcc"] == "Frame":
                pass
            else:
                self.depthMaskWriter = cv2.VideoWriter(os.path.join(self._args["main"]["resultDir"], self.depthMaskString),
                                                       self._args["saving"]["fourcc"], self._framerate[self.videoNumber],
                                                       (self._W_RGB_CROPPED, self._H_RGB_CROPPED))
    
    def Nothing(self, x):
        '''Empty method
        
        '''
        pass
        
    def AdjustRegistration(self, capRGB, capDepth, framePos):
        cv2.namedWindow('Adjust Registration')
        
        cv2.createTrackbar('x', 'Adjust Registration', self._args["segmentation"]["registrationX"], 500, self.Nothing)
        cv2.createTrackbar('y', 'Adjust Registration', self._args["segmentation"]["registrationY"], 500, self.Nothing)
        cv2.createTrackbar('resize', 'Adjust Registration', int(2.8965*10000), 30000, self.Nothing)
        
        # cv2.createTrackbar('alpha','Adjust Registration',100,255,self.Nothing)
        # cv2.createTrackbar('beta','Adjust Registration',150,255,self.Nothing)
        
        self.testRGBFrame = capRGB.get_frame(framePos)
        
        self.testDepthFrame = capDepth.get_frame(framePos)
        
        self.testHsvFrame = cv2.cvtColor(self.testRGBFrame, cv2.COLOR_BGR2HSV)
        self.testBlur = cv2.blur(self.testHsvFrame, (5, 5))
        self.testMaskCotton = cv2.inRange(self.testBlur, self._args["segmentation"]["threshMinCotton"], self._args["segmentation"]["threshMaxCotton"])
        self.testCntsCotton = cv2.findContours(self.testMaskCotton.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]  # Finds the contours of the image to identify the meaningful object
        
        while True:
            # get current positions of four trackbars
            x = cv2.getTrackbarPos('x', 'Adjust Registration')
            y = cv2.getTrackbarPos('y', 'Adjust Registration')
            resizingFactorRegistration = cv2.getTrackbarPos('resize', 'Adjust Registration')
            resizingFactorRegistration = resizingFactorRegistration / 10000
            
#            alpha = cv2.getTrackbarPos('alpha','Adjust Registration')
#            beta = cv2.getTrackbarPos('beta','Adjust Registration')
            
            resizedDepthFrame = cv2.resize(self.testDepthFrame, (0, 0),
                                           fx=resizingFactorRegistration, fy=resizingFactorRegistration)
            _H_DEPTH_RESIZED, _W_DEPTH_RESIZED = resizedDepthFrame.shape[0], resizedDepthFrame.shape[1]
            
            start_X, end_X, start_Y, end_Y, start_X_foreground, end_X_foreground, start_Y_foreground, end_Y_foreground = self.TestRegistrationParameters(x, y, _H_DEPTH_RESIZED, _W_DEPTH_RESIZED)
            self.RegisterDepth(self.testDepthFrame, resizingFactorRegistration, start_X, end_X, start_Y, end_Y, start_X_foreground, end_X_foreground, start_Y_foreground, end_Y_foreground)
            
            cv2.drawContours(self.registeredDepth, self.testCntsCotton, -1, (255, 255, 255), 2)
            # self.registeredDepthNorm = cv2.normalize(self.registeredDepth, dst=None, alpha=alpha, beta=beta)  # norm_type=cv2.NORM_MINMAX
            self.registeredDepth = self.registeredDepth[:, :, 0]
            self.registeredDepthNorm = cv2.equalizeHist(self.registeredDepth)
            self.registeredDepthDisplay = cv2.applyColorMap(self.registeredDepthNorm, cv2.COLORMAP_JET)
            
            cv2.imshow('Adjust Registration', self.registeredDepthDisplay)
            
            key = cv2.waitKey(10) & 0xFF
        
            if key == ord("q"):
                break
        
        cv2.destroyAllWindows()
        
        for i in range(1, 5):
            cv2.waitKey(1)

    def AdjustThresholding(self, capture, framePos, which='animal'):
        '''Method to check the segmentation parameters for the animal/object
        
        Args : capture (cap) : capture of the video to be analyzed
               framePos (int) : number of the frame to be analyzed
               which (str) : 'animal' if the mouse segmentation parameters have to be tested
                             'object' if the cotton segmentation parameters have to be tested
                             
        The trackbars represent respectively :
            
            The Hue values Low/High
            The Saturation values Low/High
            The Value values Low/High
            
        If the elements in the image become black : those elements will become 0 after binarization, i.e : neglected
        If the elements in the image remain colored : those elements will become 1 after binarization, i.e : segmented
        
        '''
        cv2.namedWindow('Adjust Thresholding')
        
        if which == 'animal':
            cv2.createTrackbar('H_Low', 'Adjust Thresholding',
                               self._args["segmentation"]["threshMinMouse"][0], 255, self.Nothing)
            cv2.createTrackbar('H_High', 'Adjust Thresholding',
                               self._args["segmentation"]["threshMaxMouse"][0], 255, self.Nothing)
            
            cv2.createTrackbar('S_Low', 'Adjust Thresholding',
                               self._args["segmentation"]["threshMinMouse"][1], 255, self.Nothing)
            cv2.createTrackbar('S_High', 'Adjust Thresholding',
                               self._args["segmentation"]["threshMaxMouse"][1], 255, self.Nothing)
            
            cv2.createTrackbar('V_Low', 'Adjust Thresholding',
                               self._args["segmentation"]["threshMinMouse"][2], 255, self.Nothing)
            cv2.createTrackbar('V_High', 'Adjust Thresholding',
                               self._args["segmentation"]["threshMaxMouse"][2], 255, self.Nothing)
        elif which == 'object':
            cv2.createTrackbar('H_Low', 'Adjust Thresholding',
                               self._args["segmentation"]["threshMinCotton"][0], 255, self.Nothing)
            cv2.createTrackbar('H_High', 'Adjust Thresholding',
                               self._args["segmentation"]["threshMaxCotton"][0], 255, self.Nothing)
            
            cv2.createTrackbar('S_Low', 'Adjust Thresholding',
                               self._args["segmentation"]["threshMinCotton"][1], 255, self.Nothing)
            cv2.createTrackbar('S_High', 'Adjust Thresholding',
                               self._args["segmentation"]["threshMaxCotton"][1], 255, self.Nothing)
            
            cv2.createTrackbar('V_Low', 'Adjust Thresholding',
                               self._args["segmentation"]["threshMinCotton"][2], 255, self.Nothing)
            cv2.createTrackbar('V_High', 'Adjust Thresholding',
                               self._args["segmentation"]["threshMaxCotton"][2], 255, self.Nothing)
        else :
            cv2.createTrackbar('H_Low', 'Adjust Thresholding', 0, 255, self.Nothing)
            cv2.createTrackbar('H_High', 'Adjust Thresholding', 255, 255, self.Nothing)
            
            cv2.createTrackbar('S_Low', 'Adjust Thresholding', 0, 255, self.Nothing)
            cv2.createTrackbar('S_High', 'Adjust Thresholding', 255, 255, self.Nothing)
            
            cv2.createTrackbar('V_Low', 'Adjust Thresholding', 0, 255, self.Nothing)
            cv2.createTrackbar('V_High', 'Adjust Thresholding', 255, 255, self.Nothing)

        self.testFrame = capture.get_frame(framePos)
        self.testCroppedFrame = self.testFrame[self.upLeftY:self.lowRightY, self.upLeftX:self.lowRightX]
        self.testCroppedFrameRGB = cv2.cvtColor(self.testCroppedFrame, cv2.COLOR_BGR2RGB)
        self.testHsvFrame = cv2.cvtColor(self.testCroppedFrame, cv2.COLOR_BGR2HSV)
        self.testBlur = cv2.blur(self.testHsvFrame, (5, 5))
        
        while True:
            # get current positions of four trackbars
            h_l = cv2.getTrackbarPos('H_Low', 'Adjust Thresholding')
            h_h = cv2.getTrackbarPos('H_High', 'Adjust Thresholding')
            
            s_l = cv2.getTrackbarPos('S_Low', 'Adjust Thresholding')
            s_h = cv2.getTrackbarPos('S_High', 'Adjust Thresholding')
            
            v_l = cv2.getTrackbarPos('V_Low', 'Adjust Thresholding')
            v_h = cv2.getTrackbarPos('V_High', 'Adjust Thresholding')
            
            self.testMaskMouse = cv2.inRange(self.testBlur, (h_l, s_l, v_l), (h_h, s_h, v_h))
            self.overlay = cv2.bitwise_and(self.testHsvFrame, self.testHsvFrame, mask=self.testMaskMouse)
            
            cv2.imshow('Adjust Thresholding', self.overlay)
            
            key = cv2.waitKey(10) & 0xFF
        
            if key == ord("q"):
                break
        
        cv2.destroyAllWindows()
        
        for i in range(1, 5):
            cv2.waitKey(1)
        
    def Main(self, rgbCapture, depthCapture):
        # Get frame from capture
        # ----------------------------------------------------------------------
        
        self.RGBFrame = rgbCapture.get_frame(self.frameNumber / self._framerate[self.videoNumber])  # Reads the following frame from the video capture
        
        if self._args["saving"]["segmentCotton"]:  # If the cotton segmentation mode was selected
            self.DEPTHFrame = depthCapture.get_frame(self.frameNumber / self._framerate[self.videoNumber])  # Reads the following frame from the video capture
            
        if self.videoNumber == self._nVideos:
            self._Stop = True
            
        if not self._Stop:
            self.curTime = int(self.frameNumber / self._framerate[self.videoNumber])  # Sets the time
            # If capture still has frames, and the following frame was successfully retrieved
            # ----------------------------------------------------------------------
                
            if self.videoNumber == 0:  # If the first video is being processed
                if self.frameNumber < self._tStart*self._framerate[self.videoNumber]:
                    pass
                elif self.frameNumber >= self._tStart * self._framerate[self.videoNumber] and self.frameNumber <= self._tEnd[self.videoNumber] * self._framerate[self.videoNumber] : #If the cotton was added, and if the video is not finished
                    self.RunSegmentations()
            elif self.videoNumber != 0:  # If one of the next videos is being processed
                if self.frameNumber <= self._tEnd[self.videoNumber]*self._framerate[self.videoNumber]:  # If the video is not finished
                    self.RunSegmentations()
                    
        self.fn += 1
        self.frameNumber += 1  # Increments the frame number variable

    def RunSegmentations(self):
        self.RunSegmentationMouse()  # Runs the mouse segmentation on the ROI
        
        if self._args["saving"]["segmentCotton"]:
            self.RegisterDepth(self.DEPTHFrame, self._resizingFactorRegistration,
                               self.start_X, self.end_X, self.start_Y, self.end_Y,
                               self.start_X_foreground, self.end_X_foreground,
                               self.start_Y_foreground, self.end_Y_foreground)  # Performs depth registration
            self.RunSegmentationCotton()  # Runs the cotton segmentation on the ROI
        
        if self._args["segmentation"]["showStream"] or self._args["saving"]["saveStream"]:
            self.CreateDisplay()  # Creates image for display/saving purposes
            
        if self._args["saving"]["saveStream"]:
            if not self._startSaving:
                self._startSaving = True
                        
        if self._args["saving"]["saveCottonMask"]:
            if not self._startSavingMask:
                self._startSavingMask = True
            
    def RunSegmentationMouse(self):
        # Selects only the ROI part of the image for future analysis
        # -------------------------------------------------------------------------------------------------------------
        self.cloneFrame = self.RGBFrame.copy()  # [DISPLAY ONLY] Creates a clone frame for display purposes
        self.cloneFrame = cv2.cvtColor(self.cloneFrame, cv2.COLOR_BGR2RGB)  # [DISPLAY ONLY] Changes the Frame to RGB for display purposes
        
        self.croppedFrame = self.RGBFrame[self.upLeftY:self.lowRightY, self.upLeftX:self.lowRightX]  # Crops the initial frame to the ROI
        self.maskDisplay = cv2.cvtColor(self.croppedFrame, cv2.COLOR_BGR2RGB)  # [DISPLAY ONLY] Changes the croppedFrame to RGB for display purposes
        
        # Filtering the ROI from noise
        # --------------------------------------------------------------------------------------------------------------
        self.hsvFrame = cv2.cvtColor(self.croppedFrame, cv2.COLOR_BGR2HSV)  # Changes the croppedFrame LUT to HSV for segmentation
        self.blur = cv2.blur(self.hsvFrame, (5, 5))  # Applies a Gaussian Blur to smoothen the image
        self.maskMouse = cv2.inRange(self.blur, self._args["segmentation"]["threshMinMouse"],
                                     self._args["segmentation"]["threshMaxMouse"])  # Thresholds the image to binary
        self.openingMouse = cv2.morphologyEx(self.maskMouse, cv2.MORPH_OPEN,
                                             self._args["segmentation"]["kernel"], iterations=1)  # Applies opening operation to the mask for dot removal
        self.closingMouse = cv2.morphologyEx(self.openingMouse, cv2.MORPH_CLOSE,
                                             self._args["segmentation"]["kernel"], iterations=1)  # Applies closing operation to the mask for large object filling
        self.cnts = cv2.findContours(self.closingMouse.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]  # Finds the contours of the image to identify the meaningful object
        
        # Finding the Mouse
        # --------------------------------------------------------------------------------------------------------------
        if self.cnts:  # If a contour is found in the binary mask
            self.biggestContour = max(self.cnts, key=cv2.contourArea)
            self.area = cv2.contourArea(self.biggestContour)  # Computes the area of the biggest object from the binary mask
            if self.area > self._args["segmentation"]["minAreaMask"] and self.area < self._args["segmentation"]["maxAreaMask"]:  # tmt_future computation is done only if the area of the detected object is meaningful
                ((self.x,self.y), self.radius) = cv2.minEnclosingCircle(self.biggestContour)
                self.M = cv2.moments(self.biggestContour)  # Computes the Moments of the detected object
                self.center = (int(self.M["m10"] / self.M["m00"]), int(self.M["m01"] / self.M["m00"]))  # Computes the Centroid of the detected object
                
                self.StorePosition()  # Stores the position and area of the detected object
            else:  # If the area of the detected object is too small...
                self.Error()  # Specify that no object was detected
                self.StorePosition()  # Stores old position and area
        else:  # If no contour was detected...
            self.Error()  # Specify that no object was detected
            self.area = 0  # Resets the area size to 0
            self.StorePosition()  # Stores old position and area
            
#        self.ComputeDistanceTraveled();

    def RunSegmentationCotton(self):
        # Filtering the ROI from noise
        # -----------------------------------------------------------------------------------------------------------

        self.maskCotton = cv2.inRange(self.blur, self._args["segmentation"]["threshMinCotton"],
                                      self._args["segmentation"]["threshMaxCotton"])  # Thresholds the image to binary
        self.openingCotton = cv2.morphologyEx(self.maskCotton,
                                              cv2.MORPH_OPEN, self._args["segmentation"]["kernel"], iterations=3)  # Applies opening operation to the mask for dot removal
        self.closingCotton = cv2.morphologyEx(self.openingCotton,
                                              cv2.MORPH_CLOSE, self._args["segmentation"]["kernel"], iterations=3)  # Applies closing operation to the mask for large object filling
        self.cntsCotton = cv2.findContours(self.closingCotton.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]  # Finds the contours of the image to identify the meaningful object
        self.cntsCottonFiltered = [cnt for cnt in self.cntsCotton if cv2.contourArea(cnt) >= self._args["segmentation"]["minCottonSize"]]
        self.finalmaskCotton = np.array(cv2.drawContours(np.zeros(self.closingCotton.shape),
                                                         self.cntsCottonFiltered, -1,
                                                         (255, 255, 255), cv2.FILLED), dtype="uint8")
                
#        self.cntNest = None #Variable that determines whether a nest is detected or not
#        self.cntLarge = []
#        self.largeObjects = 0 #Variable that holds the number of detected large cotton objects
#        self.smallObjects = 0 #Variable that holds the number of detected small cotton objects
        
        self.croppedRegisteredDepth = self.registeredDepth[self.upLeftY:self.lowRightY, self.upLeftX:self.lowRightX]
        self.bitwiseDepthCottonMask = cv2.bitwise_and(self.finalmaskCotton, self.croppedRegisteredDepth[:, :, 0])
        self.averagePixelIntensity = self.bitwiseDepthCottonMask[np.nonzero(self.bitwiseDepthCottonMask)]
        
        try:
            self.averagePixelIntensity = int(np.mean(self.averagePixelIntensity))
        except:
            self.averagePixelIntensity = 0
        
        self._cottonAveragePixelIntensities.append(self.averagePixelIntensity)
        
        # Connecting mask centers to estimate cotton spread
        # -------------------------------------------------------------------------------------------------------------
        
#        self.largestCotton = 0
#        self.indexLargestCotton = 0
#        self.cottonCenters = []
#        
#        for i,contour in enumerate(self.cntsCotton) :
#            area = cv2.contourArea(contour)
#            if area > self.largestCotton :
#                self.largestCotton = area
#                self.indexLargestCotton = i
#            
#        for i,contour in enumerate(self.cntsCotton):
#            area = cv2.contourArea(self.cntsCotton[i]) #Computes its area
#            if i != self.indexLargestCotton :
#                if area >= self._args["segmentation"]["minCottonSize"] and area < self._args["segmentation"]["nestCottonSize"] : #If the area in bigger than a certain threshold
#                    ((self.x,self.y), self.radius) = cv2.minEnclosingCircle(contour)
#                    center = (int(self.x),int(self.y))
#                    self.cottonCenters.append(center)
#                    
#                    self.largeObjects+=1 #Adds the object to the count of large cotton pieces
#                    self.cntLarge.append(i)
#                else : #If the area is smaller than a certain threshold
#                    self.smallObjects+=1 #Adds the object to the count of small cotton pieces
#                    
#            if i == self.indexLargestCotton :
#                if area >= self._args["segmentation"]["minCottonSize"] and area < self._args["segmentation"]["nestCottonSize"] : #If the area in bigger than a certain threshold
#                    ((self.x,self.y), self.radius) = cv2.minEnclosingCircle(contour)
#                    center = (int(self.x),int(self.y))
#                    self.cottonCenters.append(center)
#                    
#                    self.largeObjects+=1 #Adds the object to the count of large cotton pieces
#                    self.cntLarge.append(i)
#                
#                if area >= self._args["segmentation"]["nestCottonSize"] : #If the area has a size of a nest !
#                    ((self.x,self.y), self.radius) = cv2.minEnclosingCircle(contour)
#                    center = (int(self.x),int(self.y))
#                    self.cottonCenters.append(center)
#                    
#                    self.largeObjects+=1 #Adds the object to the count of large cotton pieces
#                    self.cntNest = i #Sets the self.cntNest variable to hold the position of the nest contour
#                else : #If the area is smaller than a certain threshold
#                    self.smallObjects+=1 #Adds the object to the count of small cotton pieces
#
#        self.closingCottonClone = self.closingCotton.copy()
#
#        for center in range(1,len(self.cottonCenters)) :
#            cv2.line(self.closingCottonClone, self.cottonCenters[center-1],\
#                     self.cottonCenters[center], 255, 10)
#        
#        self.cntsAllCottons = cv2.findContours(self.closingCottonClone.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]
#        
#        if self.cntsAllCottons != [] :
#            self.biggestContourCotton = max(self.cntsAllCottons, key=cv2.contourArea) #Finds the biggest contour of the binary mask
#            
#            ((self.mainX,self.mainY), self.mainRadius) = cv2.minEnclosingCircle(self.biggestContourCotton)
#        
#            if self.mainX and self.mainY != None :
#                self.correctedCenterCotton = (int(self.mainX)+self.upLeftX,\
#                                              int(self.mainY)+self.upLeftY)
#        
#            self._cottonCenter.append((self.mainX,self.mainY))
#            self._cottonSpread.append(self.mainRadius)
#
#        self._largeObjects.append(self.largeObjects)
#        self._smallObjects.append(self.smallObjects)
#        
#        if self.cntNest == None :
#            self._detectedNest.append(False)
#        else :
#            self._detectedNest.append(True)
                
    def RegisterDepth(self, depthFrame, resizingFactorRegistration, start_X, end_X, start_Y, end_Y, start_X_foreground, end_X_foreground, start_Y_foreground, end_Y_foreground) :
        self.depthFrame = cv2.resize(depthFrame, (0, 0), fx=resizingFactorRegistration, fy=resizingFactorRegistration)
            
        self.registeredDepth = np.zeros([self._H_RGB, self._W_RGB, 3], dtype=np.uint8)
        
        self.blend = cv2.addWeighted(self.depthFrame[start_X_foreground:end_X_foreground, start_Y_foreground:end_Y_foreground, :],
                        1,
                        self.registeredDepth[start_X:end_X, start_Y:end_Y, :],
                        0,
                        0,
                        self.registeredDepth)
                                
        self.registeredDepth[start_X:end_X, start_Y:end_Y, :] = self.blend
            
    def CreateDisplay(self):
        self.metaDataDisplay = np.zeros((self._H_RGB, self._metaDataCanvasSize, 3), np.uint8)  # Creates a canvas to write useful info
        self.WriteDisplay(self.metaDataPos1)
        
        # --------------------------------------------------------------------------------------------------------------
        # Draw text, contours, mouse tracker on the frames
            
        self.cloneFrame[self.upLeftY:self.upLeftY+self.maskDisplay.shape[0],\
                        self.upLeftX:self.upLeftX+self.maskDisplay.shape[1]] = self.maskDisplay
                        
        cv2.circle(self.cloneFrame, self.correctedCenter, self.centerSize, (0, 0, 255), self.centerThickness)  # Draws a the object Centroid as a point
        
        # if self.correctedCenterCotton != None :
        #    cv2.circle(self.cloneFrame, self.correctedCenterCotton, int(self.mainRadius), (255,0,255), self.contourThickness)
 
        cv2.rectangle(self.cloneFrame, (self.upLeftX, self.upLeftY), (self.lowRightX, self.lowRightY), (255, 0, 0), self.contourThickness)  # Displays the ROI square on the image
        
        # self.hStack = np.hstack((self.metaDataDisplay, self.cloneFrame))  # Generates a horizontal stacked image with useful info and stream
        
        self.hStack = self.cloneFrame[self.upLeftY:self.lowRightY, self.upLeftX:self.lowRightX]  # Crops the initial frame to the ROI
        self.hStack = cv2.resize(self.hStack, (0, 0),
                                 fx=1./self._args["saving"]["resizeTracking"],
                                 fy=1./self._args["saving"]["resizeTracking"])
        
    def WriteDisplay(self, pos):
        # Writes the fisrt information
        # ----------------------------------------------------------------------
        cv2.putText(self.metaDataDisplay,
                    "Tracking Mouse : {0}".format(self._args["main"]["mouse"]),
                    (8,pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.textFontSize,
                    (255,255,255),
                    self.textFontThickness) #Displays the mouse name
        
        # Writes the second information
        # ----------------------------------------------------------------------
        # cv2.putText(self.metaDataDisplay,
        #            "mask size : {0}px2".format(int(self.area)),
        #            (8,pos),
        #            cv2.FONT_HERSHEY_SIMPLEX,
        #            self.textFontSize,
        #            (255,255,255),
        #            self.textFontThickness) #Displays the mask size in real-time
        
        cv2.putText(self.metaDataDisplay,
                    "dist : {0}cm".format(str("%.2f" % self._distance)),
                    (8, pos+70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.textFontSize,
                    (255, 255, 255),
                    self.textFontThickness)  # Displays the distance traveled by the mouse in real-time
        
        # Writes the third information
        # ----------------------------------------------------------------------
        cv2.putText(self.metaDataDisplay,
                    "speed : {0}cm/s".format(str("%.2f" % self.realTimeSpeed)),
                    (8, pos + 70 * 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.textFontSize,
                    (255, 255, 255),
                    self.textFontThickness)  # Displays the mouse speed in real-time
                    
        # Writes the fourth information
        # ----------------------------------------------------------------------
        
        # If a mouse is detected
        if self._args["segmentation"]["minAreaMask"] < self.area < self._args["segmentation"]["maxAreaMask"]:
            cv2.putText(self.metaDataDisplay,
                        "Mouse Detected",
                        (8, pos + 70 * 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.textFontSize,
                        (0, 255, 0),
                        self.textFontThickness)  # Displays the current status of mouse detection
            cv2.drawContours(self.maskDisplay, [self.biggestContour], 0, (0, 0, 255), self.contourThickness)  # Draws the contour of the detected object on the image
        else:  # If no mouse is detected
            cv2.putText(self.metaDataDisplay,
                        "No Mouse",
                        (8, pos + 70 * 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.textFontSize,
                        (0, 0, 255),
                        self.textFontThickness)  # Displays the current status of mouse detection
                    
        # if self._args["saving"]["segmentCotton"] :
        #    #Writes the fifth information
        #    #----------------------------------------------------------------------
        #    cv2.putText(self.metaDataDisplay,
        #                '{0} obj; height : {1}'.format(self.largeObjects,self.averagePixelIntensity),
        #                (8,pos+70*4),
        #                cv2.FONT_HERSHEY_SIMPLEX,
        #                self.textFontSize,
        #                (255,255,255),
        #                self.textFontThickness) #Displays the number of cotton objects detected
                    
            # Writes the sixth information
            # ----------------------------------------------------------------------
            
            # #If a nest is detected
            # if self.cntNest != None :
            #    cv2.putText(self.metaDataDisplay,
            #                'Nest Detected',
            #                (8,pos+70*5),
            #                cv2.FONT_HERSHEY_SIMPLEX,
            #                self.textFontSize,
            #                (0,255,0),
            #                self.textFontThickness) #Displays the current status of nest detection
            #
            #    cv2.drawContours(self.maskDisplay, self.cntsCotton, self.cntNest, (255,255,0), self.contourThickness)  # Draws the contour of the nest
            #
            #    self.largeContours = [self.cntsCotton[p] for p in self.cntLarge];
            #    cv2.drawContours(self.maskDisplay, self.largeContours, -1, (0,255,0), self.contourThickness);
        if self._args["saving"]["segmentCotton"]:
            cv2.drawContours(self.maskDisplay, self.cntsCottonFiltered, -1, (0,255,0), self.contourThickness)
            # If no nest is detected
            # else:
            #    cv2.putText(self.metaDataDisplay,
            #                'No Nest',
            #                (8,pos+70*5),
            #                cv2.FONT_HERSHEY_SIMPLEX,
            #                self.textFontSize,
            #                (0,0,255),
            #                self.textFontThickness); #Displays the current status of nest detection
            #
            #    self.largeContours = [self.cntsCotton[p] for p in self.cntLarge];
            #    cv2.drawContours(self.maskDisplay, self.largeContours, -1, (0,255,0), self.contourThickness);
                
                # for cnt in self.cntLarge:
                #    cv2.drawContours(self.maskDisplay, self.cntsCotton, cnt, (0,255,0), self.contourThickness)  # Draws all the other cotton contours that are not the nest
                # cv2.drawContours(self.maskDisplay, self.cntsCotton, -1, (0,255,0), self.contourThickness)  # Draws all the cotton contours that are not the nest
        
    def SaveTracking(self):
        if self._startSaving:
            if self._args["saving"]["fourcc"] is None:  # TODO: check
                self.hStack = cv2.cvtColor(self.hStack, cv2.COLOR_BGR2RGB)
#                self.hStack = cv2.cvtColor(self.hStack, cv2.COLOR_RGB2BGR)
                self.videoWriter.writeFrame(self.hStack)
            elif self._args["saving"]["fourcc"] == "Frame":
                self.hStack = cv2.cvtColor(self.hStack, cv2.COLOR_BGR2RGB)
                self.hStack = cv2.cvtColor(self.hStack, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(self._args["main"]["segmentationDir"], "{0}.jpg".format(self.fn)), self.hStack)
            else:
               if self._displayOutputShape:
                   print(self.hStack.shape)
                   self._displayOutputShape = False
               self.videoWriter.write(self.hStack)
            
        if self._startSavingMask:
            self.bitwiseDepthCottonMaskSave = cv2.cvtColor(self.bitwiseDepthCottonMask, cv2.COLOR_GRAY2RGB)
            if self._args["saving"]["fourcc"] is None:  # TODO: check
                self.depthMaskWriter.writeFrame(self.bitwiseDepthCottonMaskSave)
            elif self._args["saving"]["fourcc"] == "Frame":
                pass
            else:
                self.depthMaskWriter.write(self.bitwiseDepthCottonMaskSave)
    
    # def ComputeDistanceTraveled(self) :
    #    if len(self.realTimePosition) == 2 and not None in self.realTimePosition : #Runs only if two positions are available and if previous or current position are not None
    #            self.realTimeSpeed = sqrt((self.realTimePosition[1][0]-self.realTimePosition[0][0])**2+\
    #                                  (self.realTimePosition[1][1]-self.realTimePosition[0][1])**2)/self.distanceRatio #Computes distance
    #
    #            self._distances.append(self.realTimeSpeed)
    #            if self.realTimeSpeed >= self._args["segmentation"]["minDist"] : #If the distance is higher than the minimal distance (filtering for movement noise)
    #                self._distance += (self.realTimeSpeed) #Adds the value to the cumulative distance varaible

    def UpdatePosition(self):
        if len(self.realTimePosition) == 0:
            self.realTimePosition.append(self.center)
        elif len(self.realTimePosition) == 1:
            self.realTimePosition.append(self.center)
        elif len(self.realTimePosition) == 2:
            self.realTimePosition[0] = self.realTimePosition[1]
            self.realTimePosition[1] = self.center
            
    def StorePosition(self):
        self._positions.append(self.center)
        self._maskAreas.append(self.area)
        if self.center is not None:  # TODO: check
            self.UpdatePosition()
            self.correctedCenter = (self.center[0] + self.upLeftX,
                                    self.center[1] + self.upLeftY)
            
    def Error(self):
        self._errors += 1
        if self._args["segmentation"]["showStream"]:
            if self._errors == 1:
                print("[WARNING] No contour detected, assuming old position !")
            elif self._errors % 100 == 0:
                print("[WARNING] No contour detected, assuming old position !")
                
    def ReturnTracking(self):
        if self._args["segmentation"]["showStream"]:
            try:
                self.newhStack = cv2.cvtColor(self.hStack, cv2.COLOR_BGR2RGB)
                self.newhStack = cv2.resize(self.newhStack, (0, 0),
                                            fx=1./self._args["segmentation"]["resize_stream"],
                                            fy=1./self._args["segmentation"]["resize_stream"])
                return self.newhStack
            except:
                return []
        else:
            return []
        
    def SendMail(self):
        self.now = time.localtime(time.time())
    
        msg = MIMEMultipart()
        msg['Subject'] = 'Mouse {0} analysis completed on {1}!'.format(self._mouse, self._args["main"]["workingStation"])
        msg['From'] = self._args["main"]["email"]
        msg['To'] = self._args["main"]["email"]
    
        s = smtplib.SMTP(self._args["main"]["smtp"], self._args["main"]["port"])
        s.ehlo()
        s.starttls()
        s.ehlo()
        s.login(self._args["main"]["email"], self._args["main"]["password"])
        s.sendmail(self._args["main"]["email"], self._args["main"]["email"], msg.as_string())
        s.quit()


def TopTracker(Tracker,**kwargs):
    if not Tracker._break:
        print("\n")
        utils.PrintColoredMessage("##################################################################################################################","darkgreen")
        utils.PrintColoredMessage("                                  [INFO] Starting segmentation for mouse {0}".format(kwargs["main"]["mouse"]),"darkgreen")
        utils.PrintColoredMessage("##################################################################################################################","darkgreen")
    else:
        print("\n")
        utils.PrintColoredMessage("##################################################################################################################","magenta")
        utils.PrintColoredMessage("                                  [INFO] Resuming segmentation for mouse {0}".format(kwargs["main"]["mouse"]),"magenta")
        utils.PrintColoredMessage("##################################################################################################################","magenta")
    
    Tracker._break = False
    Tracker._Start = time.time()
    startTime = time.localtime()
    h, m, s = startTime.tm_hour, startTime.tm_min, startTime.tm_sec
    Tracker.SetTrackingVideoSaving()
    utils.PrintColoredMessage("[INFO] Segmentation started at : {0}h {1}m {2}s".format(h,m,s),"darkgreen")
    
    nFrames = sum([x*y for x,y in zip(Tracker._tEnd, Tracker._framerate)]) - (Tracker._tStart * Tracker._framerate[0])
    
    if kwargs["saving"]["segmentCotton"]:
        for rgbCapture, depthCapture in zip(kwargs["main"]["capturesRGB"], kwargs["main"]["capturesDEPTH"]):
            try:
                while True:
                    # Charges a new frame and runs the segmentation
                    # ----------------------------------------------------------------------
                    Tracker.Main(rgbCapture, depthCapture)
                    # print(Tracker.frameNumber)
                    if not Tracker._Stop :
                        # If the tracking has to be saved
                        # ----------------------------------------------------------------------
                        if kwargs["saving"]["saveStream"]:
                            Tracker.SaveTracking()
                        
                        # If the tracking has to be displayed
                        # ----------------------------------------------------------------------
                        if kwargs["segmentation"]["showStream"]:
                            segmentation = Tracker.ReturnTracking()
                            if segmentation:
                                cv2.imshow('segmentation', segmentation)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    break
                        
                        # If the video is not empty
                        # ----------------------------------------------------------------------
                        if Tracker._tEnd[Tracker.videoNumber] != 0:
                            # VERBOSE
                            # Runs only every 1 % of the video being analyzed
                            # -----------------------------------------------------------------------------------------
                            if Tracker.fn % (int(0.01*nFrames)) == 0:
                                print('\n')
                                utils.PrintColoredMessage('Loaded and analyzed : '+str(Tracker.fn)+'/'+str( int(nFrames) )+\
                                    ' = '+ str ( round ( (float(Tracker.fn) / float( (nFrames) )) *100))\
                                    +'% frames', "darkgreen")
                                                          
                                utils.PrintColoredMessage(utils.PrintLoadingBar( round ( (float(Tracker.fn) / float(nFrames)) *100 )), "darkgreen")
                            
                            # Runs only if the video is finished
                            # ----------------------------------------------------------------------------------------
                            if Tracker.frameNumber == int(Tracker._tEnd[Tracker.videoNumber] * Tracker._framerate[Tracker.videoNumber]):
                                if kwargs["main"]["playSound"]:
                                    utils.PlaySound(2, params.sounds['Purr'])  # Plays sound when code finishes
                                
                                Tracker.videoNumber += 1  # Increments videoNumber variable to keep track which video is being processed
                                Tracker.frameNumber = 0
                                break
                    else:  # If Tracker._Stop :
                        try:
                            if kwargs["main"]["email"] is not None:
                                Tracker.SendMail()
                        except:
                            utils.PrintColoredMessage("[INFO] Sending email to {0} failed".format(kwargs["main"]["email"]), "darkred")
                    
                        if kwargs["main"]["playSound"]:
                            utils.PlaySound(2, kwargs["main"]["sound2Play"])  # Plays sound when code finishes
                            
                        Tracker.videoNumber += 1  # Increments videoNumber variable to keep track which video is being processed
                        Tracker.frameNumber = 0
                        break
                                
            except KeyboardInterrupt:
                Tracker._break = True
                break
                
    else:
        for rgbCapture in kwargs["main"]["capturesRGB"]:
            try:
                while True:
                    # Charges a new frame and runs the segmentation
                    # ----------------------------------------------------------------------
                    Tracker.Main(rgbCapture, None)
                    # print(Tracker.frameNumber)
                    if not Tracker._Stop:
                        # If the tracking has to be saved
                        # ----------------------------------------------------------------------
                        if kwargs["saving"]["saveStream"]:
                            Tracker.SaveTracking()
                        
                        # If the tracking has to be displayed
                        # ----------------------------------------------------------------------
                        if kwargs["segmentation"]["showStream"]:
                            segmentation = Tracker.ReturnTracking()
                            if segmentation:
                                cv2.imshow('segmentation', segmentation)
                                if cv2.waitKey(1) & 0xFF == ord('q'):
                                    Tracker._break = True
                                    break
                        
                        # If the video is not empty
                        # ----------------------------------------------------------------------
                        if Tracker._tEnd[Tracker.videoNumber] != 0:
                            # VERBOSE
                            # Runs only every 1 % of the video being analyzed
                            # ----------------------------------------------------------------------------------------
                            if Tracker.fn % (int(0.01*nFrames)) == 0 :
                                print('\n')
                                utils.PrintColoredMessage('Loaded and analyzed : '+str(Tracker.fn)+'/'+str( int(nFrames) )+\
                                    ' = '+ str ( round ( (float(Tracker.fn) / float( (nFrames) )) *100))\
                                    +'% frames', "darkgreen")
                                utils.PrintColoredMessage(utils.PrintLoadingBar( round ( (float(Tracker.fn) / float(nFrames)) *100 )),"darkgreen")
                            
                            # Runs only if the video is finished
                            # -----------------------------------------------------------------------------------------
                            if Tracker.frameNumber == int(Tracker._tEnd[Tracker.videoNumber] * Tracker._framerate[Tracker.videoNumber]):
                                if kwargs["main"]["playSound"]:
                                    utils.PlaySound(2, params.sounds['Purr'])  # Plays sound when code finishes
                                Tracker.videoNumber += 1  # Increments videoNumber variable to keep track which video is being processed
                                Tracker.frameNumber = 0
                                break
    
                    else:  # If Tracker._Stop :
                        try:
                            if kwargs["main"]["email"] is not None:
                                Tracker.SendMail()
                        except:
                            utils.PrintColoredMessage("[INFO] Sending email to {0} failed".format(kwargs["main"]["email"]), "darkred")
                    
                        if kwargs["main"]["playSound"]:
                            utils.PlaySound(2, kwargs["main"]["sound2Play"])  # Plays sound when code finishes
                            
                        Tracker.videoNumber += 1  # Increments videoNumber variable to keep track which video is being processed
                        Tracker.frameNumber = 0
                        break
            except KeyboardInterrupt:
                Tracker._break = True
                break

    if not Tracker._break:
        print('\n')
        utils.PrintColoredMessage('##################################################################################################################', "darkgreen")
        utils.PrintColoredMessage("                             [INFO] Mouse {1} has been successfully analyzed".format(str(Tracker.videoNumber),Tracker._mouse), "darkgreen")
        utils.PrintColoredMessage('##################################################################################################################', "darkgreen")
    else :
        print('\n')
        utils.PrintColoredMessage('##################################################################################################################', "darkred")
        utils.PrintColoredMessage("                             [INFO] Tracking for Mouse {1} has been aborted".format(str(Tracker.videoNumber),Tracker._mouse), "darkred")
        utils.PrintColoredMessage('##################################################################################################################', "darkred")

    try:
        Tracker.SendMail()
    except:
        utils.PrintColoredMessage("[INFO] Sending email to {0} failed".format(kwargs["main"]["email"]), "darkred")
              
    if kwargs["segmentation"]["showStream"]:
        cv2.destroyAllWindows()
    
    if kwargs["saving"]["saveStream"]:
        if kwargs["saving"]["fourcc"] is None:  #  TODO: check
            Tracker.videoWriter.close()
        elif kwargs["saving"]["fourcc"] == "Frame":
            pass
        else:
            Tracker.videoWriter.release()
        
    if kwargs["saving"]["saveCottonMask"]:
        if kwargs["saving"]["fourcc"] is None:
            Tracker.depthMaskWriter.close()
        elif kwargs["saving"]["fourcc"] == "Frame":
            pass
        else:
            Tracker.depthMaskWriter.release()
        
    Tracker._End = time.time()
    
    diff = Tracker._End - Tracker._Start
    h, m, s = utils.HoursMinutesSeconds(diff)
    
    print("\n")
    utils.PrintColoredMessage("[INFO] Segmentation started at : {0}h {1}m {2}s".format(h, m, s), "darkgreen")
    
    SaveTracking(Tracker, **kwargs)


def SaveTracking(Tracker, **kwargs):
    first = 0
    for point in Tracker._positions:
        if point is None:
            first += 1
    
    for n, point in enumerate(Tracker._positions):
        if point is None:
            Tracker._positions[n] = Tracker._positions[first]
            
    metaDataString = "Segmentation_MetaData_{0}".format(Tracker._mouse)
    metaDataFile = os.path.join(kwargs["main"]["resultDir"], '{0}.xls'.format(metaDataString))
    
    metaData = xlwt.Workbook()
    sheet = metaData.add_sheet("MetaData")
    
    sheet.write(0, 0, "Mouse")
    sheet.write(0, 1, kwargs["main"]["mouse"])
    
    sheet.write(1, 0, "refPt")
    sheet.write(1, 1, "[({0},{1}),({2},{3})]".format(Tracker._refPt[0][0],\
                        Tracker._refPt[0][1], Tracker._refPt[1][0], Tracker._refPt[1][1]))
    
    sheet.write(2, 0, "ThreshMinMouse")
    sheet.write(2, 1, "[{0},{1},{2}]".format(kwargs["segmentation"]["threshMinMouse"][0],
                                             kwargs["segmentation"]["threshMinMouse"][1],
                                             kwargs["segmentation"]["threshMinMouse"][2]))
    
    sheet.write(3, 0, "ThreshMaxMouse")
    sheet.write(3, 1, "[{0},{1},{2}]".format(kwargs["segmentation"]["threshMaxMouse"][0],
                                             kwargs["segmentation"]["threshMaxMouse"][1],
                                             kwargs["segmentation"]["threshMaxMouse"][2]))
    
    sheet.write(4, 0, "threshMinCotton")
    sheet.write(4, 1, "[{0},{1},{2}]".format(kwargs["segmentation"]["threshMinCotton"][0],
                                             kwargs["segmentation"]["threshMinCotton"][1],
                                             kwargs["segmentation"]["threshMinCotton"][2]))
    
    sheet.write(5, 0, "threshMaxCotton")
    sheet.write(5, 1, "[{0},{1},{2}]".format(kwargs["segmentation"]["threshMaxCotton"][0],
                                             kwargs["segmentation"]["threshMaxCotton"][1],
                                             kwargs["segmentation"]["threshMaxCotton"][2]))
    
    sheet.write(6, 0, "ElapsedTime")
    sheet.write(6, 1, Tracker._End-Tracker._Start)
    
    sheet.write(7, 0, "Errors")
    sheet.write(7, 1, Tracker._errors)
    
    metaData.save(metaDataFile)
    
    np.save(os.path.join(kwargs["main"]["resultDir"], 'Data_'+str(Tracker._mouse) + '_refPt.npy'), Tracker._refPt)
    np.save(os.path.join(kwargs["main"]["resultDir"], 'Data_'+str(Tracker._mouse) + '_Points.npy'), Tracker._positions)
    # np.save(os.path.join(kwargs["main"]["resultDir"],'Data_'+str(Tracker._mouse)+'_Areas.npy'),Tracker._maskAreas)
    # np.save(os.path.join(kwargs["main"]["resultDir"],'Data_'+str(Tracker._mouse)+'_Distances.npy'),Tracker._distances)
    
    if params.savingParameters["segmentCotton"] :
        np.save(os.path.join(kwargs["main"]["resultDir"], 'Data_'+str(Tracker._mouse) + '_CottonPixelIntensities.npy'), Tracker._cottonAveragePixelIntensities)
    # np.save(os.path.join(kwargs["main"]["resultDir"],'Data_'+str(Tracker._mouse)+'_CottonSpread.npy'),Tracker._cottonSpread)
