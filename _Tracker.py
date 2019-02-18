#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  7 14:03:44 2018

@author: tomtop
"""

import os;
import cv2;
import time;
import numpy as np;
import pandas as pd;
import fnmatch;
import xlwt;
from math import sqrt;

import TopMouseTracker.Parameters as params;
import TopMouseTracker.Utilities as utils;
import TopMouseTracker.IO as IO;

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
    
    def __init__(self,**kwargs) :
        
        #General variables
        #----------------------------------------------------------------------
        self._args = kwargs; #Loads the main arguments
        self._Start = None; #Time at which the segmentation starts
        self._Stop = False; #Trigger to stop segmentation when video is empty
        self._End = None; #Time at which the segmentation ends
        self._mouse = self._args["main"]["mouse"]; #Loads the name of the mouse
        self._cageWidth = self._args["segmentation"]["cageWidth"];
        self._cageLength = self._args["segmentation"]["cageLength"];
        self._testFrameRGB = self._args["main"]["testFrameRGB"][0].copy(); #Loads a test RGB frame
        self._H_RGB, self._W_RGB = self._testFrameRGB.shape[0], self._testFrameRGB.shape[1]; #Height, Width of the RGB frames
        self._testFrameDEPTH = self._args["main"]["testFrameDEPTH"][0].copy(); #Loads a test DEPTH frame
        self._H_DEPTH, self._W_DEPTH = self._testFrameDEPTH.shape[0], self._testFrameDEPTH.shape[1]; #Height, Width of the DEPTH frames
        
        #Registration variables
        #---------------------------------------------------------------------- 
        self._resizingFactorRegistration = 2.95; #Parameters for resizing of the depth image for registration
        self._testFrameDEPTHResized = cv2.resize(self._testFrameDEPTH,(0,0),fx = self._resizingFactorRegistration,fy = self._resizingFactorRegistration);
        self._H_DEPTH_RESIZED,self._W_DEPTH_RESIZED = self._testFrameDEPTHResized.shape[0],self._testFrameDEPTHResized.shape[1]; 
        self.SetRegistrationParameters(); #Computes all the placement parameters for registration
        
        #MetaData variables
        #---------------------------------------------------------------------- 
        self.SetMetaDataParameters();

        #Global tracking variables
        #----------------------------------------------------------------------
        self._positions = []; #Position of the mouse every s/frameRate
        self._maskAreas = []; #Size of the mouse mask every s/frameRate
        self._distances = []; #Distances traveled by the mouse every s/frameRate
        self._distance = 0.; #Cumulative distance traveled by the mouse
        self._errors = 0; #Counter for the number of times that the trackers fails to segment the animal
        self._cottonAveragePixelIntensities = []; #List containing all the depth information of segmented objects coming from the cotton mask
        
        #Real-Time tracking variables
        #----------------------------------------------------------------------
        self.videoNumber = 0;
        self.frameNumber = 0;
        self.realTimePosition = []; #Position of the mouse in real time
        self.realTimeSpeed = 0.; #Speed of the mouse in real time
        self.center = None; #Centroid (x,y) of the mouse binary mask in real time
        self.correctedCenter = None; #Corrected centroid (x,y) of the mouse binary mask in real time
        
        #Tracking canvas variables
        #----------------------------------------------------------------------
        self._metaDataCanvasSize = 750; #Size of the canvas
        self.textFontSize = 1.5; #font size of the text
        self.textFontThickness = 3; #thickness of the text
        self.contourThickness = 2; #thickness of the contours
        self.centerSize = 3; #size of the object centroid
        self.centerThickness = 5; #thickness of the object centroid
        self.metaDataPos1 = 50; #position of the metadata
        
        #Saving variables
        #----------------------------------------------------------------------
        self._startSaving = False;
        self._startSavingMask = False;
        
        if self._args["saving"]["saveStream"] :
            
            self.videoString = "Tracking_{0}.avi".format(self._mouse);
            
            self.testCanvas = np.zeros((self._W_RGB+self._metaDataCanvasSize,self._H_RGB));
            self.testCanvas = cv2.resize(self.testCanvas, (0,0),\
                                 fx = 1./self._args["saving"]["resizeTracking"],\
                                 fy = 1./self._args["saving"]["resizeTracking"]);
                                         
            self.videoWriter = cv2.VideoWriter(os.path.join(self._args["main"]["workingDir"],\
                                self.videoString), self._args["saving"]["fourcc"], self._framerate,\
                                (self.testCanvas.shape[0],self.testCanvas.shape[1]));
    
    def SetRegistrationParameters(self) : 
        
        self.start_X = 0;
        self.end_X = self.start_X+self._H_DEPTH_RESIZED;
        self.start_Y = 290;
        self.end_Y = self.start_Y+self._W_DEPTH_RESIZED;
        
        if self.end_X >= self._H_RGB :
            
            self.end_X_foreground = self._H_RGB-self.start_X;
        else :
            
            self.end_X_foreground = self._H_RGB;
            
        if self.end_Y >= self._W_RGB :
            
            self.end_Y_foreground = self._W_RGB-self.start_Y;
        else : 
            
            self.end_Y_foreground = self._W_RGB;
        
        self.start_X_foreground = 75;
        self.start_Y_foreground = 0;
        
        if self.start_X_foreground != 0 :
            
            self.end_X = self.end_X-self.start_X_foreground;
            self.end_X_foreground = self.end_X_foreground+self.start_X_foreground;
            
        if self.start_Y_foreground != 0 :
            
            self.end_Y = self.end_Y-self.start_Y_foreground;
            self.end_Y_foreground = self.end_Y_foreground+self.start_Y_foreground;
            
    def SetMetaDataParameters(self) :
        
        for file in os.listdir(self._args["main"]["workingDir"]) :
            
            path2File = os.path.join(self._args["main"]["workingDir"],file);
            
            if fnmatch.fnmatch(file, 'Mice_Video_Info.xlsx'): 

                videoInfoWorkbook = pd.read_excel(path2File,header=None); #Load video info excel sheet
                
            if fnmatch.fnmatch(file, 'MetaData*.xls'): 
                
                metaDataWorkbook = pd.read_excel(path2File,header=None); #Load video info excel sheet
        
        videoInfo = videoInfoWorkbook.as_matrix(); #Transforms it into a matrix
        
        for line in videoInfo :
            
            if not np.isnan(line[0]) : 
                
              if str(int(line[0])) == self._args["main"]["mouse"] :
                  
                  self._tStart = int(line[1]); #Moment at which the cotton is added (s)
                  self._tStartBehav = int(line[2]); #Moment at which the mouse start nest-building (s)
                  self._tEnd = [int(line[3]),int(line[4]),\
                               int(line[5]),int(line[6])]; #Length of each video of the experiment (max 4 videos)
        
        self._Length = sum(self._tEnd);
        self._nVideos = 0; #Variable holding the number of videos to be analyzed
        
        for i in self._tEnd :
            if i != 0 :
                self._nVideos+=1;
        
        metaData = metaDataWorkbook.as_matrix(); #Transforms it into a matrix
        
        for line in metaData :
            
            head = str(line[0]);
            
            if head == "Time_Stamp" :
                
                self._date = line[1];
                
            elif head == "Elapsed_Time" :
                
                self._elapsedTime = line[1];
                
            elif head == "Real_Framerate" :
                
                self._framerate = line[1];
                
            elif head == "nFrames" :
                
                self._nFrames = line[1];
        
    def SetROI(self) :
        
        self._refPt = IO.CroppingROI(self._args["main"]["testFrameRGB"][0].copy()).roi(); #Defining the ROI for segmentation
        
        self.upLeftX = int(self._refPt[0][0]); #Defines the Up Left ROI corner X coordinates
        self.upLeftY = int(self._refPt[0][1]); #Defines the Up Left ROI corner Y coordinates
        self.lowRightX = int(self._refPt[1][0]); #Defines the Low Right ROI corner X coordinates
        self.lowRightY = int(self._refPt[1][1]); #Defines the Low Right ROI corner Y coordinates
        
        self.ROIWidth = abs(self.lowRightX-self.upLeftX);
        self.ROILength = abs(self.lowRightY-self.upLeftY);
        
        self.distanceRatio = (abs(self.upLeftX-self.lowRightX)/self._args["segmentation"]["cageLength"]+\
                              abs(self.upLeftY-self.lowRightY)/self._args["segmentation"]["cageWidth"])/2; #Defines the resizing factor for the cage
                              
        self._testFrameRGBCropped = self._testFrameRGB[self.upLeftY:self.lowRightY,self.upLeftX:self.lowRightX];
        self._H_RGB_CROPPED, self._W_RGB_CROPPED = self._testFrameRGBCropped.shape[0], self._testFrameRGBCropped.shape[1];
        
        if self._args["saving"]["saveCottonMask"] :
            
            self.depthMaskString = "Mask_Cotton_{0}.avi".format(self._mouse);
            self.depthMaskWriter = cv2.VideoWriter(os.path.join(self._args["main"]["workingDir"],\
                                self.depthMaskString), self._args["saving"]["fourcc"], self._framerate,\
                                (self._W_RGB_CROPPED, self._H_RGB_CROPPED));
        
        
    def Main(self):
        
        #Get frame from capture
        #----------------------------------------------------------------------
        
        try :
            
            self.RGBFrame = next(self._args["main"]["capturesRGB"][self.videoNumber]); #Reads the following frame from the video capture
            self.DEPTHFrame = next(self._args["main"]["capturesDEPTH"][self.videoNumber]); #Reads the following frame from the video capture
            
        except StopIteration :
            
            self._Stop = True;
        
        if not self._Stop :
            
            self.frameNumber += 1; #Increments the frame number variable
            self.curTime = self.frameNumber/self._framerate; #Sets the time
               
            #If capture still has frames, and the following frame was successfully retrieved
            #----------------------------------------------------------------------
                
            if self.videoNumber == 0 : #If the first video is being processed
                
                if self.curTime >= self._tStart and self.curTime <= self._tEnd[self.videoNumber] : #If the cotton was added, and if the video is not finished
                    
                    self.RunSegmentations();
    
            elif self.videoNumber != 0 : #If the one of the next videos is being processed
                
                if self.curTime <= self._tEnd[self.videoNumber] : #If the video is not finished
                    
                    self.RunSegmentations();
                    
                    
    def RunSegmentations(self) :
        
        self.RunSegmentationMouse(); #Runs the mouse segmentation on the ROI
        
        if self._args["saving"]["segmentCotton"] :
        
            self.RegisterDepth();
            self.RunSegmentationCotton(); #Runs the cotton segmentation on the ROI
        
        if self._args["display"]["showStream"] or self._args["saving"]["saveStream"] :

            self.CreateDisplay();
            
        if self._args["saving"]["saveStream"] :
                
            if not self._startSaving :
                
                self._startSaving = True;
                        
        if self._args["saving"]["saveCottonMask"] :
        
            if not self._startSavingMask :
                
                self._startSavingMask = True;
     
            
    def RunSegmentationMouse(self) :
        
        
        #Selects only the ROI part of the image for future analysis
        #----------------------------------------------------------------------------------------------------------------------------------
        
        self.cloneFrame = self.RGBFrame.copy(); #[DISPLAY ONLY] Creates a clone frame for display purposes
        self.cloneFrame = cv2.cvtColor(self.cloneFrame, cv2.COLOR_BGR2RGB); #[DISPLAY ONLY] Changes the Frame to RGB for display purposes
        
        self.croppedFrame = self.RGBFrame[self.upLeftY:self.lowRightY,self.upLeftX:self.lowRightX]; #Crops the initial frame to the ROI
        self.maskDisplay = cv2.cvtColor(self.croppedFrame, cv2.COLOR_BGR2RGB); #[DISPLAY ONLY] Changes the croppedFrame to RGB for display purposes
        
        #Filtering the ROI from noise
        #----------------------------------------------------------------------------------------------------------------------------------
        
        self.hsvFrame = cv2.cvtColor(self.croppedFrame, cv2.COLOR_BGR2HSV); #Changes the croppedFrame LUT to HSV for segmentation
        self.blur = cv2.blur(self.hsvFrame,(5,5)); #Applies a Gaussian Blur to smoothen the image
        self.maskMouse = cv2.inRange(self.blur, self._args["segmentation"]["threshMinMouse"], self._args["segmentation"]["threshMaxMouse"]); #Thresholds the image to binary
        self.openingMouse = cv2.morphologyEx(self.maskMouse,cv2.MORPH_OPEN,self._args["segmentation"]["kernel"], iterations = 1); #Applies opening operation to the mask for dot removal
        self.closingMouse = cv2.morphologyEx(self.openingMouse,cv2.MORPH_CLOSE,self._args["segmentation"]["kernel"], iterations = 1); #Applies closing operation to the mask for large object filling
        self.cnts = cv2.findContours(self.closingMouse.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]; #Finds the contours of the image to identify the meaningful object
        
        #Finding the Mouse
        #----------------------------------------------------------------------------------------------------------------------------------
        
        if self.cnts != [] : #If a contour is found in the binary mask 
            
            self.biggestContour = max(self.cnts, key=cv2.contourArea); #Finds the biggest contour of the binary mask
            self.area = cv2.contourArea(self.biggestContour); #Computes the area of the biggest object from the binary mask
            
            if self.area > self._args["segmentation"]["minAreaMask"] and self.area < self._args["segmentation"]["maxAreaMask"] : #Future computation is done only if the area of the detected object is meaningful
                
                ((self.x,self.y), self.radius) = cv2.minEnclosingCircle(self.biggestContour);
                self.M = cv2.moments(self.biggestContour); #Computes the Moments of the detected object
                self.center = (int(self.M["m10"] / self.M["m00"]), int(self.M["m01"] / self.M["m00"])); #Computes the Centroid of the detected object
                
                self.StorePosition(); #Stores the position and area of the detected object
                
            else : #If the area of the detected object is too small...
                
                self.Error(); #Specify that no object was detected
                self.StorePosition(); #Stores old position and area
                
        else : #If no contour was detected...
            
            self.Error(); #Specify that no object was detected
            self.area = 0; #Resets the area size to 0
            self.StorePosition(); #Stores old position and area
            
        self.ComputeDistanceTraveled();
            
            
    def RunSegmentationCotton(self) :
        
        #Filtering the ROI from noise
        #----------------------------------------------------------------------------------------------------------------------------------

        self.maskCotton = cv2.inRange(self.blur, self._args["segmentation"]["threshMinCotton"], self._args["segmentation"]["threshMaxCotton"]); #Thresholds the image to binary
        self.openingCotton = cv2.morphologyEx(self.maskCotton,cv2.MORPH_OPEN,self._args["segmentation"]["kernel"], iterations = 3); #Applies opening operation to the mask for dot removal
        self.closingCotton = cv2.morphologyEx(self.openingCotton,cv2.MORPH_CLOSE,self._args["segmentation"]["kernel"], iterations = 3); #Applies closing operation to the mask for large object filling
        self.cntsCotton = cv2.findContours(self.closingCotton.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]; #Finds the contours of the image to identify the meaningful object
    
        self.cntNest = None; #Variable that determines whether a nest is detected or not
        self.cntLarge = [];
        self.largeObjects = 0; #Variable that holds the number of detected large cotton objects
        self.smallObjects = 0; #Variable that holds the number of detected small cotton objects
        
        self.croppedRegisteredDepth = self.registeredDepth[self.upLeftY:self.lowRightY,self.upLeftX:self.lowRightX];
        self.bitwiseDepthCottonMask = cv2.bitwise_and(self.closingCotton, self.croppedRegisteredDepth[:,:,0]);
        self.averagePixelIntensity = self.bitwiseDepthCottonMask[np.nonzero(self.bitwiseDepthCottonMask)];
        
        try :
            self.averagePixelIntensity = int(np.mean(self.averagePixelIntensity));
        except :
            self.averagePixelIntensity = 0;
        
        self._cottonAveragePixelIntensities.append(self.averagePixelIntensity);
        
        for i in range(len(self.cntsCotton)): #for every detected object as a cotton piece...

            area = cv2.contourArea(self.cntsCotton[i]); #Computes its area
            
            if area >= self._args["segmentation"]["minCottonSize"] : #If the area in bigger than a certain threshold
                
                self.largeObjects+=1; #Adds the object to the count of large cotton pieces
                self.cntLarge.append(i);
                
            else : #If the area in smaller than a certain threshold
                
                self.smallObjects+=1; #Adds the object to the count of small cotton pieces
                
            if area >= self._args["segmentation"]["nestCottonSize"] : #If the area has a size of a nest !
                
                self.cntNest = i; #Sets the self.cntNest variable to hold the position of the nest contour
                
    def RegisterDepth(self) :
        
        self.depthFrame = cv2.resize(self.DEPTHFrame,(0,0), fx=self._resizingFactorRegistration, fy=self._resizingFactorRegistration);
            
        self.registeredDepth = np.zeros([self._H_RGB,self._W_RGB,3],dtype=np.uint8);
        
        self.blend = cv2.addWeighted(self.depthFrame[self.start_X_foreground:self.end_X_foreground,self.start_Y_foreground:self.end_Y_foreground,:],
                        1,
                        self.registeredDepth[self.start_X:self.end_X,self.start_Y:self.end_Y,:],
                        0,
                        0,
                        self.registeredDepth);
                                
        self.registeredDepth[self.start_X:self.end_X,self.start_Y:self.end_Y,:] = self.blend; 
            
    def CreateDisplay(self) :
            
        self.metaDataDisplay = np.zeros((self._H_RGB,self._metaDataCanvasSize,3), np.uint8); #Creates a canvas to write useful info
        self.WriteDisplay(self.metaDataPos1);
        
        #----------------------------------------------------------------------------------------------------------------------------------
        #Draw text, contours, mouse tracker on the frames
            
        self.cloneFrame[self.upLeftY:self.upLeftY+self.maskDisplay.shape[0],\
                        self.upLeftX:self.upLeftX+self.maskDisplay.shape[1]] = self.maskDisplay; 
                        
        cv2.circle(self.cloneFrame, self.correctedCenter, self.centerSize, (0, 0, 255), self.centerThickness); #Draws a the object Centroid as a point
 
        cv2.rectangle(self.cloneFrame, (self.upLeftX,self.upLeftY), (self.lowRightX,self.lowRightY),(255,0,0), self.contourThickness); #Displays the ROI square on the image
        
        self.hStack = np.hstack((self.metaDataDisplay, self.cloneFrame)); #Generates a horizontal stacked image with useful info and stream
        
        self.hStack = cv2.resize(self.hStack, (0,0),\
                                 fx = 1./self._args["saving"]["resizeTracking"],\
                                 fy = 1./self._args["saving"]["resizeTracking"]);
        
    def WriteDisplay(self,pos) :
        
        #Writes the fisrt information
        #----------------------------------------------------------------------
        cv2.putText(self.metaDataDisplay,
                    "Tracking Mouse : {0}".format(self._args["main"]["mouse"]),
                    (8,pos),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.textFontSize,
                    (255,255,255),
                    self.textFontThickness); #Displays the mouse name
        
        #Writes the second information
        #----------------------------------------------------------------------
#        cv2.putText(self.metaDataDisplay,
#                    "mask size : {0}px2".format(int(self.area)),
#                    (8,pos),
#                    cv2.FONT_HERSHEY_SIMPLEX,
#                    self.textFontSize,
#                    (255,255,255),
#                    self.textFontThickness); #Displays the mask size in real-time
        
        cv2.putText(self.metaDataDisplay,
                    "dist : {0}cm".format(str("%.2f" % self._distance)),
                    (8,pos+70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.textFontSize,
                    (255,255,255),
                    self.textFontThickness); #Displays the distance traveled by the mouse in real-time
        
        #Writes the third information
        #----------------------------------------------------------------------
        cv2.putText(self.metaDataDisplay,
                    "speed : {0}cm/s".format(str("%.2f" % self.realTimeSpeed)),
                    (8,pos+70*2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.textFontSize,
                    (255,255,255),
                    self.textFontThickness); #Displays the mouse speed in real-time
                    
        if self._args["saving"]["segmentCotton"] : 
        
            #Writes the fourth information
            #----------------------------------------------------------------------   
            cv2.putText(self.metaDataDisplay,
                        '{0} obj; height : {1}'.format(self.largeObjects,self.averagePixelIntensity),
                        (8,pos+70*3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.textFontSize,
                        (255,255,255),
                        self.textFontThickness); #Displays the number of cotton objects detected   
                    
            #Writes the fifth information
            #----------------------------------------------------------------------
            
            #If a nest is detected
            if self.cntNest != None :
                
                cv2.putText(self.metaDataDisplay,
                            'Nest Detected',
                            (8,pos+70*4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.textFontSize,
                            (0,255,0),
                            self.textFontThickness); #Displays the current status of nest detection
                            
                cv2.drawContours(self.maskDisplay, self.cntsCotton, self.cntNest, (255,255,0), self.contourThickness); #Draws the contour of the nest
                
                self.largeContours = [self.cntsCotton[p] for p in self.cntLarge];
                cv2.drawContours(self.maskDisplay, self.largeContours, -1, (0,255,0), self.contourThickness);
                
    #            for cnt in self.cntLarge :
    #                if cnt != self.cntNest :
    #                    cv2.drawContours(self.maskDisplay, self.cntsCotton, cnt, (0,255,0), self.contourThickness); #Draws all the other cotton contours that are not the nest
                
    #            for cnt in range(len(self.cntsCotton)) :
    #                if cnt != self.cntNest :
    #                    cv2.drawContours(self.maskDisplay, self.cntsCotton, cnt, (0,255,0), self.contourThickness); #Draws all the other cotton contours that are not the nest
            
            #If no nest is detected
            
            else : 
                
                cv2.putText(self.metaDataDisplay,
                            'No Nest',
                            (8,pos+70*4),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            self.textFontSize,
                            (0,0,255),
                            self.textFontThickness); #Displays the current status of nest detection
                            
                self.largeContours = [self.cntsCotton[p] for p in self.cntLarge];
                cv2.drawContours(self.maskDisplay, self.largeContours, -1, (0,255,0), self.contourThickness);
                
    #            for cnt in self.cntLarge :
    #                cv2.drawContours(self.maskDisplay, self.cntsCotton, cnt, (0,255,0), self.contourThickness); #Draws all the other cotton contours that are not the nest
                #cv2.drawContours(self.maskDisplay, self.cntsCotton, -1, (0,255,0), self.contourThickness); #Draws all the cotton contours that are not the nest
        
        
        #Writes the sixth information
        #----------------------------------------------------------------------
        
        #If a mouse is detected
        if self.area > self._args["segmentation"]["minAreaMask"] and self.area < self._args["segmentation"]["maxAreaMask"] :
        
            cv2.putText(self.metaDataDisplay,
                        "Mouse Detected",
                        (8,pos+70*5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.textFontSize,
                        (0,255,0),
                        self.textFontThickness); #Displays the current status of mouse detection
                        
            cv2.drawContours(self.maskDisplay, [self.biggestContour], 0, (0,0,255), self.contourThickness); #Draws the contour of the detected object on the image
        
        #If no mouse is detected
        else :
        
            cv2.putText(self.metaDataDisplay,
                        "No Mouse",
                        (8,pos+70*5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.textFontSize,
                        (0,0,255),
                        self.textFontThickness); #Displays the current status of mouse detection
        
    def SaveTracking(self) :
        
        if self._startSaving :
        
            self.videoWriter.write(self.hStack);
            
        if self._startSavingMask :
            
            self.bitwiseDepthCottonMaskSave = cv2.cvtColor(self.bitwiseDepthCottonMask,cv2.COLOR_GRAY2RGB);
            self.depthMaskWriter.write(self.bitwiseDepthCottonMaskSave);
            
    
    def ComputeDistanceTraveled(self) :
        
        if len(self.realTimePosition) == 2 and not None in self.realTimePosition : #Runs only if two positions are available and if previous or current position are not None
                
                self.realTimeSpeed = sqrt((self.realTimePosition[1][0]-self.realTimePosition[0][0])**2+\
                                      (self.realTimePosition[1][1]-self.realTimePosition[0][1])**2)/self.distanceRatio; #Computes distance
                                          
                self._distances.append(self.realTimeSpeed);
    
                if self.realTimeSpeed >= self._args["segmentation"]["minDist"] : #If the distance is higher than the minimal distance (filtering for movement noise)
                    
                    self._distance += (self.realTimeSpeed); #Adds the value to the cumulative distance varaible
           
            
    def UpdatePosition(self) :

        if len(self.realTimePosition) == 0 :
            
            self.realTimePosition.append(self.center);
            
        elif len(self.realTimePosition) == 1 :
            
            self.realTimePosition.append(self.center);
            
        elif len(self.realTimePosition) == 2 :
            
            self.realTimePosition[0] = self.realTimePosition[1];
            self.realTimePosition[1] = self.center;
         
            
    def StorePosition(self) :
        
        self._positions.append(self.center);
        self._maskAreas.append(self.area);
        
        if self.center != None :
            self.UpdatePosition();
            self.correctedCenter = (self.center[0]+self.upLeftX,\
                                    self.center[1]+self.upLeftY);
          
            
    def Error(self) : 
        
        self._errors += 1;
        
        if self._args["display"]["showStream"] :
            
            if self._errors == 1 :
                
                print("[WARNING] No contour detected, assuming old position !");
                
            elif self._errors % 100 == 0 :
                
                print("[WARNING] No contour detected, assuming old position !");
             
                
    def ReturnTracking(self) :
        
        if self._args["display"]["showStream"] :
            
            try :
                
                return self.hStack;
            
            except :
                
                return [];
        else :
            
            return [];
   
         
def TopTracker(Tracker,**kwargs) :
    
    print("\n");
    utils.PrintColoredMessage("#########################################################","darkgreen");
    utils.PrintColoredMessage("[INFO] Starting segmentation for mouse {0}".format(kwargs["main"]["mouse"]),"darkgreen");
    utils.PrintColoredMessage("#########################################################","darkgreen");
                              
    Tracker._Start = time.time();

    for capture in kwargs["main"]["capturesRGB"] :
        
        try :
    
            while(True):

                #Charges a new frame and runs the segmentation
                #----------------------------------------------------------------------
                Tracker.Main();
                
                if not Tracker._Stop :
                
                    #If the tracking has to be saved
                    #----------------------------------------------------------------------
                    if kwargs["saving"]["saveStream"] :
                        
                        Tracker.SaveTracking();
                    
                    #If the tracking has to be displayed
                    #----------------------------------------------------------------------
                    if kwargs["display"]["showStream"] :
                        
                        segmentation = Tracker.ReturnTracking();
                        
                        if segmentation != [] :
                            
                            cv2.imshow('segmentation',segmentation);
                            
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break;
                    
                    #If the video is not empty
                    #----------------------------------------------------------------------
                    if not Tracker._tEnd[Tracker.videoNumber] == 0 :
                        
                        #VERBOSE
                        #Runs only every 10 minutes of the video being analyzed
                        #----------------------------------------------------------------------------------------------------------------------------------
                        if Tracker.frameNumber%(600*int(Tracker._framerate)) == 0 :
                            
                            print('\n'); 
                            utils.PrintColoredMessage('Loaded and analyzed : '+str(Tracker.frameNumber)+'/'+str(int(Tracker._tEnd[Tracker.videoNumber]*Tracker._framerate))+\
                                ' = '+(str(int(float(Tracker.frameNumber)/float(Tracker._tEnd[Tracker.videoNumber]*Tracker._framerate)*100)))\
                                +'% frames from video n°'+str(Tracker.videoNumber)+'/'+str(Tracker._nVideos), "darkgreen");
                                                      
                            utils.PrintColoredMessage(utils.PrintLoadingBar(int(float(Tracker.frameNumber)/float(Tracker._tEnd[Tracker.videoNumber]*Tracker._framerate)*100)),"darkgreen");
                        
                        #Runs only if the video is finished
                        #----------------------------------------------------------------------------------------------------------------------------------
                        if Tracker.frameNumber == int(Tracker._tEnd[Tracker.videoNumber]*Tracker._framerate) :
                
                            print('\n'); 
                            utils.PrintColoredMessage('#########################################################',"darkgreen");
                            utils.PrintColoredMessage("[INFO] Video {0} for mouse {1} has been successfully analyzed".format(str(Tracker.videoNumber),Tracker._mouse),"darkgreen");
                            utils.PrintColoredMessage('#########################################################',"darkgreen");
                                                      
                            if kwargs["main"]["playSound"] :
                                utils.PlaySound(2,params.sounds['Purr']); #Plays sound when code finishes
                            
                            Tracker.videoNumber += 1; #Increments videoNumber variable to keep track which video is being processed
                            Tracker.frameNumber = 0;
                            break;
                            
                else :
                    
                    print('\n'); 
                    utils.PrintColoredMessage('#########################################################',"darkgreen");
                    utils.PrintColoredMessage("[INFO] Video {0} for mouse {1} has been successfully analyzed".format(str(Tracker.videoNumber),Tracker._mouse),"darkgreen");
                    utils.PrintColoredMessage('#########################################################',"darkgreen");
                                              
                    if kwargs["main"]["playSound"] :
                        utils.PlaySound(2,params.sounds['Purr']); #Plays sound when code finishes
                        
                    Tracker.videoNumber += 1; #Increments videoNumber variable to keep track which video is being processed
                    Tracker.frameNumber = 0;
                    
                    break;
                        
        except KeyboardInterrupt :
            
            pass;
              
        if kwargs["display"]["showStream"] :       
            
            cv2.destroyAllWindows();
        
        if kwargs["saving"]["saveStream"] :
            
            Tracker.videoWriter.release();
            
        if kwargs["saving"]["saveCottonMask"] :
            
            Tracker.depthMaskWriter.release();
            
        Tracker._End = time.time();
        
        SaveTracking(Tracker,**kwargs);


def SaveTracking(Tracker,**kwargs) :
        
    first = 0;

    for point in Tracker._positions :
        if point == None :
            first += 1;
    
    for n,point in enumerate(Tracker._positions) :
        if point == None :
            Tracker._positions[n] = Tracker._positions[first];
            
    metaDataString = "Segmentation_MetaData_{0}".format(Tracker._mouse);
    metaDataFile = os.path.join(kwargs["main"]["workingDir"],'{0}.xls'.format(metaDataString));
    
    metaData = xlwt.Workbook();
    sheet = metaData.add_sheet("MetaData");
    
    sheet.write(0, 0, "Mouse");
    sheet.write(0, 1, kwargs["main"]["mouse"]);
    
    sheet.write(1, 0, "refPt");
    sheet.write(1, 1, "[({0},{1}),({2},{3})]".format(Tracker._refPt[0][0],\
                        Tracker._refPt[0][1],Tracker._refPt[1][0],Tracker._refPt[1][1]));
    
    sheet.write(2, 0, "ThreshMinMouse");
    sheet.write(2, 1, "[{0},{1},{2}]".format(kwargs["segmentation"]["threshMinMouse"][0],\
                kwargs["segmentation"]["threshMinMouse"][1],kwargs["segmentation"]["threshMinMouse"][2]));
    
    sheet.write(3, 0, "ThreshMaxMouse");
    sheet.write(3, 1, "[{0},{1},{2}]".format(kwargs["segmentation"]["threshMaxMouse"][0],\
                kwargs["segmentation"]["threshMaxMouse"][1],kwargs["segmentation"]["threshMaxMouse"][2]));
    
    sheet.write(4, 0, "threshMinCotton");
    sheet.write(4, 1, "[{0},{1},{2}]".format(kwargs["segmentation"]["threshMinCotton"][0],\
                kwargs["segmentation"]["threshMinCotton"][1],kwargs["segmentation"]["threshMinCotton"][2]));
    
    sheet.write(5, 0, "threshMaxCotton");
    sheet.write(5, 1, "[{0},{1},{2}]".format(kwargs["segmentation"]["threshMaxCotton"][0],\
                kwargs["segmentation"]["threshMaxCotton"][1],kwargs["segmentation"]["threshMaxCotton"][2]));
    
    sheet.write(6, 0, "ElapsedTime");
    sheet.write(6, 1, Tracker._End-Tracker._Start);
    
    sheet.write(7, 0, "Errors");
    sheet.write(7, 1, Tracker._errors);
    
    metaData.save(metaDataFile);
    
    np.save(os.path.join(kwargs["main"]["workingDir"],'Data_'+str(Tracker._mouse)+'_refPt.npy'),Tracker._refPt);
    np.save(os.path.join(kwargs["main"]["workingDir"],'Data_'+str(Tracker._mouse)+'_Points.npy'),Tracker._positions);
    np.save(os.path.join(kwargs["main"]["workingDir"],'Data_'+str(Tracker._mouse)+'_Areas.npy'),Tracker._maskAreas);
    np.save(os.path.join(kwargs["main"]["workingDir"],'Data_'+str(Tracker._mouse)+'_Distances.npy'),Tracker._distances);
    np.save(os.path.join(kwargs["main"]["workingDir"],'Data_'+str(Tracker._mouse)+'_CottonPixelIntensities.npy'),Tracker._cottonAveragePixelIntensities);
                