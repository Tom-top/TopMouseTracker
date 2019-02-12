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
from math import sqrt;

import TopMouseTracker.Parameters as params;
import TopMouseTracker.Utilities as utils;
import TopMouseTracker.IO as IO;

class TopMouseTracker():
    
    '''Class that holds all parameters usefull for the segmentation and launches the tracker
    
    Params :
        **kwargs (dict) : dictionnary with all the parameters useful for the tracker
    '''
    
    def __init__(self,**kwargs) :
        
        #General variables
        #----------------------------------------------------------------------
        self._args = kwargs; #Arguments
        self._mouse = self._args["main"]["mouse"]; #Name of the mouse
        self._testFrameRGB = self._args["main"]["testFrameRGB"].copy(); #A test RGB frame for ROI setup
        self._H, self._W, self._D = self._testFrameRGB.shape; #Height, Width and Depth of the RGB frames
        self.metaDataSize = 750;
        self._registrationResize = 2.95 #Parameters for resizing of the depth image for registration
        
        for file in os.listdir(self._args["main"]["resultDir"]) :
            
            path2File = os.path.join(self._args["main"]["resultDir"],file);
            
            if fnmatch.fnmatch(file, 'Mice_Video_Info.xlsx'): 

                videoInfoWorkbook = pd.read_excel(path2File,header=None); #Load video info excel sheet
                
            if fnmatch.fnmatch(file, 'MetaData*.xlsx'): 
                
                metaDataWorkbook = pd.read_excel(path2File,header=None); #Load video info excel sheet
        
        videoInfo = videoInfoWorkbook.as_matrix(); #Transforms it into a matrix
        
        for line in videoInfo :
            
            if not np.isnan(line[0]) : 
                
              if str(int(line[0])) == self._args["main"]["mouse"] :
                  
                  self._tStart = int(line[1]); #Moment at which the cotton is added (s)
                  self._tStartBehav = int(line[2]); #Moment at which the mouse start nest-building (s)
                  self._tEnd = [int(line[3]),int(line[4]),\
                               int(line[5]),int(line[6])]; #Length of each video of the experiment (max 4 videos)
                                
        self._nVideos = 0; #Variable holding the number of videos to be analyzed
        
        for i in self._tEnd :
            if i != 0 :
                self._nVideos+=1;
        
        metaData = metaDataWorkbook.as_matrix(); #Transforms it into a matrix
        
        for line in metaData :
            
            head = str(line[0]);
            
            if head == "TimeStamp" :
                
                self._date = line[1];
                
            elif head == "Elapsed_Time" :
                
                self._elapsedTime = line[1];
                
            elif head == "Framerate" :
                
                self._framerate = line[1];
                
            elif head == "nFrames" :
                
                self._nFrames = line[1];

        #Global tracking variables
        #----------------------------------------------------------------------
        self._positions = []; #Position of the mouse every s/frameRate
        self._maskAreas = []; #Size of the mouse mask every s/frameRate
        self._distance = 0.; #Cumulative distance traveled by the mouse
        self._errors = 0; #Counter for the number of times that the trackers fails to segment the animal
        self._cottonContours = []; #List containing all the depth information of segmented objects coming from the cotton mask
        
        #Real-Time tracking variables
        #----------------------------------------------------------------------
        self.videoNumber = 0;
        self.frameNumber = 0;
        self.realTimePosition = []; #Position of the mouse in real time
        self.realTimeSpeed = 0.; #Speed of the mouse in real time
        self.center = None; #Centroid (x,y) of the mouse binary mask in real time
        self.correctedCenter = None; #Corrected centroid (x,y) of the mouse binary mask in real time
        
        #Saving variables
        #----------------------------------------------------------------------
        self.time = time.localtime(time.time());
        self.savingTrigger = False;
        
        if self._args["saving"]["saveStream"] :
            
            self.videoString = "Tracking_{0}.avi".format(self._mouse);
            self.videoWriter = cv2.VideoWriter(os.path.join(self._args["main"]["workingDir"],\
                                self.videoString), self._args["fourcc"], self._framerate,\
                                (self._W+self.metaDataSize,self._H));
        
        
    def SetROI(self) :
        
        '''Function to select and set ROI for analysis
        
        Output : _refPt (tuple) : tuple of top-right and low-left coordinates of the ROI
        '''
        
        self._refPt = IO.CroppingROI(self._args["testFrame"][0].copy()).roi(); #Defining the ROI for segmentation
        
        self.upLeftX = int(self._refPt[0][0]); #Defines the Up Left ROI corner X coordinates
        self.upLeftY = int(self._refPt[0][1]); #Defines the Up Left ROI corner Y coordinates
        self.lowRightX = int(self._refPt[1][0]); #Defines the Low Right ROI corner X coordinates
        self.lowRightY = int(self._refPt[1][1]); #Defines the Low Right ROI corner Y coordinates
        
        
    def Main(self):
        
        #Get frame from capture
        #----------------------------------------------------------------------
        self.RGBFrame = next(self._args["capturesRGB"][self.videoNumber]); #Reads the following frame from the video capture
        self.DEPTHFrame = next(self._args["capturesDEPTH"][self.videoNumber]); #Reads the following frame from the video capture
        
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
        self.RunSegmentationCotton(); #Runs the cotton segmentation on the ROI
        
        if self._args["showStream"] or self._args["saveStream"] :

            self.CreateDisplay();
            
            
    def RunSegmentationMouse(self) :
        
        
        #Selects only the ROI part of the image for future analysis
        #----------------------------------------------------------------------------------------------------------------------------------
        
        self.cloneFrame = self.RGBFrame.copy(); #[DISPLAY ONLY] Creates a clone frame for display purposes
        
        self.distanceRatio = (abs(self.upLeftX-self.lowRightX)/self._args["cageLength"]+\
                              abs(self.upLeftY-self.lowRightY)/self._args["cageWidth"])/2; #Defines the resizing factor for the cage
        
        self.croppedFrame = self.RGBFrame[self.upLeftY:self.lowRightY,self.upLeftX:self.lowRightX]; #Crops the initial frame to the ROI

        self.maskDisplay = cv2.cvtColor(self.croppedFrame, cv2.COLOR_BGR2RGB); #[DISPLAY ONLY] Changes the croppedFrame to RGB for display purposes
        self.colorFrame = cv2.cvtColor(self.cloneFrame, cv2.COLOR_BGR2RGB); #[DISPLAY ONLY] Changes the Frame to RGB for display purposes
        
        #Filtering the ROI from noise
        #----------------------------------------------------------------------------------------------------------------------------------
        
        self.hsvFrame = cv2.cvtColor(self.croppedFrame, cv2.COLOR_BGR2HSV); #Changes the croppedFrame LUT to HSV for segmentation
        self.blur = cv2.blur(self.hsvFrame,(5,5)); #Applies a Gaussian Blur to smoothen the image
        self.maskMouse = cv2.inRange(self.blur, self._args["threshMinMouse"], self._args["threshMaxMouse"]); #Thresholds the image to binary
        self.openingMouse = cv2.morphologyEx(self.maskMouse,cv2.MORPH_OPEN,self._args["kernel"], iterations = 1); #Applies opening operation to the mask for dot removal
        self.closingMouse = cv2.morphologyEx(self.openingMouse,cv2.MORPH_CLOSE,self._args["kernel"], iterations = 1); #Applies closing operation to the mask for large object filling
        self.cnts = cv2.findContours(self.closingMouse.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]; #Finds the contours of the image to identify the meaningful object
        
        #Finding the Mouse
        #----------------------------------------------------------------------------------------------------------------------------------
        
        if self.cnts != [] : #If a contour is found in the binary mask 
            
            self.biggestContour = max(self.cnts, key=cv2.contourArea); #Finds the biggest contour of the binary mask
            self.area = cv2.contourArea(self.biggestContour); #Computes the area of the biggest object from the binary mask
            
            if self.area > self._args["minAreaMask"] and self.area < self._args["maxAreaMask"] : #Future computation is done only if the area of the detected object is meaningful
                
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
            
            
    def RunSegmentationCotton(self) :
        
        #Filtering the ROI from noise
        #----------------------------------------------------------------------------------------------------------------------------------

        self.maskCotton = cv2.inRange(self.blur, self._args["threshMinCotton"], self._args["threshMaxCotton"]); #Thresholds the image to binary
        self.openingCotton = cv2.morphologyEx(self.maskCotton,cv2.MORPH_OPEN,self._args["kernel"], iterations = 2); #Applies opening operation to the mask for dot removal
        self.closingCotton = cv2.morphologyEx(self.openingCotton,cv2.MORPH_CLOSE,self._args["kernel"], iterations = 1); #Applies closing operation to the mask for large object filling
        self.cntsCotton = cv2.findContours(self.closingCotton.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[-2]; #Finds the contours of the image to identify the meaningful object
        
        self.objectIntensities = []; #List that will hold all the information about the detected contours
        self.cntNest = None; #Variable that determines whether a nest is detected or not
        self.largeObjects = 0; #Variable that holds the number of detected large cotton objects
        self.smallObjects = 0; #Variable that holds the number of detected small cotton objects
        
        for i in range(len(self.cntsCotton)): #for every detected object as a cotton piece...

            area = cv2.contourArea(self.cntsCotton[i]); #Computes its area
            canvas = np.zeros_like(self.registeredDepth); #Creates a canvas to draw the individual objects
            cv2.drawContours(canvas, self.cntsCotton, i, color=255, thickness=-1); #Draws individual contours on the canvas...

            pts = np.where(canvas == 255); #checks where the contours are on the canvas
            self.objectIntensities.append(self.registeredDepth[pts[0], pts[1]]); #Appends the information about each object (pixel intensities) to the sink list = self.objectIntensities
            
            if area >= self._args["segmentation"]["minCottonSize"] : #If the area in bigger than a certain threshold
                
                self.largeObjects+=1; #Adds the object to the count of large cotton pieces
                
            else : #If the area in smaller than a certain threshold
                
                self.smallObjects+=1; #Adds the object to the count of small cotton pieces
                
            if area >= self._args["segmentation"]["nestCottonSize"] : #If the area has a size of a nest !
                
                self.cntNest = i; #Sets the self.cntNest variable to hold the position of the nest contour
            
        self.averageObjectIntensities = [[sum(x)/len(x) for x in element] for element in self.objectIntensities]; #Computes the average intensity of each pixel of each detected object
        self.averageObjectIntensities = [item for sublist in self.averageObjectIntensities for item in sublist];
 
        self._cottonContours.append(self.objectIntensities); #Appends the list with all the information about the detected contours to the main list holding this info for each frame
                
    def RegisterDepth(self) :
        
        self.depthFrame = cv2.resize(self.DEPTHFrame,(0,0),fx = self._registrationResize,fy = self._registrationResize)
        self.depthFrame = cv2.cvtColor(self.depthFrame, cv2.COLOR_GRAY2BGR);
        
        self._HDepth,self._WDepth = self.depthFrame.shape[0],self.depthFrame.shape[1]
        
        start_X = 0;
        end_X = start_X+self._HDepth;
        start_Y = 290;
        end_Y = start_Y+self._WDepth;
        
        if end_X >= self._H :
            end_X_foreground = self._H-start_X;
        else :
            end_X_foreground = self._H;
            
        if end_Y >= self._W :
            end_Y_foreground = self._W-start_Y;
        else : 
            end_Y_foreground = self._W;
        
        start_X_foreground = 75;
        start_Y_foreground = 0;
        
        if start_X_foreground != 0 :
            end_X = end_X-start_X_foreground
            end_X_foreground = end_X_foreground+start_X_foreground
        if start_Y_foreground != 0 :
            end_Y = end_Y-start_Y_foreground
            end_Y_foreground = end_Y_foreground+start_Y_foreground
            
        self.registeredDepth = np.zeros([self._H,self._W,3],dtype=np.uint8);
        
        self.blend = cv2.addWeighted(self.depthFrame[start_X_foreground:end_X_foreground,start_Y_foreground:end_Y_foreground,:],
                        1,
                        self.registeredDepth[start_X:end_X,start_Y:end_Y,:],
                        0,
                        0,
                        self.registeredDepth);
                                
        self.registeredDepth[start_X:end_X,start_Y:end_Y,:] = self.blend; 
            
    def CreateDisplay(self) :
        
        self.textFontSize = 1.5;
        self.textFontThickness = 3;
        self.contourThickness = 2;
        self.centerSize = 3;
        self.centerThickness = 5;
        
        self.metaDataPos1 = 50;
        self.metaDataPos2 = 600;
        
        if self.source == None :
            
            self.metaDataDisplay = np.zeros((self._H,750,3), np.uint8); #Creates a canvas to write useful info
            self.WriteDisplay(self.metaDataPos1);
            
        else :
            
            self.metaDataDisplay = self.source.metaDataDisplay; #[DISPLAY ONLY] Creates a clone frame for display purposes
            self.WriteDisplay(self.metaDataPos2);
        
        self.ComputeDistanceTraveled();
        
        #----------------------------------------------------------------------------------------------------------------------------------
        #Draw text, contours, mouse tracker on the frames
            
        self.cloneFrame[self.upLeftY:self.upLeftY+self.maskDisplay.shape[0],\
                        self.upLeftX:self.upLeftX+self.maskDisplay.shape[1]] = self.maskDisplay; 
                        
        cv2.circle(self.cloneFrame, self.correctedCenter, self.centerSize, (0, 0, 255), self.centerThickness); #Draws a the object Centroid as a point
 
        cv2.rectangle(self.cloneFrame, (self.upLeftX,self.upLeftY), (self.lowRightX,self.lowRightY),(255,0,0), self.contourThickness); #Displays the ROI square on the image
        
        self.hStack = np.hstack((self.metaDataDisplay, self.cloneFrame)); #Generates a horizontal stacked image with useful info and stream
        
    def WriteDisplay(self,pos) :
        
        cv2.putText(self.metaDataDisplay,'mask size : '+str(int(self.area))+'px2',(8,pos),cv2.FONT_HERSHEY_SIMPLEX,self.textFontSize,(255,255,255),self.textFontThickness); #Displays the mask size in real-time
        cv2.putText(self.metaDataDisplay,'dist : '+str("%.2f" % self._distance)+'cm',(8,pos+70),cv2.FONT_HERSHEY_SIMPLEX,self.textFontSize,(255,255,255),self.textFontThickness); #Displays the distance traveled by the object in real-time
        cv2.putText(self.metaDataDisplay,'speed : '+str("%.2f" % self.realTimeSpeed)+'cm/s',(8,pos+70*2),cv2.FONT_HERSHEY_SIMPLEX,self.textFontSize,(255,255,255),self.textFontThickness); #Displays the mask size in real-time
        cv2.putText(self.metaDataDisplay,'{0} large, {1} small objects'.format(self.largeObjects,self.smallObjects),(8,pos+70*3),cv2.FONT_HERSHEY_SIMPLEX,self.textFontSize,(255,255,255),self.textFontThickness); #Displays the number of cotton objects detected
        
        if self.cntNest != None :
            
            cv2.putText(self.metaDataDisplay,'Nest Detected'.format(self.averageObjectIntensities),(8,pos+70*4),cv2.FONT_HERSHEY_SIMPLEX,self.textFontSize,(0,255,0),self.textFontThickness); #Displays the average Height of cotton
            cv2.drawContours(self.maskDisplay, self.cntsCotton, self.cntNest, (255,255,0), self.contourThickness);
            
            for cnt in range(len(self.cntsCotton)) :
                if cnt != self.cntNest :
                    cv2.drawContours(self.maskDisplay, self.cntsCotton, cnt, (0,255,0), self.contourThickness);
            
        else : 
            
            cv2.putText(self.metaDataDisplay,'No Nest'.format(self.averageObjectIntensities),(8,pos+70*4),cv2.FONT_HERSHEY_SIMPLEX,self.textFontSize,(0,0,255),self.textFontThickness); #Displays the average Height of cotton
            cv2.drawContours(self.maskDisplay, self.cntsCotton, -1, (0,255,0), self.contourThickness);
        
        if self.area > self._args["segmentation"]["minAreaMask"] and self.area < self._args["segmentation"]["maxAreaMask"] :
        
            cv2.drawContours(self.maskDisplay, [self.biggestContour], 0, (0,0,255), self.contourThickness); #Draws the contour of the detected object on the image
            cv2.putText(self.metaDataDisplay,"Mouse Detected",(8,pos+70*5),cv2.FONT_HERSHEY_SIMPLEX,self.textFontSize,(0,255,0),self.textFontThickness); #Displays the detection status
        
        else :
        
            cv2.putText(self.metaDataDisplay,"No Mouse",(8,pos+70*5),cv2.FONT_HERSHEY_SIMPLEX,self.textFontSize,(0,0,255),self.textFontThickness); #Displays the detection status
        
    def SaveTracking(self) :

        if self.savingTrigger :
        
            self.videoWriter.write(self.hStack[:,:,0]);
    
    
    def ComputeDistanceTraveled(self) :
        
        if len(self.realTimePosition) == 2 and not None in self.realTimePosition : #Runs only if two positions are available and if previous or current position are not None
                
                self.realTimeSpeed = sqrt((self.realTimePosition[1][0]-self.realTimePosition[0][0])**2+\
                                      (self.realTimePosition[1][1]-self.realTimePosition[0][1])**2)/self.distanceRatio; #Computes distance
    
                if self.realTimeSpeed >= self._args["minDist"] : #If the distance is higher than the minimal distance (filtering for movement noise)
                    
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
        
        if self._args["showStream"] :
            
            if self._errors == 1 :
                
                print("[WARNING] No contour detected, assuming old position !");
                
            elif self._errors % 100 == 0 :
                
                print("[WARNING] No contour detected, assuming old position !");
                self._errors = 0;
             
                
    def ReturnTracking(self) :
        
        if self._args["showStream"] :
            
            try :
                
                return self.hStack;
            
            except :
                
                return [];
        else :
            
            return [];
   
         
def TopTracker(data,**kwargs) :
    
    Tracker = data; #Initialize all variables

    for capture in kwargs["capturesRGB"] :
    
        #Starts segmentation#W+700
        while(True):

            Tracker.Main();
            tracker = Tracker.ReturnTracking();
            Tracker.SaveTracking();
            
            #Runs only if the videoLength is not null
            if not Tracker._tEnd[Tracker.videoNumber] == 0 :
                
                #----------------------------------------------------------------------------------------------------------------------------------
                #Runs only every 10 minutes of the video being analyzed
                if Tracker.frameNumber%(600*Tracker._framerate) == 0 :
                    
                    utils.PrintColoredMessage('Loaded and analyzed : '+str(Tracker.frameNumber)+'/'+str(int(Tracker._tEnd[Tracker.videoNumber]*Tracker._framerate))+\
                        ' = '+(str(int(float(Tracker.frameNumber)/float(Tracker._tEnd[Tracker.videoNumber]*Tracker._framerate)*100)))\
                        +'% frames from video n°'+str(Tracker.videoNumber)+'/'+str(Tracker._nVideos), "darkgreen");
                    utils.PrintColoredMessage(utils.PrintLoadingBar(int(float(Tracker.frameNumber)/float(Tracker._tEnd[Tracker.videoNumber]*Tracker._framerate)*100)),"darkgreen");
                        
                #----------------------------------------------------------------------------------------------------------------------------------
                if Tracker.frameNumber == int(Tracker._tEnd[Tracker.videoNumber]*Tracker._framerate) :
        
                    print('\n'); 
                    utils.PrintColoredMessage('#########################################################',"darkgreen");
                    utils.PrintColoredMessage("[INFO] Video {0} for mouse {1} has been successfully analyzed".format(str(Tracker.videoNumber),Tracker._mouse),"darkgreen");
                    utils.PrintColoredMessage('#########################################################',"darkgreen");
                    utils.PlaySound(2,params.sounds['Purr']); #Plays sound when code finishes
                    Tracker.videoNumber += 1; #Increments videoNumber variable to keep track which video is being processed
                    Tracker.frameNumber = 0;
                    break;

            if kwargs["showStream"] :
                
                if tracker != [] :
                    
                    cv2.imshow('tracker',tracker);
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break;
                        
        cv2.destroyAllWindows();
        
        if kwargs["saveStream"] :
            
            data.videoWriter.release();

def SaveTracking(data,directory):
    
    savingDir = directory+'Results';
    utils.CheckDirectoryExists(savingDir);
        
    first = 0;

    for point in data._positions :
        if point == None :
            first += 1;
    
    for n,point in enumerate(data._positions) :
        if point == None :
            data._positions[n] = data._positions[first];
            
    np.save(os.path.join(savingDir,'Mouse_Data_All_'+str(data._mouse)+'_Points.npy'),data._positions);
    np.save(os.path.join(savingDir,'Mouse_Data_All_'+str(data._mouse)+'_refPt.npy'),data._refPt);
    np.save(os.path.join(savingDir,'Mouse_Data_All_'+str(data._mouse)+'_Areas.npy'),data._maskAreas);
    
def SaveStream(data,directory):
    
    savingDir = directory+'Results';
    utils.CheckDirectoryExists(savingDir);
    
    name = 'Tracking_{0}.avi'.format(data._mouse);

    images = data.trackingStream;
    frame = images[0];
    _H, _W, _L = frame.shape;
    
    video = cv2.VideoWriter(os.path.join(savingDir,name), 0, 1, (_W,_H));
    
    for image in images:
        video.write(image);
    
    cv2.destroyAllWindows();
    video.release();

                